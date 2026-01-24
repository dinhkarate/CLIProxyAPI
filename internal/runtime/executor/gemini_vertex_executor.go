// Package executor provides runtime execution capabilities for various AI service providers.
// This file implements the Vertex AI Gemini executor that talks to Google Vertex AI
// endpoints using service account credentials or API keys.
package executor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	vertexauth "github.com/router-for-me/CLIProxyAPI/v6/internal/auth/vertex"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	// vertexAPIVersion aligns with current public Vertex Generative AI API.
	vertexAPIVersion = "v1"

	// veoPollingInterval is the initial interval between LRO polling requests.
	veoPollingInterval = 5 * time.Second

	// veoMaxPollingInterval is the maximum interval between polling requests (backoff cap).
	veoMaxPollingInterval = 30 * time.Second

	// veoPollingTimeout is the maximum time to wait for a Veo LRO to complete.
	veoPollingTimeout = 5 * time.Minute
)

// isImagenModel checks if the model name is an Imagen image generation model.
// Imagen models use the :predict action instead of :generateContent.
func isImagenModel(model string) bool {
	lowerModel := strings.ToLower(model)
	return strings.Contains(lowerModel, "imagen")
}

// isVeoModel checks if the model name is a Veo video generation model.
// Veo models use the :predictLongRunning action (async LRO) instead of :generateContent.
func isVeoModel(model string) bool {
	lowerModel := strings.ToLower(model)
	return strings.Contains(lowerModel, "veo")
}

// getVertexAction returns the appropriate action for the given model.
// Imagen models use "predict", Veo models use "predictLongRunning", while Gemini models use "generateContent".
func getVertexAction(model string, isStream bool) string {
	if isImagenModel(model) {
		return "predict"
	}
	if isVeoModel(model) {
		return "predictLongRunning"
	}
	if isStream {
		return "streamGenerateContent"
	}
	return "generateContent"
}

// convertImagenToGeminiResponse converts Imagen API response to Gemini format
// so it can be processed by the standard translation pipeline.
// This ensures Imagen models return responses in the same format as gemini-3-pro-image-preview.
func convertImagenToGeminiResponse(data []byte, model string) []byte {
	predictions := gjson.GetBytes(data, "predictions")
	if !predictions.Exists() || !predictions.IsArray() {
		return data
	}

	// Build Gemini-compatible response with inlineData
	parts := make([]map[string]any, 0)
	for _, pred := range predictions.Array() {
		imageData := pred.Get("bytesBase64Encoded").String()
		mimeType := pred.Get("mimeType").String()
		if mimeType == "" {
			mimeType = "image/png"
		}
		if imageData != "" {
			parts = append(parts, map[string]any{
				"inlineData": map[string]any{
					"mimeType": mimeType,
					"data":     imageData,
				},
			})
		}
	}

	// Generate unique response ID using timestamp
	responseId := fmt.Sprintf("imagen-%d", time.Now().UnixNano())

	response := map[string]any{
		"candidates": []map[string]any{{
			"content": map[string]any{
				"parts": parts,
				"role":  "model",
			},
			"finishReason": "STOP",
		}},
		"responseId":   responseId,
		"modelVersion": model,
		// Imagen API doesn't return token counts, set to 0 for tracking purposes
		"usageMetadata": map[string]any{
			"promptTokenCount":     0,
			"candidatesTokenCount": 0,
			"totalTokenCount":      0,
		},
	}

	result, err := json.Marshal(response)
	if err != nil {
		return data
	}
	return result
}

// convertToImagenRequest converts a Gemini-style request to Imagen API format.
// Imagen API uses a different structure: instances[].prompt instead of contents[].
func convertToImagenRequest(payload []byte) ([]byte, error) {
	// Extract prompt from Gemini-style contents
	prompt := ""

	// Try to get prompt from contents[0].parts[0].text
	contentsText := gjson.GetBytes(payload, "contents.0.parts.0.text")
	if contentsText.Exists() {
		prompt = contentsText.String()
	}

	// If no contents, try messages format (OpenAI-compatible)
	if prompt == "" {
		messagesText := gjson.GetBytes(payload, "messages.#.content")
		if messagesText.Exists() && messagesText.IsArray() {
			for _, msg := range messagesText.Array() {
				if msg.String() != "" {
					prompt = msg.String()
					break
				}
			}
		}
	}

	// If still no prompt, try direct prompt field
	if prompt == "" {
		directPrompt := gjson.GetBytes(payload, "prompt")
		if directPrompt.Exists() {
			prompt = directPrompt.String()
		}
	}

	if prompt == "" {
		return nil, fmt.Errorf("imagen: no prompt found in request")
	}

	// Build Imagen API request
	imagenReq := map[string]any{
		"instances": []map[string]any{
			{
				"prompt": prompt,
			},
		},
		"parameters": map[string]any{
			"sampleCount": 1,
		},
	}

	// Extract optional parameters
	if aspectRatio := gjson.GetBytes(payload, "aspectRatio"); aspectRatio.Exists() {
		imagenReq["parameters"].(map[string]any)["aspectRatio"] = aspectRatio.String()
	}
	if sampleCount := gjson.GetBytes(payload, "sampleCount"); sampleCount.Exists() {
		imagenReq["parameters"].(map[string]any)["sampleCount"] = int(sampleCount.Int())
	}
	if negativePrompt := gjson.GetBytes(payload, "negativePrompt"); negativePrompt.Exists() {
		imagenReq["instances"].([]map[string]any)[0]["negativePrompt"] = negativePrompt.String()
	}

	return json.Marshal(imagenReq)
}

// convertToVeoRequest converts a Gemini-style request to Veo API format.
// Veo API uses instances[].prompt with parameters for video generation settings.
func convertToVeoRequest(payload []byte) ([]byte, error) {
	// Extract prompt from Gemini-style contents
	prompt := ""

	// Try to get prompt from contents[0].parts[0].text
	contentsText := gjson.GetBytes(payload, "contents.0.parts.0.text")
	if contentsText.Exists() {
		prompt = contentsText.String()
	}

	// If no contents, try messages format (OpenAI-compatible)
	if prompt == "" {
		messagesText := gjson.GetBytes(payload, "messages.#.content")
		if messagesText.Exists() && messagesText.IsArray() {
			for _, msg := range messagesText.Array() {
				if msg.String() != "" {
					prompt = msg.String()
					break
				}
			}
		}
	}

	// If still no prompt, try direct prompt field
	if prompt == "" {
		directPrompt := gjson.GetBytes(payload, "prompt")
		if directPrompt.Exists() {
			prompt = directPrompt.String()
		}
	}

	if prompt == "" {
		return nil, fmt.Errorf("veo: no prompt found in request")
	}

	// Build Veo API request
	veoReq := map[string]any{
		"instances": []map[string]any{
			{
				"prompt": prompt,
			},
		},
		"parameters": map[string]any{
			"sampleCount":   1,
			"generateAudio": true, // Veo 3 requires generateAudio parameter
		},
	}

	// Extract optional parameters
	if aspectRatio := gjson.GetBytes(payload, "aspectRatio"); aspectRatio.Exists() {
		veoReq["parameters"].(map[string]any)["aspectRatio"] = aspectRatio.String()
	}
	if sampleCount := gjson.GetBytes(payload, "sampleCount"); sampleCount.Exists() {
		veoReq["parameters"].(map[string]any)["sampleCount"] = int(sampleCount.Int())
	}
	if durationSeconds := gjson.GetBytes(payload, "durationSeconds"); durationSeconds.Exists() {
		veoReq["parameters"].(map[string]any)["durationSeconds"] = int(durationSeconds.Int())
	}
	if resolution := gjson.GetBytes(payload, "resolution"); resolution.Exists() {
		veoReq["parameters"].(map[string]any)["resolution"] = resolution.String()
	}
	if generateAudio := gjson.GetBytes(payload, "generateAudio"); generateAudio.Exists() {
		veoReq["parameters"].(map[string]any)["generateAudio"] = generateAudio.Bool()
	}
	if negativePrompt := gjson.GetBytes(payload, "negativePrompt"); negativePrompt.Exists() {
		veoReq["instances"].([]map[string]any)[0]["negativePrompt"] = negativePrompt.String()
	}

	return json.Marshal(veoReq)
}

// convertVeoToGeminiResponse converts Veo API response to Gemini format
// so it can be processed by the standard translation pipeline.
// This ensures Veo models return responses in the same format as other Gemini models.
func convertVeoToGeminiResponse(data []byte, model string) []byte {
	// Veo response contains videos array with bytesBase64Encoded or gcsUri
	videos := gjson.GetBytes(data, "videos")
	if !videos.Exists() || !videos.IsArray() {
		// Try predictions format (some versions may use this)
		predictions := gjson.GetBytes(data, "predictions")
		if predictions.Exists() && predictions.IsArray() {
			return convertVeoPredictionsToGeminiResponse(predictions.Array(), model)
		}
		return data
	}

	// Build Gemini-compatible response with inlineData
	parts := make([]map[string]any, 0)
	for _, video := range videos.Array() {
		videoData := video.Get("bytesBase64Encoded").String()
		mimeType := video.Get("mimeType").String()
		if mimeType == "" {
			mimeType = "video/mp4"
		}
		if videoData != "" {
			parts = append(parts, map[string]any{
				"inlineData": map[string]any{
					"mimeType": mimeType,
					"data":     videoData,
				},
			})
		}
	}

	// Generate unique response ID using timestamp
	responseId := fmt.Sprintf("veo-%d", time.Now().UnixNano())

	response := map[string]any{
		"candidates": []map[string]any{{
			"content": map[string]any{
				"parts": parts,
				"role":  "model",
			},
			"finishReason": "STOP",
		}},
		"responseId":   responseId,
		"modelVersion": model,
		// Veo API doesn't return token counts, set to 0 for tracking purposes
		"usageMetadata": map[string]any{
			"promptTokenCount":     0,
			"candidatesTokenCount": 0,
			"totalTokenCount":      0,
		},
	}

	result, err := json.Marshal(response)
	if err != nil {
		return data
	}
	return result
}

// convertVeoPredictionsToGeminiResponse handles the predictions format for Veo responses.
func convertVeoPredictionsToGeminiResponse(predictions []gjson.Result, model string) []byte {
	parts := make([]map[string]any, 0)
	for _, pred := range predictions {
		videoData := pred.Get("bytesBase64Encoded").String()
		mimeType := pred.Get("mimeType").String()
		if mimeType == "" {
			mimeType = "video/mp4"
		}
		if videoData != "" {
			parts = append(parts, map[string]any{
				"inlineData": map[string]any{
					"mimeType": mimeType,
					"data":     videoData,
				},
			})
		}
	}

	responseId := fmt.Sprintf("veo-%d", time.Now().UnixNano())

	response := map[string]any{
		"candidates": []map[string]any{{
			"content": map[string]any{
				"parts": parts,
				"role":  "model",
			},
			"finishReason": "STOP",
		}},
		"responseId":   responseId,
		"modelVersion": model,
		"usageMetadata": map[string]any{
			"promptTokenCount":     0,
			"candidatesTokenCount": 0,
			"totalTokenCount":      0,
		},
	}

	result, err := json.Marshal(response)
	if err != nil {
		return nil
	}
	return result
}

// pollVeoOperation polls a Veo LRO until completion or timeout.
// Returns the final response data when done=true, or an error on timeout/failure.
func pollVeoOperation(ctx context.Context, cfg *config.Config, auth *cliproxyauth.Auth, operationName string, saJSON []byte) ([]byte, error) {
	// Extract location from operation name: projects/{project}/locations/{location}/operations/{id}
	parts := strings.Split(operationName, "/")
	location := "us-central1"
	for i, p := range parts {
		if p == "locations" && i+1 < len(parts) {
			location = parts[i+1]
			break
		}
	}

	baseURL := vertexBaseURL(location)
	pollURL := fmt.Sprintf("%s/%s/%s", baseURL, vertexAPIVersion, operationName)

	interval := veoPollingInterval
	deadline := time.Now().Add(veoPollingTimeout)

	for {
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("veo: operation timed out after %v", veoPollingTimeout)
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}

		httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodGet, pollURL, nil)
		if errNewReq != nil {
			return nil, errNewReq
		}

		// Set authorization
		if token, errTok := vertexAccessToken(ctx, cfg, auth, saJSON); errTok == nil && token != "" {
			httpReq.Header.Set("Authorization", "Bearer "+token)
		} else if errTok != nil {
			return nil, fmt.Errorf("veo: access token error during polling: %w", errTok)
		}

		httpClient := newProxyAwareHTTPClient(ctx, cfg, auth, 0)
		httpResp, errDo := httpClient.Do(httpReq)
		if errDo != nil {
			// Retry on transient errors
			log.Warnf("veo: polling error (will retry): %v", errDo)
			interval = minDuration(interval*2, veoMaxPollingInterval)
			continue
		}

		data, errRead := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("veo: close response body error: %v", errClose)
		}
		if errRead != nil {
			log.Warnf("veo: read response error (will retry): %v", errRead)
			interval = minDuration(interval*2, veoMaxPollingInterval)
			continue
		}

		// Check for retryable status codes
		if httpResp.StatusCode == 503 || httpResp.StatusCode == 504 || httpResp.StatusCode == 429 {
			log.Warnf("veo: retryable status %d (will retry)", httpResp.StatusCode)
			interval = minDuration(interval*2, veoMaxPollingInterval)
			continue
		}

		if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
			return nil, statusErr{code: httpResp.StatusCode, msg: string(data)}
		}

		// Check if operation is done
		done := gjson.GetBytes(data, "done").Bool()
		if done {
			// Check for error in response
			if errField := gjson.GetBytes(data, "error"); errField.Exists() {
				errCode := errField.Get("code").Int()
				errMsg := errField.Get("message").String()
				return nil, fmt.Errorf("veo: operation failed with code %d: %s", errCode, errMsg)
			}

			// Extract the response field which contains the actual result
			response := gjson.GetBytes(data, "response")
			if response.Exists() {
				return []byte(response.Raw), nil
			}
			// If no response field, return the whole data
			return data, nil
		}

		// Not done yet, continue polling with backoff
		interval = minDuration(interval*3/2, veoMaxPollingInterval)
		log.Debugf("veo: operation not done, polling again in %v", interval)
	}
}

// minDuration returns the smaller of two durations.
func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

// GeminiVertexExecutor sends requests to Vertex AI Gemini endpoints using service account credentials.
type GeminiVertexExecutor struct {
	cfg *config.Config
}

// NewGeminiVertexExecutor creates a new Vertex AI Gemini executor instance.
//
// Parameters:
//   - cfg: The application configuration
//
// Returns:
//   - *GeminiVertexExecutor: A new Vertex AI Gemini executor instance
func NewGeminiVertexExecutor(cfg *config.Config) *GeminiVertexExecutor {
	return &GeminiVertexExecutor{cfg: cfg}
}

// Identifier returns the executor identifier.
func (e *GeminiVertexExecutor) Identifier() string { return "vertex" }

// PrepareRequest injects Vertex credentials into the outgoing HTTP request.
func (e *GeminiVertexExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	apiKey, _ := vertexAPICreds(auth)
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("x-goog-api-key", apiKey)
		req.Header.Del("Authorization")
		return nil
	}
	_, _, saJSON, errCreds := vertexCreds(auth)
	if errCreds != nil {
		return errCreds
	}
	token, errToken := vertexAccessToken(req.Context(), e.cfg, auth, saJSON)
	if errToken != nil {
		return errToken
	}
	if strings.TrimSpace(token) == "" {
		return statusErr{code: http.StatusUnauthorized, msg: "missing access token"}
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Del("x-goog-api-key")
	return nil
}

// HttpRequest injects Vertex credentials into the request and executes it.
func (e *GeminiVertexExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("vertex executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

// Execute performs a non-streaming request to the Vertex AI API.
func (e *GeminiVertexExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	// Try API key authentication first
	apiKey, baseURL := vertexAPICreds(auth)

	// If no API key found, fall back to service account authentication
	if apiKey == "" {
		projectID, location, saJSON, errCreds := vertexCreds(auth)
		if errCreds != nil {
			return resp, errCreds
		}
		return e.executeWithServiceAccount(ctx, auth, req, opts, projectID, location, saJSON)
	}

	// Use API key authentication
	return e.executeWithAPIKey(ctx, auth, req, opts, apiKey, baseURL)
}

// ExecuteStream performs a streaming request to the Vertex AI API.
func (e *GeminiVertexExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (stream <-chan cliproxyexecutor.StreamChunk, err error) {
	// Try API key authentication first
	apiKey, baseURL := vertexAPICreds(auth)

	// If no API key found, fall back to service account authentication
	if apiKey == "" {
		projectID, location, saJSON, errCreds := vertexCreds(auth)
		if errCreds != nil {
			return nil, errCreds
		}
		return e.executeStreamWithServiceAccount(ctx, auth, req, opts, projectID, location, saJSON)
	}

	// Use API key authentication
	return e.executeStreamWithAPIKey(ctx, auth, req, opts, apiKey, baseURL)
}

// CountTokens counts tokens for the given request using the Vertex AI API.
func (e *GeminiVertexExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	// Try API key authentication first
	apiKey, baseURL := vertexAPICreds(auth)

	// If no API key found, fall back to service account authentication
	if apiKey == "" {
		projectID, location, saJSON, errCreds := vertexCreds(auth)
		if errCreds != nil {
			return cliproxyexecutor.Response{}, errCreds
		}
		return e.countTokensWithServiceAccount(ctx, auth, req, opts, projectID, location, saJSON)
	}

	// Use API key authentication
	return e.countTokensWithAPIKey(ctx, auth, req, opts, apiKey, baseURL)
}

// Refresh refreshes the authentication credentials (no-op for Vertex).
func (e *GeminiVertexExecutor) Refresh(_ context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	return auth, nil
}

// executeWithServiceAccount handles authentication using service account credentials.
// This method contains the original service account authentication logic.
func (e *GeminiVertexExecutor) executeWithServiceAccount(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, projectID, location string, saJSON []byte) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	var body []byte

	// Handle Imagen models with special request format
	if isImagenModel(baseModel) {
		imagenBody, errImagen := convertToImagenRequest(req.Payload)
		if errImagen != nil {
			return resp, errImagen
		}
		body = imagenBody
	} else if isVeoModel(baseModel) {
		// Handle Veo models with special request format
		veoBody, errVeo := convertToVeoRequest(req.Payload)
		if errVeo != nil {
			return resp, errVeo
		}
		body = veoBody
	} else {
		// Standard Gemini translation flow
		from := opts.SourceFormat
		to := sdktranslator.FromString("gemini")

		originalPayload := bytes.Clone(req.Payload)
		if len(opts.OriginalRequest) > 0 {
			originalPayload = bytes.Clone(opts.OriginalRequest)
		}
		originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, false)
		body = sdktranslator.TranslateRequest(from, to, baseModel, bytes.Clone(req.Payload), false)

		body, err = thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
		if err != nil {
			return resp, err
		}

		body = fixGeminiImageAspectRatio(baseModel, body)
		requestedModel := payloadRequestedModel(opts, req.Model)
		body = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel)
		body, _ = sjson.SetBytes(body, "model", baseModel)
	}

	action := getVertexAction(baseModel, false)
	if req.Metadata != nil {
		if a, _ := req.Metadata["action"].(string); a == "countTokens" {
			action = "countTokens"
		}
	}
	baseURL := vertexBaseURL(location)
	url := fmt.Sprintf("%s/%s/projects/%s/locations/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, projectID, location, baseModel, action)
	if opts.Alt != "" && action != "countTokens" {
		url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
	}
	body, _ = sjson.DeleteBytes(body, "session_id")

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if errNewReq != nil {
		return resp, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if token, errTok := vertexAccessToken(ctx, e.cfg, auth, saJSON); errTok == nil && token != "" {
		httpReq.Header.Set("Authorization", "Bearer "+token)
	} else if errTok != nil {
		log.Errorf("vertex executor: access token error: %v", errTok)
		return resp, statusErr{code: 500, msg: "internal server error"}
	}
	applyGeminiHeaders(httpReq, auth)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		recordAPIResponseError(ctx, e.cfg, errDo)
		return resp, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		log.Debugf("request error, error status: %d, error body: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return resp, err
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		recordAPIResponseError(ctx, e.cfg, errRead)
		return resp, errRead
	}
	appendAPIResponseChunk(ctx, e.cfg, data)

	// For Veo models, handle LRO polling
	if isVeoModel(baseModel) {
		// Extract operation name from response
		operationName := gjson.GetBytes(data, "name").String()
		if operationName == "" {
			// If no operation name, check if response is already complete (done=true)
			if gjson.GetBytes(data, "done").Bool() {
				// Already done, extract response
				if response := gjson.GetBytes(data, "response"); response.Exists() {
					data = []byte(response.Raw)
				}
			} else {
				return resp, fmt.Errorf("veo: no operation name in response")
			}
		} else {
			// Poll for completion
			log.Debugf("veo: starting LRO polling for operation: %s", operationName)
			pollData, errPoll := pollVeoOperation(ctx, e.cfg, auth, operationName, saJSON)
			if errPoll != nil {
				return resp, errPoll
			}
			data = pollData
		}
		// Convert Veo response to Gemini format
		data = convertVeoToGeminiResponse(data, baseModel)
	}

	reporter.publish(ctx, parseGeminiUsage(data))

	// For Imagen models, convert response to Gemini format before translation
	// This ensures Imagen responses use the same format as gemini-3-pro-image-preview
	if isImagenModel(baseModel) {
		data = convertImagenToGeminiResponse(data, baseModel)
	}

	// Standard Gemini translation (works for both Gemini and converted Imagen responses)
	from := opts.SourceFormat
	to := sdktranslator.FromString("gemini")
	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, bytes.Clone(opts.OriginalRequest), body, data, &param)
	resp = cliproxyexecutor.Response{Payload: []byte(out)}
	return resp, nil
}

// executeWithAPIKey handles authentication using API key credentials.
func (e *GeminiVertexExecutor) executeWithAPIKey(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, apiKey, baseURL string) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	to := sdktranslator.FromString("gemini")

	originalPayload := bytes.Clone(req.Payload)
	if len(opts.OriginalRequest) > 0 {
		originalPayload = bytes.Clone(opts.OriginalRequest)
	}
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, false)
	body := sdktranslator.TranslateRequest(from, to, baseModel, bytes.Clone(req.Payload), false)

	body, err = thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	body = fixGeminiImageAspectRatio(baseModel, body)
	requestedModel := payloadRequestedModel(opts, req.Model)
	body = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel)
	body, _ = sjson.SetBytes(body, "model", baseModel)

	action := getVertexAction(baseModel, false)
	if req.Metadata != nil {
		if a, _ := req.Metadata["action"].(string); a == "countTokens" {
			action = "countTokens"
		}
	}

	// For API key auth, use simpler URL format without project/location
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	url := fmt.Sprintf("%s/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, baseModel, action)
	if opts.Alt != "" && action != "countTokens" {
		url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
	}
	body, _ = sjson.DeleteBytes(body, "session_id")

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if errNewReq != nil {
		return resp, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("x-goog-api-key", apiKey)
	}
	applyGeminiHeaders(httpReq, auth)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		recordAPIResponseError(ctx, e.cfg, errDo)
		return resp, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		log.Debugf("request error, error status: %d, error body: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return resp, err
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		recordAPIResponseError(ctx, e.cfg, errRead)
		return resp, errRead
	}
	appendAPIResponseChunk(ctx, e.cfg, data)
	reporter.publish(ctx, parseGeminiUsage(data))
	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, bytes.Clone(opts.OriginalRequest), body, data, &param)
	resp = cliproxyexecutor.Response{Payload: []byte(out)}
	return resp, nil
}

// executeStreamWithServiceAccount handles streaming authentication using service account credentials.
func (e *GeminiVertexExecutor) executeStreamWithServiceAccount(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, projectID, location string, saJSON []byte) (stream <-chan cliproxyexecutor.StreamChunk, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	to := sdktranslator.FromString("gemini")

	originalPayload := bytes.Clone(req.Payload)
	if len(opts.OriginalRequest) > 0 {
		originalPayload = bytes.Clone(opts.OriginalRequest)
	}
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, true)
	body := sdktranslator.TranslateRequest(from, to, baseModel, bytes.Clone(req.Payload), true)

	body, err = thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return nil, err
	}

	body = fixGeminiImageAspectRatio(baseModel, body)
	requestedModel := payloadRequestedModel(opts, req.Model)
	body = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel)
	body, _ = sjson.SetBytes(body, "model", baseModel)

	action := getVertexAction(baseModel, true)
	baseURL := vertexBaseURL(location)
	url := fmt.Sprintf("%s/%s/projects/%s/locations/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, projectID, location, baseModel, action)
	// Imagen models don't support streaming, skip SSE params
	if !isImagenModel(baseModel) {
		if opts.Alt == "" {
			url = url + "?alt=sse"
		} else {
			url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
		}
	}
	body, _ = sjson.DeleteBytes(body, "session_id")

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if errNewReq != nil {
		return nil, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if token, errTok := vertexAccessToken(ctx, e.cfg, auth, saJSON); errTok == nil && token != "" {
		httpReq.Header.Set("Authorization", "Bearer "+token)
	} else if errTok != nil {
		log.Errorf("vertex executor: access token error: %v", errTok)
		return nil, statusErr{code: 500, msg: "internal server error"}
	}
	applyGeminiHeaders(httpReq, auth)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		recordAPIResponseError(ctx, e.cfg, errDo)
		return nil, errDo
	}
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		log.Debugf("request error, error status: %d, error body: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
		return nil, statusErr{code: httpResp.StatusCode, msg: string(b)}
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	stream = out
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("vertex executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, streamScannerBuffer)
		var param any
		for scanner.Scan() {
			line := scanner.Bytes()
			appendAPIResponseChunk(ctx, e.cfg, line)
			if detail, ok := parseGeminiStreamUsage(line); ok {
				reporter.publish(ctx, detail)
			}
			lines := sdktranslator.TranslateStream(ctx, to, from, req.Model, bytes.Clone(opts.OriginalRequest), body, bytes.Clone(line), &param)
			for i := range lines {
				out <- cliproxyexecutor.StreamChunk{Payload: []byte(lines[i])}
			}
		}
		lines := sdktranslator.TranslateStream(ctx, to, from, req.Model, bytes.Clone(opts.OriginalRequest), body, []byte("[DONE]"), &param)
		for i := range lines {
			out <- cliproxyexecutor.StreamChunk{Payload: []byte(lines[i])}
		}
		if errScan := scanner.Err(); errScan != nil {
			recordAPIResponseError(ctx, e.cfg, errScan)
			reporter.publishFailure(ctx)
			out <- cliproxyexecutor.StreamChunk{Err: errScan}
		}
	}()
	return stream, nil
}

// executeStreamWithAPIKey handles streaming authentication using API key credentials.
func (e *GeminiVertexExecutor) executeStreamWithAPIKey(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, apiKey, baseURL string) (stream <-chan cliproxyexecutor.StreamChunk, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	to := sdktranslator.FromString("gemini")

	originalPayload := bytes.Clone(req.Payload)
	if len(opts.OriginalRequest) > 0 {
		originalPayload = bytes.Clone(opts.OriginalRequest)
	}
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, true)
	body := sdktranslator.TranslateRequest(from, to, baseModel, bytes.Clone(req.Payload), true)

	body, err = thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return nil, err
	}

	body = fixGeminiImageAspectRatio(baseModel, body)
	requestedModel := payloadRequestedModel(opts, req.Model)
	body = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel)
	body, _ = sjson.SetBytes(body, "model", baseModel)

	action := getVertexAction(baseModel, true)
	// For API key auth, use simpler URL format without project/location
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	url := fmt.Sprintf("%s/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, baseModel, action)
	// Imagen models don't support streaming, skip SSE params
	if !isImagenModel(baseModel) {
		if opts.Alt == "" {
			url = url + "?alt=sse"
		} else {
			url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
		}
	}
	body, _ = sjson.DeleteBytes(body, "session_id")

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if errNewReq != nil {
		return nil, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("x-goog-api-key", apiKey)
	}
	applyGeminiHeaders(httpReq, auth)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		recordAPIResponseError(ctx, e.cfg, errDo)
		return nil, errDo
	}
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		log.Debugf("request error, error status: %d, error body: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
		return nil, statusErr{code: httpResp.StatusCode, msg: string(b)}
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	stream = out
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("vertex executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, streamScannerBuffer)
		var param any
		for scanner.Scan() {
			line := scanner.Bytes()
			appendAPIResponseChunk(ctx, e.cfg, line)
			if detail, ok := parseGeminiStreamUsage(line); ok {
				reporter.publish(ctx, detail)
			}
			lines := sdktranslator.TranslateStream(ctx, to, from, req.Model, bytes.Clone(opts.OriginalRequest), body, bytes.Clone(line), &param)
			for i := range lines {
				out <- cliproxyexecutor.StreamChunk{Payload: []byte(lines[i])}
			}
		}
		lines := sdktranslator.TranslateStream(ctx, to, from, req.Model, bytes.Clone(opts.OriginalRequest), body, []byte("[DONE]"), &param)
		for i := range lines {
			out <- cliproxyexecutor.StreamChunk{Payload: []byte(lines[i])}
		}
		if errScan := scanner.Err(); errScan != nil {
			recordAPIResponseError(ctx, e.cfg, errScan)
			reporter.publishFailure(ctx)
			out <- cliproxyexecutor.StreamChunk{Err: errScan}
		}
	}()
	return stream, nil
}

// countTokensWithServiceAccount counts tokens using service account credentials.
func (e *GeminiVertexExecutor) countTokensWithServiceAccount(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, projectID, location string, saJSON []byte) (cliproxyexecutor.Response, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	from := opts.SourceFormat
	to := sdktranslator.FromString("gemini")

	translatedReq := sdktranslator.TranslateRequest(from, to, baseModel, bytes.Clone(req.Payload), false)

	translatedReq, err := thinking.ApplyThinking(translatedReq, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	translatedReq = fixGeminiImageAspectRatio(baseModel, translatedReq)
	translatedReq, _ = sjson.SetBytes(translatedReq, "model", baseModel)
	respCtx := context.WithValue(ctx, "alt", opts.Alt)
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "tools")
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "generationConfig")
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "safetySettings")

	baseURL := vertexBaseURL(location)
	url := fmt.Sprintf("%s/%s/projects/%s/locations/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, projectID, location, baseModel, "countTokens")

	httpReq, errNewReq := http.NewRequestWithContext(respCtx, http.MethodPost, url, bytes.NewReader(translatedReq))
	if errNewReq != nil {
		return cliproxyexecutor.Response{}, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if token, errTok := vertexAccessToken(ctx, e.cfg, auth, saJSON); errTok == nil && token != "" {
		httpReq.Header.Set("Authorization", "Bearer "+token)
	} else if errTok != nil {
		log.Errorf("vertex executor: access token error: %v", errTok)
		return cliproxyexecutor.Response{}, statusErr{code: 500, msg: "internal server error"}
	}
	applyGeminiHeaders(httpReq, auth)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translatedReq,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		recordAPIResponseError(ctx, e.cfg, errDo)
		return cliproxyexecutor.Response{}, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		log.Debugf("request error, error status: %d, error body: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		return cliproxyexecutor.Response{}, statusErr{code: httpResp.StatusCode, msg: string(b)}
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		recordAPIResponseError(ctx, e.cfg, errRead)
		return cliproxyexecutor.Response{}, errRead
	}
	appendAPIResponseChunk(ctx, e.cfg, data)
	count := gjson.GetBytes(data, "totalTokens").Int()
	out := sdktranslator.TranslateTokenCount(ctx, to, from, count, data)
	return cliproxyexecutor.Response{Payload: []byte(out)}, nil
}

// countTokensWithAPIKey handles token counting using API key credentials.
func (e *GeminiVertexExecutor) countTokensWithAPIKey(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, apiKey, baseURL string) (cliproxyexecutor.Response, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	from := opts.SourceFormat
	to := sdktranslator.FromString("gemini")

	translatedReq := sdktranslator.TranslateRequest(from, to, baseModel, bytes.Clone(req.Payload), false)

	translatedReq, err := thinking.ApplyThinking(translatedReq, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	translatedReq = fixGeminiImageAspectRatio(baseModel, translatedReq)
	translatedReq, _ = sjson.SetBytes(translatedReq, "model", baseModel)
	respCtx := context.WithValue(ctx, "alt", opts.Alt)
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "tools")
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "generationConfig")
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "safetySettings")

	// For API key auth, use simpler URL format without project/location
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	url := fmt.Sprintf("%s/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, baseModel, "countTokens")

	httpReq, errNewReq := http.NewRequestWithContext(respCtx, http.MethodPost, url, bytes.NewReader(translatedReq))
	if errNewReq != nil {
		return cliproxyexecutor.Response{}, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("x-goog-api-key", apiKey)
	}
	applyGeminiHeaders(httpReq, auth)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translatedReq,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		recordAPIResponseError(ctx, e.cfg, errDo)
		return cliproxyexecutor.Response{}, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		log.Debugf("request error, error status: %d, error body: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		return cliproxyexecutor.Response{}, statusErr{code: httpResp.StatusCode, msg: string(b)}
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		recordAPIResponseError(ctx, e.cfg, errRead)
		return cliproxyexecutor.Response{}, errRead
	}
	appendAPIResponseChunk(ctx, e.cfg, data)
	count := gjson.GetBytes(data, "totalTokens").Int()
	out := sdktranslator.TranslateTokenCount(ctx, to, from, count, data)
	return cliproxyexecutor.Response{Payload: []byte(out)}, nil
}

// vertexCreds extracts project, location and raw service account JSON from auth metadata.
func vertexCreds(a *cliproxyauth.Auth) (projectID, location string, serviceAccountJSON []byte, err error) {
	if a == nil || a.Metadata == nil {
		return "", "", nil, fmt.Errorf("vertex executor: missing auth metadata")
	}
	if v, ok := a.Metadata["project_id"].(string); ok {
		projectID = strings.TrimSpace(v)
	}
	if projectID == "" {
		// Some service accounts may use "project"; still prefer standard field
		if v, ok := a.Metadata["project"].(string); ok {
			projectID = strings.TrimSpace(v)
		}
	}
	if projectID == "" {
		return "", "", nil, fmt.Errorf("vertex executor: missing project_id in credentials")
	}
	if v, ok := a.Metadata["location"].(string); ok && strings.TrimSpace(v) != "" {
		location = strings.TrimSpace(v)
	} else {
		location = "us-central1"
	}
	var sa map[string]any
	if raw, ok := a.Metadata["service_account"].(map[string]any); ok {
		sa = raw
	}
	if sa == nil {
		return "", "", nil, fmt.Errorf("vertex executor: missing service_account in credentials")
	}
	normalized, errNorm := vertexauth.NormalizeServiceAccountMap(sa)
	if errNorm != nil {
		return "", "", nil, fmt.Errorf("vertex executor: %w", errNorm)
	}
	saJSON, errMarshal := json.Marshal(normalized)
	if errMarshal != nil {
		return "", "", nil, fmt.Errorf("vertex executor: marshal service_account failed: %w", errMarshal)
	}
	return projectID, location, saJSON, nil
}

// vertexAPICreds extracts API key and base URL from auth attributes following the claudeCreds pattern.
func vertexAPICreds(a *cliproxyauth.Auth) (apiKey, baseURL string) {
	if a == nil {
		return "", ""
	}
	if a.Attributes != nil {
		apiKey = a.Attributes["api_key"]
		baseURL = a.Attributes["base_url"]
	}
	if apiKey == "" && a.Metadata != nil {
		if v, ok := a.Metadata["access_token"].(string); ok {
			apiKey = v
		}
	}
	return
}

func vertexBaseURL(location string) string {
	loc := strings.TrimSpace(location)
	if loc == "" {
		loc = "us-central1"
	}
	return fmt.Sprintf("https://%s-aiplatform.googleapis.com", loc)
}

func vertexAccessToken(ctx context.Context, cfg *config.Config, auth *cliproxyauth.Auth, saJSON []byte) (string, error) {
	if httpClient := newProxyAwareHTTPClient(ctx, cfg, auth, 0); httpClient != nil {
		ctx = context.WithValue(ctx, oauth2.HTTPClient, httpClient)
	}
	// Use cloud-platform scope for Vertex AI.
	creds, errCreds := google.CredentialsFromJSON(ctx, saJSON, "https://www.googleapis.com/auth/cloud-platform")
	if errCreds != nil {
		return "", fmt.Errorf("vertex executor: parse service account json failed: %w", errCreds)
	}
	tok, errTok := creds.TokenSource.Token()
	if errTok != nil {
		return "", fmt.Errorf("vertex executor: get access token failed: %w", errTok)
	}
	return tok.AccessToken, nil
}

// resolveVertexConfig finds the matching vertex-api-key configuration entry for the given auth.
func (e *GeminiVertexExecutor) resolveVertexConfig(auth *cliproxyauth.Auth) *config.VertexCompatKey {
	if auth == nil || e.cfg == nil {
		return nil
	}
	var attrKey, attrBase string
	if auth.Attributes != nil {
		attrKey = strings.TrimSpace(auth.Attributes["api_key"])
		attrBase = strings.TrimSpace(auth.Attributes["base_url"])
	}
	for i := range e.cfg.VertexCompatAPIKey {
		entry := &e.cfg.VertexCompatAPIKey[i]
		cfgKey := strings.TrimSpace(entry.APIKey)
		cfgBase := strings.TrimSpace(entry.BaseURL)
		if attrKey != "" && attrBase != "" {
			if strings.EqualFold(cfgKey, attrKey) && strings.EqualFold(cfgBase, attrBase) {
				return entry
			}
			continue
		}
		if attrKey != "" && strings.EqualFold(cfgKey, attrKey) {
			if cfgBase == "" || strings.EqualFold(cfgBase, attrBase) {
				return entry
			}
		}
		if attrKey == "" && attrBase != "" && strings.EqualFold(cfgBase, attrBase) {
			return entry
		}
	}
	if attrKey != "" {
		for i := range e.cfg.VertexCompatAPIKey {
			entry := &e.cfg.VertexCompatAPIKey[i]
			if strings.EqualFold(strings.TrimSpace(entry.APIKey), attrKey) {
				return entry
			}
		}
	}
	return nil
}
