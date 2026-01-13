# Add --qr and --ip flags for OAuth login

**Bead:** CLIProxyAPI-ifg
**Created:** 2026-01-13

## Goal

Add QR code generation (`--qr`) and custom IP support (`--ip`) for OAuth login flows, enabling easier mobile scanning and remote machine callback support.

## Success Criteria

- [x] `--qr` flag generates QR code in terminal for login URL
  - Verify: `./cliproxy --login --qr` displays QR code
- [x] `--ip=x.x.x.x` flag overrides localhost in callback URL
  - Verify: `./cliproxy --login --ip=192.168.1.100` uses custom IP in OAuth redirect
- [x] Works with all login types (--login, --claude-login, --codex-login, etc.)
  - Verify: Test each login flow with --qr and --ip flags
- [x] QR code contains the full OAuth authorization URL
  - Verify: Scan QR code and confirm it opens correct auth URL

## Implementation Notes

**Completed 2026-01-13:**

- Added `--qr` and `--ip` flags to `cmd/server/main.go`
- Updated `LoginOptions` structs in `internal/cmd/openai_login.go` and `sdk/auth/interfaces.go`
- Modified `internal/auth/gemini/gemini_auth.go` to:
  - Use custom callback IP when `--ip` is provided
  - Display QR code using `github.com/mdp/qrterminal/v3` when `--qr` is set
- Propagated options through all login flows (Gemini, Claude, Codex, iFlow, Antigravity)

## Constraints

**Must:**

- Use terminal-based QR code (no external dependencies like image viewers)
- Support both IPv4 addresses
- Preserve existing login behavior when flags not used

**Never:**

- Break existing login flows
- Store IP addresses in credentials

## Technical Approach

1. Add flags to `cmd/server/main.go`:
   - `--qr` (bool): Enable QR code display
   - `--ip` (string): Custom IP for callback URL

2. Update `LoginOptions` struct in `internal/cmd/openai_login.go`:
   - Add `ShowQR bool`
   - Add `CallbackIP string`

3. Modify OAuth callback URL generation:
   - Replace `localhost` with custom IP when `--ip` is provided
   - Example: `http://192.168.1.100:8085/oauth2callback`

4. Add QR code generation:
   - Use `github.com/skip2/go-qrcode` or similar terminal QR library
   - Display QR when `--qr` flag is set and auth URL is generated

## Files to Modify

- `cmd/server/main.go` - Add CLI flags
- `internal/cmd/login.go` - Pass options to login flows
- `internal/cmd/openai_login.go` - LoginOptions struct
- `internal/auth/gemini/gemini_auth.go` - Callback URL generation
- `sdk/auth/*.go` - Various authenticator implementations

## Notes

Use case: Running CLIProxyAPI on a remote server, user wants to authenticate via phone by scanning QR code that points to the server's IP address for callback.
