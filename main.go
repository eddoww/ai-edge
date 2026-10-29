// Command ai-edge is a small HTTP front door for CLIProxyAPI.
//
// It serves the three endpoints a CLIProxyAPI plugin cannot add — TypeSafe
// (Jev) /v1/systemone, /v1/embeddings and /v1/audio/transcriptions — and
// reverse-proxies every other request to CLIProxyAPI unchanged, including
// SSE and WebSocket upgrades. That keeps one base URL for clients while
// leaving the proxy itself on upstream releases.
package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httputil"
	"net/textproto"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	maxJSONBody           = 32 << 20 // 32 MiB
	maxUploadBody         = 25 << 20 // Whisper's own per-file limit
	maxUpstreamJSONBody   = 64 << 20 // embeddings and Jev answers
	maxUpstreamAudioBody  = 16 << 20
	shutdownGrace         = 25 * time.Second
	defaultTranscriptions = 2
)

type config struct {
	Listen        string
	CPAUpstream   string
	CPAAPIKey     string
	TypeSafeKey   string
	TypeSafeBase  string
	TypeSafeModel string
	MistralKey    string
	MistralBase   string
	MistralEmbed  string
	GroqKey       string
	GroqBase      string
	GroqSTT       string
	MaxUploads    int
	AdminToken    string
	ConfigFile    string
}

func env(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func loadConfig() config {
	return config{
		Listen:        env("LISTEN_ADDR", ":8080"),
		CPAUpstream:   env("CPA_UPSTREAM", "http://cliproxyapi.ai-tools.svc.cluster.local:80"),
		CPAAPIKey:     os.Getenv("CPA_API_KEY"),
		TypeSafeKey:   os.Getenv("TYPESAFE_API_KEY"),
		TypeSafeBase:  env("TYPESAFE_BASE_URL", "https://api.typesafe.ai"),
		TypeSafeModel: env("TYPESAFE_MODEL", "jev-latest"),
		MistralKey:    os.Getenv("MISTRAL_API_KEY"),
		MistralBase:   env("MISTRAL_BASE_URL", "https://api.mistral.ai"),
		MistralEmbed:  env("MISTRAL_EMBED_MODEL", "mistral-embed"),
		GroqKey:       os.Getenv("GROQ_API_KEY"),
		GroqBase:      env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
		GroqSTT:       env("GROQ_STT_MODEL", "whisper-large-v3"),
		MaxUploads:    envInt("MAX_CONCURRENT_TRANSCRIPTIONS", defaultTranscriptions),
		AdminToken:    os.Getenv("EDGE_ADMIN_TOKEN"),
		ConfigFile:    env("EDGE_CONFIG_FILE", "/data/edge-config.json"),
	}
}

func envInt(key string, fallback int) int {
	if value, errAtoi := strconv.Atoi(strings.TrimSpace(os.Getenv(key))); errAtoi == nil && value > 0 {
		return value
	}
	return fallback
}

func main() {
	cfg := loadConfig()
	upstream, errParse := url.Parse(cfg.CPAUpstream)
	if errParse != nil {
		log.Fatalf("invalid CPA_UPSTREAM %q: %v", cfg.CPAUpstream, errParse)
	}
	if cfg.CPAAPIKey == "" {
		log.Fatal("CPA_API_KEY is required: the added endpoints authenticate against it")
	}

	started := time.Now()
	store := newSettingsStore(cfg.ConfigFile, defaultSettings(cfg))
	limiter := newDynLimiter(store.get().Transcriptions.MaxConcurrent)

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = io.WriteString(w, "ok\n")
	})
	mux.Handle("/v1/systemone", requireKey(cfg, handleSystemOne(cfg, store)))
	mux.Handle("/v1/embeddings", requireKey(cfg, handleEmbeddings(cfg, store)))
	mux.Handle("/v1/audio/transcriptions", requireKey(cfg, handleTranscriptions(cfg, store, limiter)))
	// The management UI talks to /v8/management; older clients use /v0. Serve both.
	for _, prefix := range []string{"/v0/management", "/v8/management"} {
		mux.Handle(prefix+"/edge-admin/status", requireAdmin(cfg, handleAdminStatus(cfg, store, started)))
		mux.Handle(prefix+"/edge-admin/config", requireAdmin(cfg, handleAdminConfig(store, limiter)))
		mux.Handle(prefix+"/edge-admin/models", requireAdmin(cfg, handleAdminModels(cfg)))
	}
	mux.Handle("/", newProxy(upstream))

	server := &http.Server{
		Addr:              cfg.Listen,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 20 * time.Second,
		// No write timeout: proxied generations and SSE responses can be long.
	}

	// Shutdown does not close hijacked (WebSocket) connections; track them so
	// they are closed once HTTP draining finishes instead of at hard exit.
	var hijackedMu sync.Mutex
	hijacked := map[net.Conn]struct{}{}
	server.ConnState = func(conn net.Conn, state http.ConnState) {
		if state != http.StateHijacked {
			return
		}
		hijackedMu.Lock()
		hijacked[conn] = struct{}{}
		hijackedMu.Unlock()
	}

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		<-signals
		log.Printf("shutdown requested; draining for up to %s", shutdownGrace)
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
		defer cancel()
		// Shutdown drains in-flight HTTP requests; hijacked WebSocket
		// connections are closed below, once draining finishes.
		if errShutdown := server.Shutdown(shutdownCtx); errShutdown != nil {
			log.Printf("shutdown error: %v", errShutdown)
		}
	}()

	log.Printf("ai-edge listening on %s (proxy -> %s)", cfg.Listen, upstream)
	if errListen := server.ListenAndServe(); errListen != nil && !errors.Is(errListen, http.ErrServerClosed) {
		log.Fatalf("server failed: %v", errListen)
	}
	// Wait for the drain: returning here would kill the shutdown goroutine.
	<-shutdownDone
	hijackedMu.Lock()
	for conn := range hijacked {
		_ = conn.Close()
	}
	hijackedMu.Unlock()
	log.Printf("ai-edge stopped")
}

func newProxy(upstream *url.URL) *httputil.ReverseProxy {
	proxy := httputil.NewSingleHostReverseProxy(upstream)
	// FlushInterval -1 streams SSE chunks as they arrive instead of buffering.
	proxy.FlushInterval = -1
	proxy.Transport = &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   50,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ForceAttemptHTTP2:     true,
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("proxy error for %s %s: %v", r.Method, r.URL.Path, err)
		writeError(w, http.StatusBadGateway, "upstream_unavailable", "CLIProxyAPI is unreachable")
	}
	return proxy
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		recorder := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(recorder, r)
		log.Printf("%s %s %d %s", r.Method, r.URL.Path, recorder.status, time.Since(start).Round(time.Millisecond))
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (s *statusRecorder) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

// Flush keeps SSE streaming working through the recorder.
func (s *statusRecorder) Flush() {
	if flusher, ok := s.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// Unwrap lets http.ResponseController reach Hijack (WebSocket upgrades) and
// other optional interfaces on the real writer.
func (s *statusRecorder) Unwrap() http.ResponseWriter {
	return s.ResponseWriter
}

func requireKey(cfg config, next http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := bearerToken(r)
		if token == "" || subtle.ConstantTimeCompare([]byte(token), []byte(cfg.CPAAPIKey)) != 1 {
			writeError(w, http.StatusUnauthorized, "invalid_api_key", "invalid or missing API key")
			return
		}
		next(w, r)
	})
}

func bearerToken(r *http.Request) string {
	header := strings.TrimSpace(r.Header.Get("Authorization"))
	if len(header) > 7 && strings.EqualFold(header[:7], "bearer ") {
		return strings.TrimSpace(header[7:])
	}
	return ""
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{"type": code, "message": message, "code": code},
	})
}

// handleSystemOne forwards TypeSafe Jev requests. Jev answers typed questions
// about supplied state and returns calibrated probabilities, so the body is
// passed through almost verbatim; only a missing model is defaulted.
func handleSystemOne(cfg config, store *settingsStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
			return
		}
		settings := store.get().TypeSafe
		if !settings.Enabled {
			writeError(w, http.StatusServiceUnavailable, "disabled", "the systemone endpoint is disabled in the dashboard")
			return
		}
		if cfg.TypeSafeKey == "" {
			writeError(w, http.StatusServiceUnavailable, "not_configured", "TYPESAFE_API_KEY is not configured")
			return
		}
		body, errRead := readLimited(r, maxJSONBody)
		if errRead != nil {
			writeError(w, http.StatusRequestEntityTooLarge, "body_too_large", errRead.Error())
			return
		}
		body = ensureModel(body, settings.Model)

		status, headers, respBody, errDo := doWithRetry(r.Context(), func() (*http.Request, error) {
			return newJSONRequest(r.Context(), cfg.TypeSafeBase+"/v1/systemone", cfg.TypeSafeKey, body)
		})
		if errDo != nil {
			log.Printf("typesafe request failed: %v", errDo)
			writeError(w, http.StatusBadGateway, "upstream_error", "TypeSafe request failed")
			return
		}
		copyResponse(w, status, headers, respBody, "typesafe")
	}
}

// handleEmbeddings maps OpenAI-style embedding requests onto Mistral's
// /v1/embeddings endpoint, which is the embedding provider configured here.
func handleEmbeddings(cfg config, store *settingsStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
			return
		}
		settings := store.get().Embeddings
		if !settings.Enabled {
			writeError(w, http.StatusServiceUnavailable, "disabled", "the embeddings endpoint is disabled in the dashboard")
			return
		}
		if cfg.MistralKey == "" {
			writeError(w, http.StatusServiceUnavailable, "not_configured", "MISTRAL_API_KEY is not configured")
			return
		}
		body, errRead := readLimited(r, maxJSONBody)
		if errRead != nil {
			writeError(w, http.StatusRequestEntityTooLarge, "body_too_large", errRead.Error())
			return
		}
		var payload map[string]json.RawMessage
		if errUnmarshal := json.Unmarshal(body, &payload); errUnmarshal != nil || payload == nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "request body must be a JSON object")
			return
		}
		model := strings.TrimSpace(jsonString(payload["model"]))
		if !strings.HasPrefix(strings.ToLower(model), "mistral") {
			modelJSON, _ := json.Marshal(settings.Model)
			payload["model"] = modelJSON
		}
		rewritten, errMarshal := json.Marshal(payload)
		if errMarshal != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "request body must be a JSON object")
			return
		}

		status, headers, respBody, errDo := doWithRetry(r.Context(), func() (*http.Request, error) {
			return newJSONRequest(r.Context(), cfg.MistralBase+"/v1/embeddings", cfg.MistralKey, rewritten)
		})
		if errDo != nil {
			log.Printf("mistral embeddings request failed: %v", errDo)
			writeError(w, http.StatusBadGateway, "upstream_error", "embedding request failed")
			return
		}
		copyResponse(w, status, headers, respBody, "mistral-embeddings")
	}
}

// handleTranscriptions forwards multipart audio to Groq's Whisper endpoint,
// defaulting the model when the client did not send one. A semaphore bounds how
// many uploads are buffered at once so a burst cannot exhaust the container.
func handleTranscriptions(cfg config, store *settingsStore, limiter *dynLimiter) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
			return
		}
		settings := store.get().Transcriptions
		if !settings.Enabled {
			writeError(w, http.StatusServiceUnavailable, "disabled", "the transcription endpoint is disabled in the dashboard")
			return
		}
		if cfg.GroqKey == "" {
			writeError(w, http.StatusServiceUnavailable, "not_configured", "GROQ_API_KEY is not configured")
			return
		}
		limiter.ensureCapacity(settings.MaxConcurrent)
		release, okAcquire := limiter.acquire(r.Context())
		if !okAcquire {
			writeError(w, http.StatusRequestTimeout, "request_cancelled", "request cancelled")
			return
		}
		defer release()

		body, errRead := readLimited(r, maxUploadBody)
		if errRead != nil {
			status := http.StatusBadRequest
			if strings.Contains(errRead.Error(), "exceeds") {
				status = http.StatusRequestEntityTooLarge
			}
			writeError(w, status, "invalid_upload", errRead.Error())
			return
		}
		rewritten, contentType, errRewrite := withDefaultFormField(body, r.Header.Get("Content-Type"), "model", settings.Model)
		if errRewrite != nil {
			writeError(w, http.StatusBadRequest, "invalid_multipart", errRewrite.Error())
			return
		}

		req, errRequest := http.NewRequestWithContext(r.Context(), http.MethodPost, cfg.GroqBase+"/audio/transcriptions", bytes.NewReader(rewritten))
		if errRequest != nil {
			writeError(w, http.StatusInternalServerError, "request_error", errRequest.Error())
			return
		}
		req.Header.Set("Content-Type", contentType)
		req.Header.Set("Authorization", "Bearer "+cfg.GroqKey)

		resp, errDo := http.DefaultClient.Do(req)
		if errDo != nil {
			log.Printf("groq transcription request failed: %v", errDo)
			writeError(w, http.StatusBadGateway, "upstream_error", "transcription request failed")
			return
		}
		defer func() { _ = resp.Body.Close() }()
		respBody, errBody := readCapped(resp.Body, maxUpstreamAudioBody)
		if errBody != nil {
			writeError(w, http.StatusBadGateway, "upstream_read_error", errBody.Error())
			return
		}
		copyResponse(w, resp.StatusCode, resp.Header, respBody, "groq-whisper")
	}
}

// withDefaultFormField appends a form field when the client did not send it.
// Existing parts keep their headers and order; the original bytes are returned
// untouched when the field is already present.
func withDefaultFormField(body []byte, contentType, field, value string) ([]byte, string, error) {
	mediaType, params, errParse := mime.ParseMediaType(contentType)
	if errParse != nil || mediaType != "multipart/form-data" {
		return nil, "", fmt.Errorf("expected multipart/form-data, got %q", mediaType)
	}
	boundary := params["boundary"]
	if boundary == "" {
		return nil, "", errors.New("missing multipart boundary")
	}

	hasField, errScan := scanFormField(body, boundary, field)
	if errScan != nil {
		return nil, "", errScan
	}
	if hasField {
		return body, contentType, nil
	}

	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)

	fieldHeader := make(textproto.MIMEHeader)
	fieldHeader.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"`, escapeQuotes(field)))
	part, errCreate := writer.CreatePart(fieldHeader)
	if errCreate != nil {
		return nil, "", errCreate
	}
	if _, errWrite := io.WriteString(part, value); errWrite != nil {
		return nil, "", errWrite
	}

	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		source, errNext := reader.NextRawPart()
		if errors.Is(errNext, io.EOF) {
			break
		}
		if errNext != nil {
			return nil, "", errNext
		}
		// Any part with this name is blank here, otherwise we would not rewrite.
		if source.FormName() == field && source.FileName() == "" {
			_ = source.Close()
			continue
		}
		copied := make(textproto.MIMEHeader, len(source.Header))
		for key, values := range source.Header {
			copied[key] = append([]string(nil), values...)
		}
		target, errCreatePart := writer.CreatePart(copied)
		if errCreatePart != nil {
			return nil, "", errCreatePart
		}
		if _, errCopy := io.Copy(target, source); errCopy != nil {
			return nil, "", errCopy
		}
		_ = source.Close()
	}
	if errClose := writer.Close(); errClose != nil {
		return nil, "", errClose
	}
	return buffer.Bytes(), writer.FormDataContentType(), nil
}

func scanFormField(body []byte, boundary, field string) (bool, error) {
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		part, errNext := reader.NextRawPart()
		if errors.Is(errNext, io.EOF) {
			return false, nil
		}
		if errNext != nil {
			return false, errNext
		}
		if part.FormName() != field || part.FileName() != "" {
			_ = part.Close()
			continue
		}
		// Read the whole field (bounded) so a long value is not mistaken for
		// blank; an oversized value counts as present and is left untouched.
		raw, errRead := io.ReadAll(io.LimitReader(part, 64<<10))
		_ = part.Close()
		if errRead != nil {
			return false, errRead
		}
		if len(raw) >= 64<<10 {
			return true, nil
		}
		if strings.TrimSpace(string(raw)) != "" {
			return true, nil
		}
	}
}

func escapeQuotes(value string) string {
	return strings.NewReplacer("\\", "\\\\", `"`, "\\\"").Replace(value)
}

func readLimited(r *http.Request, limit int64) ([]byte, error) {
	body, errRead := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if errRead != nil {
		return nil, errRead
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("request body exceeds limit of %s bytes", strconv.FormatInt(limit, 10))
	}
	return body, nil
}

func readCapped(reader io.Reader, limit int64) ([]byte, error) {
	body, errRead := io.ReadAll(io.LimitReader(reader, limit+1))
	if errRead != nil {
		return nil, errRead
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("upstream response exceeds limit of %s bytes", strconv.FormatInt(limit, 10))
	}
	return body, nil
}

func ensureModel(body []byte, fallback string) []byte {
	var payload map[string]json.RawMessage
	if errUnmarshal := json.Unmarshal(body, &payload); errUnmarshal != nil || payload == nil {
		return body
	}
	if strings.TrimSpace(jsonString(payload["model"])) != "" {
		return body
	}
	modelJSON, _ := json.Marshal(fallback)
	payload["model"] = modelJSON
	rewritten, errMarshal := json.Marshal(payload)
	if errMarshal != nil {
		return body
	}
	return rewritten
}

func jsonString(raw json.RawMessage) string {
	var value string
	if len(raw) == 0 {
		return ""
	}
	if errUnmarshal := json.Unmarshal(raw, &value); errUnmarshal != nil {
		return ""
	}
	return value
}

func newJSONRequest(ctx context.Context, endpoint, apiKey string, body []byte) (*http.Request, error) {
	req, errRequest := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if errRequest != nil {
		return nil, errRequest
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "application/json")
	return req, nil
}

// doWithRetry retries only the statuses that mean "try again": 429 and 529,
// honouring Retry-After (seconds or HTTP date) with a small cap.
func doWithRetry(ctx context.Context, build func() (*http.Request, error)) (int, http.Header, []byte, error) {
	const attempts = 3
	var lastStatus int
	var lastHeaders http.Header
	var lastBody []byte
	for attempt := 0; attempt < attempts; attempt++ {
		req, errBuild := build()
		if errBuild != nil {
			return 0, nil, nil, errBuild
		}
		resp, errDo := http.DefaultClient.Do(req)
		if errDo != nil {
			return 0, nil, nil, errDo
		}
		respBody, errRead := readCapped(resp.Body, maxUpstreamJSONBody)
		_ = resp.Body.Close()
		if errRead != nil {
			return 0, nil, nil, errRead
		}
		lastStatus, lastHeaders, lastBody = resp.StatusCode, resp.Header, respBody
		if resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode != 529 {
			return lastStatus, lastHeaders, lastBody, nil
		}
		if attempt < attempts-1 {
			delay, okRetry := retryDelay(resp.Header)
			if !okRetry {
				// The upstream asked for more delay than the retry budget:
				// return its response instead of retrying early.
				return lastStatus, lastHeaders, lastBody, nil
			}
			if delay <= 0 {
				continue
			}
			timer := time.NewTimer(delay)
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				return lastStatus, lastHeaders, lastBody, nil
			}
		}
	}
	return lastStatus, lastHeaders, lastBody, nil
}

// retryDelay returns the requested wait and whether a retry is still worthwhile.
// A delay beyond the retry budget returns false: retrying early would ignore the
// upstream's guidance, so the response is returned instead.
func retryDelay(header http.Header) (time.Duration, bool) {
	const maxDelay = 5 * time.Second
	parse := func(delay time.Duration) (time.Duration, bool) {
		if delay <= 0 {
			return 0, true
		}
		if delay > maxDelay {
			return 0, false
		}
		return delay, true
	}
	if raw := strings.TrimSpace(header.Get("Retry-After")); raw != "" {
		if seconds, errAtoi := strconv.Atoi(raw); errAtoi == nil {
			if seconds > 3600 {
				seconds = 3600
			}
			if seconds <= 0 {
				return 0, true
			}
			return parse(time.Duration(seconds) * time.Second)
		}
		if when, errParse := http.ParseTime(raw); errParse == nil {
			return parse(time.Until(when))
		}
	}
	if raw := strings.TrimSpace(header.Get("Retry-After-Ms")); raw != "" {
		if millis, errAtoi := strconv.Atoi(raw); errAtoi == nil {
			if millis > 3_600_000 {
				millis = 3_600_000
			}
			if millis <= 0 {
				return 0, true
			}
			return parse(time.Duration(millis) * time.Millisecond)
		}
	}
	return 300 * time.Millisecond, true
}

func copyResponse(w http.ResponseWriter, status int, headers http.Header, body []byte, upstream string) {
	if contentType := headers.Get("Content-Type"); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}
	// Keep the retry hint so clients can back off correctly.
	if retryAfter := headers.Get("Retry-After"); retryAfter != "" {
		w.Header().Set("Retry-After", retryAfter)
	}
	w.Header().Set("X-Ai-Edge-Upstream", upstream)
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
