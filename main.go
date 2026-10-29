// Command ai-edge is a small HTTP front door for CLIProxyAPI.
//
// It serves the three endpoints a CLIProxyAPI plugin cannot add — TypeSafe
// (Jev) /v1/systemone, /v1/embeddings and /v1/audio/transcriptions — and
// reverse-proxies every other request to CLIProxyAPI unchanged. That keeps one
// base URL for clients while leaving the proxy itself on upstream releases.
package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const maxJSONBody = 32 << 20   // 32 MiB
const maxUploadBody = 64 << 20 // 64 MiB

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
	}
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

	proxy := newProxy(upstream)
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = io.WriteString(w, "ok\n")
	})
	mux.Handle("/v1/systemone", requireKey(cfg, handleSystemOne(cfg)))
	mux.Handle("/v1/embeddings", requireKey(cfg, handleEmbeddings(cfg)))
	mux.Handle("/v1/audio/transcriptions", requireKey(cfg, handleTranscriptions(cfg)))
	mux.Handle("/", proxy)

	server := &http.Server{
		Addr:              cfg.Listen,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 20 * time.Second,
		// No write timeout: proxied generations and SSE responses can be long.
	}
	log.Printf("ai-edge listening on %s (proxy -> %s)", cfg.Listen, upstream)
	if errListen := server.ListenAndServe(); errListen != nil && !errors.Is(errListen, http.ErrServerClosed) {
		log.Fatalf("server failed: %v", errListen)
	}
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
func handleSystemOne(cfg config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
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
		body = ensureModel(body, cfg.TypeSafeModel)

		status, headers, respBody, errDo := doWithRetry(func() (*http.Request, error) {
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
// /v1/embeddings endpoint, which is the embedding provider already configured
// for this stack.
func handleEmbeddings(cfg config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
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
		if errUnmarshal := json.Unmarshal(body, &payload); errUnmarshal != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "request body must be JSON")
			return
		}
		model := strings.TrimSpace(jsonString(payload["model"]))
		if !strings.HasPrefix(strings.ToLower(model), "mistral") {
			modelJSON, _ := json.Marshal(cfg.MistralEmbed)
			payload["model"] = modelJSON
		}
		rewritten, errMarshal := json.Marshal(payload)
		if errMarshal != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "request body must be JSON")
			return
		}

		status, headers, respBody, errDo := doWithRetry(func() (*http.Request, error) {
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
// defaulting the model when the client did not send one.
func handleTranscriptions(cfg config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
			return
		}
		if cfg.GroqKey == "" {
			writeError(w, http.StatusServiceUnavailable, "not_configured", "GROQ_API_KEY is not configured")
			return
		}
		body, errRead := readLimited(r, maxUploadBody)
		if errRead != nil {
			writeError(w, http.StatusRequestEntityTooLarge, "body_too_large", errRead.Error())
			return
		}
		rewritten, contentType, errRewrite := withDefaultFormField(body, r.Header.Get("Content-Type"), "model", cfg.GroqSTT)
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
		respBody, _ := io.ReadAll(resp.Body)
		copyResponse(w, resp.StatusCode, resp.Header, respBody, "groq-whisper")
	}
}

// withDefaultFormField appends a form field when the client did not send it.
func withDefaultFormField(body []byte, contentType, field, value string) ([]byte, string, error) {
	mediaType, params, errParse := mimeParse(contentType)
	if errParse != nil || !strings.HasPrefix(mediaType, "multipart/") {
		return nil, "", errors.New("expected a multipart/form-data upload")
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	form, errRead := reader.ReadForm(maxUploadBody)
	if errRead != nil {
		return nil, "", errRead
	}
	defer func() { _ = form.RemoveAll() }()

	if len(form.Value[field]) > 0 && strings.TrimSpace(form.Value[field][0]) != "" {
		// Field present: forward untouched so the client keeps control.
		return body, contentType, nil
	}

	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)
	for key, values := range form.Value {
		if key == field {
			continue
		}
		for _, item := range values {
			_ = writer.WriteField(key, item)
		}
	}
	_ = writer.WriteField(field, value)
	for key, files := range form.File {
		for _, fileHeader := range files {
			part, errCreate := writer.CreateFormFile(key, fileHeader.Filename)
			if errCreate != nil {
				return nil, "", errCreate
			}
			source, errOpen := fileHeader.Open()
			if errOpen != nil {
				return nil, "", errOpen
			}
			if _, errCopy := io.Copy(part, source); errCopy != nil {
				_ = source.Close()
				return nil, "", errCopy
			}
			_ = source.Close()
		}
	}
	if errClose := writer.Close(); errClose != nil {
		return nil, "", errClose
	}
	return buffer.Bytes(), writer.FormDataContentType(), nil
}

func mimeParse(contentType string) (string, map[string]string, error) {
	mediaType := contentType
	params := map[string]string{}
	if idx := strings.Index(contentType, ";"); idx >= 0 {
		mediaType = strings.TrimSpace(contentType[:idx])
		for _, part := range strings.Split(contentType[idx+1:], ";") {
			kv := strings.SplitN(strings.TrimSpace(part), "=", 2)
			if len(kv) == 2 {
				params[strings.ToLower(strings.TrimSpace(kv[0]))] = strings.Trim(strings.TrimSpace(kv[1]), `"`)
			}
		}
	}
	if params["boundary"] == "" {
		return "", nil, errors.New("missing multipart boundary")
	}
	return mediaType, params, nil
}

func readLimited(r *http.Request, limit int64) ([]byte, error) {
	body, errRead := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if errRead != nil {
		return nil, errRead
	}
	if int64(len(body)) > limit {
		return nil, errors.New("request body exceeds limit of " + strconv.FormatInt(limit, 10) + " bytes")
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

// doWithRetry retries only the upstream statuses that mean "try again":
// 429 and 529, honouring retry-after with a small cap.
func doWithRetry(build func() (*http.Request, error)) (int, http.Header, []byte, error) {
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
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		lastStatus, lastHeaders, lastBody = resp.StatusCode, resp.Header, respBody
		if resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode != 529 {
			return lastStatus, lastHeaders, lastBody, nil
		}
		if attempt < attempts-1 {
			time.Sleep(retryDelay(resp.Header))
		}
	}
	return lastStatus, lastHeaders, lastBody, nil
}

func retryDelay(header http.Header) time.Duration {
	if raw := strings.TrimSpace(header.Get("Retry-After")); raw != "" {
		if seconds, errAtoi := strconv.Atoi(raw); errAtoi == nil {
			if seconds > 5 {
				seconds = 5
			}
			return time.Duration(seconds) * time.Second
		}
	}
	if raw := strings.TrimSpace(header.Get("Retry-After-Ms")); raw != "" {
		if millis, errAtoi := strconv.Atoi(raw); errAtoi == nil {
			if millis > 5000 {
				millis = 5000
			}
			return time.Duration(millis) * time.Millisecond
		}
	}
	return 300 * time.Millisecond
}

func copyResponse(w http.ResponseWriter, status int, headers http.Header, body []byte, upstream string) {
	if contentType := headers.Get("Content-Type"); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}
	w.Header().Set("X-Ai-Edge-Upstream", upstream)
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
