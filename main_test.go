package main

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func testConfig(overrides ...func(*config)) config {
	cfg := loadConfig()
	cfg.CPAAPIKey = "test-key"
	cfg.TypeSafeKey = "ts-key"
	cfg.MistralKey = "mistral-key"
	cfg.GroqKey = "groq-key"
	for _, override := range overrides {
		override(&cfg)
	}
	return cfg
}

func TestRequireKey(t *testing.T) {
	handler := requireKey(testConfig(), func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/systemone", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing key: expected 401, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/v1/systemone", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong key: expected 401, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/v1/systemone", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("valid key: expected 200, got %d", rec.Code)
	}
}

func TestSystemOneDefaultsModelAndForwardsKey(t *testing.T) {
	var seenAuth string
	var seenBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&seenBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"jev-1.14.0","answers":{"is_bug":{"type":"noul","noul":0.91}}}`)
	}))
	defer upstream.Close()

	cfg := testConfig(func(c *config) { c.TypeSafeBase = upstream.URL; c.TypeSafeModel = "jev-latest" })
	handler := requireKey(cfg, handleSystemOne(cfg))

	req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"state":"it crashes","questions":{"is_bug":{"type":"noul","instructions":"bug?"}}}`))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if seenAuth != "Bearer ts-key" {
		t.Fatalf("upstream must receive the TypeSafe key, got %q", seenAuth)
	}
	if seenBody["model"] != "jev-latest" {
		t.Fatalf("missing model must default to jev-latest, got %v", seenBody["model"])
	}
}

func TestEmbeddingsMapsUnknownModel(t *testing.T) {
	var seen map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&seen)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"embedding":[0.1,0.2],"index":0}]}`)
	}))
	defer upstream.Close()

	cfg := testConfig(func(c *config) { c.MistralBase = upstream.URL; c.MistralEmbed = "mistral-embed" })
	handler := requireKey(cfg, handleEmbeddings(cfg))

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"text-embedding-3-small","input":"hello"}`))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if seen["model"] != "mistral-embed" {
		t.Fatalf("expected model remap to mistral-embed, got %v", seen["model"])
	}
	if seen["input"] != "hello" {
		t.Fatalf("input must pass through, got %v", seen["input"])
	}
}

func TestEmbeddingsKeepsMistralModel(t *testing.T) {
	var seen map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&seen)
		_, _ = io.WriteString(w, `{"data":[]}`)
	}))
	defer upstream.Close()

	cfg := testConfig(func(c *config) { c.MistralBase = upstream.URL })
	handler := requireKey(cfg, handleEmbeddings(cfg))
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"mistral-embed","input":["a"]}`))
	req.Header.Set("Authorization", "Bearer test-key")
	handler.ServeHTTP(httptest.NewRecorder(), req)

	if seen["model"] != "mistral-embed" {
		t.Fatalf("mistral model must be preserved, got %v", seen["model"])
	}
}

func TestTranscriptionsInjectsDefaultModel(t *testing.T) {
	var gotModel, gotFile string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if errParse := r.ParseMultipartForm(maxUploadBody); errParse != nil {
			t.Errorf("upstream parse: %v", errParse)
			return
		}
		gotModel = r.FormValue("model")
		file, _, errFile := r.FormFile("file")
		if errFile == nil {
			content, _ := io.ReadAll(file)
			gotFile = string(content)
			_ = file.Close()
		}
		_, _ = io.WriteString(w, `{"text":"hello"}`)
	}))
	defer upstream.Close()

	cfg := testConfig(func(c *config) { c.GroqBase = upstream.URL; c.GroqSTT = "whisper-large-v3" })
	handler := requireKey(cfg, handleTranscriptions(cfg))

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, _ := writer.CreateFormFile("file", "clip.webm")
	_, _ = part.Write([]byte("fake-audio"))
	_ = writer.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", writer.FormDataContentType())
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if gotModel != "whisper-large-v3" {
		t.Fatalf("expected default model injection, got %q", gotModel)
	}
	if gotFile != "fake-audio" {
		t.Fatalf("file must survive the rewrite, got %q", gotFile)
	}
}

func TestProxyForwardsEverythingElse(t *testing.T) {
	var path string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	upstreamURL, _ := url.Parse(upstream.URL)
	proxy := newProxy(upstreamURL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || path != "/v1/chat/completions" {
		t.Fatalf("expected passthrough, got %d %q", rec.Code, path)
	}
}

func TestWithDefaultFormFieldKeepsExplicitModel(t *testing.T) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	_ = writer.WriteField("model", "whisper-large-v3-turbo")
	part, _ := writer.CreateFormFile("file", "a.webm")
	_, _ = part.Write([]byte("x"))
	_ = writer.Close()

	rewritten, _, errRewrite := withDefaultFormField(body.Bytes(), writer.FormDataContentType(), "model", "whisper-large-v3")
	if errRewrite != nil {
		t.Fatalf("rewrite failed: %v", errRewrite)
	}
	if !bytes.Equal(rewritten, body.Bytes()) {
		t.Fatal("explicit model must not be rewritten")
	}
}
