package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const edgeVersion = "0.2.1"

// endpointSettings describe one added endpoint that can be toggled and pointed
// at a different upstream model from the dashboard.
type endpointSettings struct {
	Enabled bool   `json:"enabled"`
	Model   string `json:"model"`
}

type transcriptionSettings struct {
	Enabled       bool   `json:"enabled"`
	Model         string `json:"model"`
	MaxConcurrent int    `json:"max_concurrent"`
}

type edgeSettings struct {
	TypeSafe       endpointSettings      `json:"typesafe"`
	Embeddings     endpointSettings      `json:"embeddings"`
	Transcriptions transcriptionSettings `json:"transcriptions"`
}

// defaultSettings derives the initial values from the environment so the
// dashboard only stores deltas from the deployment defaults.
func defaultSettings(cfg config) edgeSettings {
	return edgeSettings{
		TypeSafe:   endpointSettings{Enabled: true, Model: cfg.TypeSafeModel},
		Embeddings: endpointSettings{Enabled: true, Model: cfg.MistralEmbed},
		Transcriptions: transcriptionSettings{
			Enabled:       true,
			Model:         cfg.GroqSTT,
			MaxConcurrent: cfg.MaxUploads,
		},
	}
}

func (s edgeSettings) normalized() (edgeSettings, error) {
	s.TypeSafe.Model = strings.TrimSpace(s.TypeSafe.Model)
	s.Embeddings.Model = strings.TrimSpace(s.Embeddings.Model)
	s.Transcriptions.Model = strings.TrimSpace(s.Transcriptions.Model)
	if s.TypeSafe.Model == "" {
		return s, errors.New("typesafe model must not be empty")
	}
	if s.Embeddings.Model == "" {
		return s, errors.New("embeddings model must not be empty")
	}
	if s.Transcriptions.Model == "" {
		return s, errors.New("transcription model must not be empty")
	}
	if s.Transcriptions.MaxConcurrent < 1 || s.Transcriptions.MaxConcurrent > 16 {
		return s, errors.New("max_concurrent must be between 1 and 16")
	}
	return s, nil
}

// settingsStore keeps the runtime settings and persists changes to a file so
// they survive restarts.
type settingsStore struct {
	mu       sync.RWMutex
	current  edgeSettings
	path     string
	modTime  time.Time
	lastStat time.Time
}

func newSettingsStore(path string, defaults edgeSettings) *settingsStore {
	store := &settingsStore{current: defaults, path: path}
	if path == "" {
		return store
	}
	raw, errRead := os.ReadFile(path)
	if errRead != nil {
		return store
	}
	info, errStat := os.Stat(path)
	if errStat != nil {
		return store
	}
	var loaded edgeSettings
	if errUnmarshal := json.Unmarshal(raw, &loaded); errUnmarshal != nil {
		log.Printf("settings file %s is invalid, using defaults: %v", path, errUnmarshal)
		return store
	}
	normalized, errNormalize := loaded.normalized()
	if errNormalize != nil {
		log.Printf("settings file %s rejected, using defaults: %v", path, errNormalize)
		return store
	}
	store.current = normalized
	store.modTime = info.ModTime()
	return store
}

// get returns the current settings. When several replicas share the file, a
// change made on one pod is picked up by the others within a second.
func (s *settingsStore) get() edgeSettings {
	s.reloadIfChanged()
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.current
}

func (s *settingsStore) reloadIfChanged() {
	if s.path == "" {
		return
	}
	s.mu.RLock()
	lastStat := s.lastStat
	knownMod := s.modTime
	s.mu.RUnlock()
	if time.Since(lastStat) < time.Second {
		return
	}
	info, errStat := os.Stat(s.path)
	if errStat != nil || info.ModTime().Equal(knownMod) {
		s.mu.Lock()
		s.lastStat = time.Now()
		s.mu.Unlock()
		return
	}
	raw, errRead := os.ReadFile(s.path)
	if errRead != nil {
		return
	}
	var loaded edgeSettings
	if errUnmarshal := json.Unmarshal(raw, &loaded); errUnmarshal != nil {
		return
	}
	normalized, errNormalize := loaded.normalized()
	if errNormalize != nil {
		return
	}
	s.mu.Lock()
	s.current = normalized
	s.modTime = info.ModTime()
	s.lastStat = time.Now()
	s.mu.Unlock()
}

func (s *settingsStore) persisted() bool {
	if s.path == "" {
		return false
	}
	_, errStat := os.Stat(s.path)
	return errStat == nil
}

func (s *settingsStore) save(next edgeSettings) error {
	normalized, errNormalize := next.normalized()
	if errNormalize != nil {
		return errNormalize
	}
	if s.path != "" {
		raw, errMarshal := json.MarshalIndent(normalized, "", "  ")
		if errMarshal != nil {
			return errMarshal
		}
		if errMkdir := os.MkdirAll(filepath.Dir(s.path), 0o755); errMkdir != nil {
			return errMkdir
		}
		tmp := s.path + ".tmp"
		if errWrite := os.WriteFile(tmp, raw, 0o600); errWrite != nil {
			return errWrite
		}
		if errRename := os.Rename(tmp, s.path); errRename != nil {
			return errRename
		}
	}
	s.mu.Lock()
	s.current = normalized
	if info, errStat := os.Stat(s.path); errStat == nil {
		s.modTime = info.ModTime()
	}
	s.lastStat = time.Now()
	s.mu.Unlock()
	return nil
}

// dynLimiter bounds concurrent work with a resizable semaphore.
type dynLimiter struct {
	mu sync.Mutex
	ch chan struct{}
}

func newDynLimiter(size int) *dynLimiter {
	if size < 1 {
		size = 1
	}
	return &dynLimiter{ch: make(chan struct{}, size)}
}

func (d *dynLimiter) acquire(ctx context.Context) (func(), bool) {
	d.mu.Lock()
	ch := d.ch
	d.mu.Unlock()
	select {
	case ch <- struct{}{}:
		return func() { <-ch }, true
	case <-ctx.Done():
		return nil, false
	}
}

func (d *dynLimiter) resize(size int) {
	if size < 1 {
		size = 1
	}
	d.mu.Lock()
	d.ch = make(chan struct{}, size)
	d.mu.Unlock()
}

// requireAdmin guards the dashboard endpoints with the management token.
func requireAdmin(cfg config, next http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if cfg.AdminToken == "" {
			writeError(w, http.StatusNotFound, "not_found", "admin API is disabled")
			return
		}
		token := bearerToken(r)
		if token == "" || subtle.ConstantTimeCompare([]byte(token), []byte(cfg.AdminToken)) != 1 {
			writeError(w, http.StatusUnauthorized, "invalid_api_key", "invalid or missing admin token")
			return
		}
		next(w, r)
	})
}

func handleAdminStatus(cfg config, store *settingsStore, started time.Time) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
			return
		}
		settings := store.get()
		writeJSON(w, http.StatusOK, map[string]any{
			"version":        edgeVersion,
			"uptime_seconds": int(time.Since(started).Seconds()),
			"settings":       settings,
			"providers": map[string]bool{
				"typesafe": cfg.TypeSafeKey != "",
				"mistral":  cfg.MistralKey != "",
				"groq":     cfg.GroqKey != "",
			},
			"persisted": store.persisted(),
			"upstream":  cfg.CPAUpstream,
			"limits": map[string]int{
				"max_upload_mb": maxUploadBody >> 20,
				"max_json_mb":   maxJSONBody >> 20,
			},
		})
	}
}

func handleAdminConfig(store *settingsStore, limiter *dynLimiter) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			writeJSON(w, http.StatusOK, store.get())
		case http.MethodPut:
			var next edgeSettings
			body, errRead := readLimited(r, 1<<20)
			if errRead != nil {
				writeError(w, http.StatusBadRequest, "invalid_body", errRead.Error())
				return
			}
			if errUnmarshal := json.Unmarshal(body, &next); errUnmarshal != nil {
				writeError(w, http.StatusBadRequest, "invalid_json", "body must be a settings object")
				return
			}
			if errSave := store.save(next); errSave != nil {
				writeError(w, http.StatusBadRequest, "invalid_settings", errSave.Error())
				return
			}
			limiter.resize(store.get().Transcriptions.MaxConcurrent)
			writeJSON(w, http.StatusOK, store.get())
		default:
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET or PUT required")
		}
	}
}

// handleAdminModels exposes the proxy's model catalog so the dashboard can
// offer suggestions without needing the client API key.
func handleAdminModels(cfg config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()
		req, errRequest := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(cfg.CPAUpstream, "/")+"/v1/models", nil)
		if errRequest != nil {
			writeError(w, http.StatusInternalServerError, "request_error", errRequest.Error())
			return
		}
		req.Header.Set("Authorization", "Bearer "+cfg.CPAAPIKey)
		resp, errDo := http.DefaultClient.Do(req)
		if errDo != nil {
			writeError(w, http.StatusBadGateway, "upstream_error", "could not list models")
			return
		}
		defer func() { _ = resp.Body.Close() }()
		body, errBody := readCapped(resp.Body, 8<<20)
		if errBody != nil {
			writeError(w, http.StatusBadGateway, "upstream_read_error", errBody.Error())
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(body)
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if errEncode := json.NewEncoder(w).Encode(payload); errEncode != nil {
		log.Printf("admin response encode failed: %v", errEncode)
	}
}

// ensureCapacity resizes when the stored setting changed on another replica.
func (d *dynLimiter) ensureCapacity(size int) {
	if size < 1 {
		size = 1
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if cap(d.ch) != size {
		d.ch = make(chan struct{}, size)
	}
}
