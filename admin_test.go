package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func adminTestConfig(t *testing.T) config {
	t.Helper()
	cfg := testConfig()
	cfg.AdminToken = "admin-token"
	cfg.ConfigFile = filepath.Join(t.TempDir(), "edge-config.json")
	return cfg
}

func TestAdminRequiresToken(t *testing.T) {
	cfg := adminTestConfig(t)
	store := newSettingsStore(cfg.ConfigFile, defaultSettings(cfg))
	handler := requireAdmin(cfg, handleAdminStatus(cfg, store, time.Time{}))

	req := httptest.NewRequest(http.MethodGet, "/v0/management/edge-admin/status", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing token: expected 401, got %d", rec.Code)
	}

	req.Header.Set("Authorization", "Bearer admin-token")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("valid token: expected 200, got %d", rec.Code)
	}
}

func TestAdminDisabledWithoutToken(t *testing.T) {
	cfg := adminTestConfig(t)
	cfg.AdminToken = ""
	store := newSettingsStore(cfg.ConfigFile, defaultSettings(cfg))
	handler := requireAdmin(cfg, handleAdminStatus(cfg, store, time.Time{}))

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v0/management/edge-admin/status", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 when admin is disabled, got %d", rec.Code)
	}
}

func TestAdminConfigRoundTrip(t *testing.T) {
	cfg := adminTestConfig(t)
	store := newSettingsStore(cfg.ConfigFile, defaultSettings(cfg))
	limiter := newDynLimiter(2)
	handler := requireAdmin(cfg, handleAdminConfig(store, limiter))

	next := store.get()
	next.TypeSafe.Model = "jev-1.14.0"
	next.Embeddings.Enabled = false
	next.Transcriptions.MaxConcurrent = 4
	body, _ := json.Marshal(next)

	req := httptest.NewRequest(http.MethodPut, "/v0/management/edge-admin/config", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer admin-token")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("save: expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	// Persisted to disk and reloadable.
	reloaded := newSettingsStore(cfg.ConfigFile, defaultSettings(cfg))
	if got := reloaded.get(); got.TypeSafe.Model != "jev-1.14.0" || got.Embeddings.Enabled || got.Transcriptions.MaxConcurrent != 4 {
		t.Fatalf("reloaded settings mismatch: %+v", got)
	}
	if _, errStat := os.Stat(cfg.ConfigFile); errStat != nil {
		t.Fatalf("settings file not written: %v", errStat)
	}
}

func TestAdminConfigRejectsInvalid(t *testing.T) {
	cfg := adminTestConfig(t)
	store := newSettingsStore(cfg.ConfigFile, defaultSettings(cfg))
	handler := requireAdmin(cfg, handleAdminConfig(store, newDynLimiter(2)))

	next := store.get()
	next.Transcriptions.MaxConcurrent = 99
	body, _ := json.Marshal(next)

	req := httptest.NewRequest(http.MethodPut, "/v0/management/edge-admin/config", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer admin-token")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for invalid settings, got %d", rec.Code)
	}
}

func TestDisabledEndpointReturns503(t *testing.T) {
	cfg := testConfig()
	store := newSettingsStore("", defaultSettings(cfg))
	settings := store.get()
	settings.TypeSafe.Enabled = false
	if errSave := store.save(settings); errSave != nil {
		t.Fatalf("save: %v", errSave)
	}
	handler := requireKey(cfg, handleSystemOne(cfg, store))
	req := httptest.NewRequest(http.MethodPost, "/v1/systemone", bytes.NewReader([]byte(`{"state":"x","questions":{}}`)))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 for a disabled endpoint, got %d", rec.Code)
	}
}
