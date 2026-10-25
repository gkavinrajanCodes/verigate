// Package proxy implements the OpenAI-compatible hot path: routing, streaming
// passthrough, teeing, and post-hoc audit hand-off.
package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"verigate/gateway/internal/audit"
	"verigate/gateway/internal/config"
	"verigate/gateway/internal/metrics"
)

var bufioReaderPool = sync.Pool{
	New: func() any {
		return bufio.NewReaderSize(nil, 32<<10)
	},
}

const maxBodyBytes = 32 << 20

// Submitter receives audit jobs without blocking.
type Submitter interface {
	Submit(*audit.Job) bool
}

// Server is the gateway HTTP server.
type Server struct {
	cfg    *config.Config
	client *http.Client
	enq    Submitter
	m      *metrics.Registry

	// UpstreamHook, if set, observes every upstream request body (tests).
	UpstreamHook func(provider string, body []byte)
}

// New builds a gateway server.
func New(cfg *config.Config, enq Submitter, m *metrics.Registry) *Server {
	tr := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:          1024,
		MaxIdleConnsPerHost:   512,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: time.Duration(cfg.Gateway.UpstreamTimeoutSecs) * time.Second,
		ForceAttemptHTTP2:     true,
	}
	return &Server{cfg: cfg, client: &http.Client{Transport: tr}, enq: enq, m: m}
}

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })
	mux.Handle("/metrics", s.m)
	mux.HandleFunc("/debug/overhead", s.handleOverhead)
	return mux
}

func writeErr(w http.ResponseWriter, status int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": typ, "code": status},
	})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	seen := map[string]bool{}
	data := []map[string]any{}
	for _, name := range s.cfg.ProviderNames() {
		for alias := range s.cfg.Providers[name].Models {
			if !seen[alias] {
				seen[alias] = true
				data = append(data, map[string]any{"id": alias, "object": "model", "owned_by": "verigate"})
			}
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}

func (s *Server) handleOverhead(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost || r.Method == http.MethodDelete {
		s.m.ResetOverhead()
		w.WriteHeader(http.StatusNoContent)
		return
	}
	n, q := s.m.OverheadQuantiles(0.5, 0.9, 0.99, 0.999)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"n": n, "p50_us": us(q[0]), "p90_us": us(q[1]), "p99_us": us(q[2]), "p999_us": us(q[3]),
	})
}

func us(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e3 }

// parsedRequest holds the fields of the client request we care about.
type parsedRequest struct {
	fields      map[string]json.RawMessage
	model       string
	stream      bool
	wantsLP     bool
	topLP       int
	messages    []json.RawMessage
	params      audit.Params
	unsupported bool // sampling modifiers that invalidate distributional signals
}

func parseRequest(body []byte) (*parsedRequest, error) {
	pr := &parsedRequest{}
	if err := json.Unmarshal(body, &pr.fields); err != nil {
		return nil, fmt.Errorf("invalid JSON body: %w", err)
	}
	get := func(k string, v any) {
		if raw, ok := pr.fields[k]; ok && string(raw) != "null" {
			_ = json.Unmarshal(raw, v)
		}
	}
	get("model", &pr.model)
	if pr.model == "" {
		return nil, errors.New("missing 'model'")
	}
	get("stream", &pr.stream)
	get("logprobs", &pr.wantsLP)
	get("top_logprobs", &pr.topLP)
	get("messages", &pr.messages)
	if len(pr.messages) == 0 {
		return nil, errors.New("missing 'messages'")
	}
	p := &pr.params
	get("temperature", &p.Temperature)
	get("top_p", &p.TopP)
	get("top_k", &p.TopK)
	get("seed", &p.Seed)
	var nChoice int
	get("n", &nChoice)
	if nChoice > 1 {
		pr.unsupported = true
	}
	get("max_tokens", &p.MaxTokens)
	if p.MaxTokens == nil {
		get("max_completion_tokens", &p.MaxTokens)
	}
	for _, k := range []string{"frequency_penalty", "presence_penalty", "repetition_penalty", "min_p", "logit_bias"} {
		raw, ok := pr.fields[k]
		if !ok {
			continue
		}
		s := strings.TrimSpace(string(raw))
		if s != "null" && s != "0" && s != "0.0" && s != "{}" && !(k == "repetition_penalty" && (s == "1" || s == "1.0")) {
			pr.unsupported = true
		}
	}
	return pr, nil
}

// buildUpstream renders the upstream body. It depends only on the client
// request and provider config, never on audit selection (G5 / A7 defense).
func buildUpstream(pr *parsedRequest, p *config.Provider) ([]byte, error) {
	out := make(map[string]json.RawMessage, len(pr.fields)+3)
	for k, v := range pr.fields {
		out[k] = v
	}
	mid, _ := json.Marshal(p.Models[pr.model])
	out["model"] = mid
	if p.RequestLogprobs {
		out["logprobs"] = json.RawMessage("true")
		top := pr.topLP
		if top < 5 {
			top = 5
		}
		out["top_logprobs"] = json.RawMessage(fmt.Sprint(top))
	}
	if pr.stream {
		if _, ok := out["stream_options"]; !ok {
			out["stream_options"] = json.RawMessage(`{"include_usage":true}`)
		}
	}
	if p.PinUpstreamProvider != "" {
		pin, _ := json.Marshal(map[string]any{
			"order":           []string{p.PinUpstreamProvider},
			"allow_fallbacks": false,
		})
		out["provider"] = pin
	}
	return json.Marshal(out)
}

var hopHeaders = map[string]bool{
	"Connection": true, "Keep-Alive": true, "Proxy-Authenticate": true, "Proxy-Authorization": true,
	"Te": true, "Trailer": true, "Transfer-Encoding": true, "Upgrade": true, "Content-Length": true,
}

func bearer(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if strings.HasPrefix(h, "Bearer ") {
		return strings.TrimPrefix(h, "Bearer ")
	}
	return ""
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	var busy time.Duration // gateway processing time, excluding upstream and client I/O waits

	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "invalid_request_error", "use POST")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid_request_error", "could not read body")
		return
	}
	if len(body) > maxBodyBytes {
		writeErr(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "body too large")
		return
	}
	tParse := time.Now()
	pr, err := parseRequest(body)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	prov, err := s.cfg.Resolve(pr.model, r.Header.Get("X-Verigate-Provider"))
	if err != nil {
		writeErr(w, http.StatusNotFound, "invalid_request_error", err.Error())
		return
	}
	upBody, err := buildUpstream(pr, prov)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "server_error", "could not build upstream request")
		return
	}
	requestID := audit.NewID("req_")
	receivedAt := t0.UTC().Format(time.RFC3339Nano)
	stripLP := prov.RequestLogprobs && !pr.wantsLP

	ctx, cancel := context.WithTimeout(r.Context(), time.Duration(s.cfg.Gateway.UpstreamTimeoutSecs)*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, prov.BaseURL+"/chat/completions", bytes.NewReader(upBody))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "server_error", "bad upstream url")
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if pr.stream {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
	if k := prov.APIKey(); k != "" {
		req.Header.Set("Authorization", "Bearer "+k)
	}
	if s.UpstreamHook != nil {
		s.UpstreamHook(prov.Name, upBody)
	}
	busy += time.Since(tParse)

	tUp := time.Now()
	busyBeforeUp := busy
	resp, err := s.client.Do(req)
	if err != nil {
		s.m.Request(prov.Name, http.StatusBadGateway)
		writeErr(w, http.StatusBadGateway, "upstream_error", "upstream request failed")
		log.Printf("gateway: upstream %s error: %v", prov.Name, err)
		return
	}
	defer resp.Body.Close()

	tHdr := time.Now()
	h := w.Header()
	for k, vs := range resp.Header {
		if hopHeaders[k] {
			continue
		}
		for _, v := range vs {
			h.Add(k, v)
		}
	}
	h.Set("X-Verigate-Request-Id", requestID)
	h.Set("X-Verigate-Provider", prov.Name)
	isSSE := strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream")
	busy += time.Since(tHdr)

	auditSelected := !s.cfg.OptedOut(bearer(r)) && audit.Selected(s.cfg.AuditSecret, requestID, prov.Rate())
	var t *tee
	if auditSelected {
		t = &tee{}
	}

	ok := false
	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	} else if isSSE {
		ok, busy = s.pipeStream(w, resp.Body, t, stripLP, busy)
	} else {
		ok, busy = s.pipeJSON(w, resp.Body, t, stripLP, busy)
	}
	upstreamDur := time.Since(tUp) - (busy - busyBeforeUp)
	s.m.Upstream(prov.Name, upstreamDur)
	s.m.Request(prov.Name, resp.StatusCode)

	// Post-hoc audit selection: enqueue if selected and response succeeded.
	tPost := time.Now()
	if ok && auditSelected && t != nil && r.Context().Err() == nil {
		job := &audit.Job{
			AuditID:    audit.NewID("aud_"),
			RequestID:  requestID,
			Provider:   prov.Name,
			ModelAlias: pr.model,
			ReceivedAt: receivedAt,
			Messages:   pr.messages,
			Params:     pr.params,
			Output:     t.output(),
			Usage:      t.usage,
			Trusted:    prov.Trusted,
		}
		job.Params.SamplingModifiers = pr.unsupported
		s.m.AuditSelected(prov.Name)
		s.enq.Submit(job) // never blocks (GW-8)
	}
	busy += time.Since(tPost)
	s.m.Overhead(busy)
}

// pipeJSON handles non-streaming responses.
func (s *Server) pipeJSON(w http.ResponseWriter, body io.Reader, t *tee, strip bool, busy time.Duration) (bool, time.Duration) {
	raw, err := io.ReadAll(body)
	if err != nil {
		writeErr(w, http.StatusBadGateway, "upstream_error", "upstream read failed")
		return false, busy
	}
	tp := time.Now()
	out := raw
	ok := true
	if t != nil || strip {
		var rb respBody
		ok = json.Unmarshal(raw, &rb) == nil
		if ok && t != nil {
			t.add(&rb)
		}
		if strip && ok {
			if nb, err := stripLogprobs(raw); err == nil {
				out = nb
			}
		}
	}
	w.Header().Set("Content-Length", fmt.Sprint(len(out)))
	busy += time.Since(tp)
	w.WriteHeader(http.StatusOK)
	_, err = w.Write(out)
	return ok && err == nil, busy
}

var dataPrefix = []byte("data:")

// pipeStream forwards SSE events as they arrive, without buffering (GW-3).
// Lines are written byte-for-byte unless injected logprobs must be stripped.
func (s *Server) pipeStream(w http.ResponseWriter, body io.Reader, t *tee, strip bool, busy time.Duration) (bool, time.Duration) {
	fl, _ := w.(http.Flusher)
	w.WriteHeader(http.StatusOK)
	if fl != nil {
		fl.Flush()
	}

	br := bufioReaderPool.Get().(*bufio.Reader)
	br.Reset(body)
	defer bufioReaderPool.Put(br)

	// Fast-path: when auditing is not active for this request and logprobs don't need stripping,
	// stream directly without JSON unmarshaling of chunks.
	if t == nil && !strip {
		sawDone := false
		for {
			line, err := br.ReadBytes('\n')
			if len(line) > 0 {
				tp := time.Now()
				trimmed := bytes.TrimSpace(line)
				if bytes.HasPrefix(trimmed, dataPrefix) && bytes.Equal(bytes.TrimSpace(trimmed[len(dataPrefix):]), []byte("[DONE]")) {
					sawDone = true
				}
				busy += time.Since(tp)
				if _, werr := w.Write(line); werr != nil {
					return false, busy
				}
				if len(trimmed) == 0 && fl != nil {
					fl.Flush()
				}
			}
			if err != nil {
				if fl != nil {
					fl.Flush()
				}
				return sawDone || errors.Is(err, io.EOF), busy
			}
		}
	}

	sawDone := false
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			tp := time.Now()
			out := line
			trimmed := bytes.TrimSpace(line)
			if bytes.HasPrefix(trimmed, dataPrefix) {
				payload := bytes.TrimSpace(trimmed[len(dataPrefix):])
				if bytes.Equal(payload, []byte("[DONE]")) {
					sawDone = true
				} else if len(payload) > 0 && payload[0] == '{' {
					var rb respBody
					if json.Unmarshal(payload, &rb) == nil {
						if t != nil {
							t.add(&rb)
						}
						if strip {
							if nb, e := stripLogprobs(payload); e == nil && !bytes.Equal(nb, payload) {
								out = append(append([]byte("data: "), nb...), '\n')
							}
						}
					}
				}
			}
			busy += time.Since(tp)
			if _, werr := w.Write(out); werr != nil {
				return false, busy
			}
			if len(trimmed) == 0 && fl != nil {
				fl.Flush() // end of an SSE event
			}
		}
		if err != nil {
			if fl != nil {
				fl.Flush()
			}
			if errors.Is(err, io.EOF) {
				finish := ""
				if t != nil {
					finish = t.finish
				}
				return sawDone || finish != "", busy
			}
			return false, busy
		}
	}
}
