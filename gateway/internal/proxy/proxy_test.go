package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"verigate/gateway/internal/audit"
	"verigate/gateway/internal/config"
	"verigate/gateway/internal/metrics"
)

const sseBody = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"logprobs\":{\"content\":[{\"token\":\"Hel\",\"logprob\":-0.1,\"bytes\":[72,101,108],\"top_logprobs\":[{\"token\":\"Hel\",\"logprob\":-0.1}]}]},\"finish_reason\":null}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"logprobs\":{\"content\":[{\"token\":\"lo\",\"logprob\":-0.2,\"bytes\":[108,111],\"top_logprobs\":[]}]},\"finish_reason\":null}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
	"data: [DONE]\n\n"

const jsonBody = `{"id":"c2","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"logprobs":{"content":[{"token":"Hello","logprob":-0.05,"top_logprobs":[]}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}`

type recorder struct {
	mu   sync.Mutex
	jobs []*audit.Job
}

func (r *recorder) Submit(j *audit.Job) bool {
	r.mu.Lock()
	r.jobs = append(r.jobs, j)
	r.mu.Unlock()
	return true
}

func upstream(t *testing.T, bodies *[][]byte, mu *sync.Mutex) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		*bodies = append(*bodies, b)
		mu.Unlock()
		var req map[string]any
		_ = json.Unmarshal(b, &req)
		if req["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, sseBody)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, jsonBody)
	}))
}

func newGateway(t *testing.T, upURL string, rate float64, logprobs bool, sub Submitter) *httptest.Server {
	t.Setenv("TEST_SECRET", "s3cret")
	yaml := fmt.Sprintf(`
gateway: {audit_secret_env: TEST_SECRET}
providers:
  up:
    base_url: %q
    models: {m: upstream-model}
    request_logprobs: %v
    audit_rate: %v
`, upURL+"/v1", logprobs, rate)
	cfg, err := config.Parse([]byte(yaml))
	if err != nil {
		t.Fatal(err)
	}
	s := New(cfg, sub, metrics.New())
	return httptest.NewServer(s.Handler())
}

func post(t *testing.T, url, body string) (*http.Response, []byte) {
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, b
}

func TestStreamingPassthroughByteIdentical(t *testing.T) {
	var bodies [][]byte
	var mu sync.Mutex
	up := upstream(t, &bodies, &mu)
	defer up.Close()
	rec := &recorder{}
	gw := newGateway(t, up.URL, 1.0, true, rec)
	defer gw.Close()

	resp, b := post(t, gw.URL, `{"model":"m","stream":true,"logprobs":true,"messages":[{"role":"user","content":"hi"}]}`)
	if string(b) != sseBody {
		t.Fatalf("stream not byte-identical:\n%s", b)
	}
	if resp.Header.Get("X-Verigate-Request-Id") == "" {
		t.Fatal("missing X-Verigate-Request-Id")
	}
	if len(rec.jobs) != 1 {
		t.Fatalf("expected 1 audit job, got %d", len(rec.jobs))
	}
	j := rec.jobs[0]
	if j.Output.Text != "Hello" || strings.Join(j.Output.Tokens, "|") != "Hel|lo" || j.Output.FinishReason != "stop" {
		t.Fatalf("bad tee: %+v", j.Output)
	}
	if len(j.Output.TokenBytes) != 2 || j.Output.TokenLogprobs[1] != -0.2 {
		t.Fatalf("bad logprob tee: %+v", j.Output)
	}
	if j.ModelAlias != "m" || j.Provider != "up" || j.RequestID != resp.Header.Get("X-Verigate-Request-Id") {
		t.Fatalf("bad job metadata: %+v", j)
	}
}

func TestInjectedLogprobsStripped(t *testing.T) {
	var bodies [][]byte
	var mu sync.Mutex
	up := upstream(t, &bodies, &mu)
	defer up.Close()
	rec := &recorder{}
	gw := newGateway(t, up.URL, 1.0, true, rec)
	defer gw.Close()

	// Non-streaming: client did not ask for logprobs.
	_, b := post(t, gw.URL, `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	if bytes.Contains(b, []byte("logprob")) {
		t.Fatalf("logprobs leaked to client: %s", b)
	}
	// Streaming.
	_, b = post(t, gw.URL, `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if bytes.Contains(b, []byte("logprob")) {
		t.Fatalf("logprobs leaked to streaming client: %s", b)
	}
	if !bytes.Contains(b, []byte(`"content":"Hel"`)) || !bytes.HasSuffix(b, []byte("data: [DONE]\n\n")) {
		t.Fatalf("stream content damaged: %s", b)
	}
	// Upstream always received logprobs=true, top_logprobs=5; model rewritten.
	for _, ub := range bodies {
		var m map[string]any
		_ = json.Unmarshal(ub, &m)
		if m["logprobs"] != true || m["top_logprobs"] != float64(5) || m["model"] != "upstream-model" {
			t.Fatalf("bad upstream body: %s", ub)
		}
	}
	// The audit still captured the logprobs.
	if len(rec.jobs) != 2 || len(rec.jobs[0].Output.TokenLogprobs) != 1 || len(rec.jobs[1].Output.TokenLogprobs) != 2 {
		t.Fatalf("audit jobs missing logprobs: %d", len(rec.jobs))
	}
}

// Property test (PRD §17): audited and non-audited requests produce
// byte-identical upstream request bodies.
func TestUpstreamBodyIndependentOfAudit(t *testing.T) {
	reqs := []string{
		`{"model":"m","messages":[{"role":"user","content":"a"}],"temperature":0.7}`,
		`{"model":"m","stream":true,"messages":[{"role":"system","content":"s"},{"role":"user","content":"b"}],"top_p":0.9,"seed":3}`,
		`{"model":"m","logprobs":true,"top_logprobs":8,"messages":[{"role":"user","content":"c"}],"max_tokens":16}`,
	}
	collect := func(rate float64) [][]byte {
		var bodies [][]byte
		var mu sync.Mutex
		up := upstream(t, &bodies, &mu)
		defer up.Close()
		gw := newGateway(t, up.URL, rate, true, &recorder{})
		defer gw.Close()
		for _, r := range reqs {
			post(t, gw.URL, r)
		}
		return bodies
	}
	never, always := collect(0), collect(1)
	for i := range reqs {
		if !bytes.Equal(never[i], always[i]) {
			t.Fatalf("upstream body differs with auditing:\n%s\n%s", never[i], always[i])
		}
	}
}

type nopSub struct{}

func (nopSub) Submit(*audit.Job) bool { return false }

func TestQueueOutageDoesNotAffectResponses(t *testing.T) {
	var bodies [][]byte
	var mu sync.Mutex
	up := upstream(t, &bodies, &mu)
	defer up.Close()
	// A real enqueuer pointing at a closed port.
	p, _ := audit.NewRedisPusher("redis://127.0.0.1:1/0", "k", 60)
	enq := audit.NewEnqueuer(p, 16, 1)
	defer enq.Close()
	gw := newGateway(t, up.URL, 1.0, false, enq)
	defer gw.Close()
	for i := 0; i < 20; i++ {
		resp, b := post(t, gw.URL, `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
		if resp.StatusCode != 200 || string(b) != jsonBody {
			t.Fatalf("response affected by queue outage: %d %s", resp.StatusCode, b)
		}
	}
}

func TestRoutingErrors(t *testing.T) {
	var bodies [][]byte
	var mu sync.Mutex
	up := upstream(t, &bodies, &mu)
	defer up.Close()
	gw := newGateway(t, up.URL, 0, false, nopSub{})
	defer gw.Close()
	resp, _ := post(t, gw.URL, `{"model":"nope","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 404 {
		t.Fatalf("unknown model: got %d", resp.StatusCode)
	}
	req, _ := http.NewRequest("POST", gw.URL+"/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("X-Verigate-Provider", "other")
	r2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	r2.Body.Close()
	if r2.StatusCode != 404 {
		t.Fatalf("unknown provider: got %d", r2.StatusCode)
	}
	resp, _ = post(t, gw.URL, `not json`)
	if resp.StatusCode != 400 {
		t.Fatalf("bad json: got %d", resp.StatusCode)
	}
}

func TestSamplingModifiersFlagged(t *testing.T) {
	var bodies [][]byte
	var mu sync.Mutex
	up := upstream(t, &bodies, &mu)
	defer up.Close()
	rec := &recorder{}
	gw := newGateway(t, up.URL, 1.0, false, rec)
	defer gw.Close()
	post(t, gw.URL, `{"model":"m","messages":[{"role":"user","content":"hi"}],"frequency_penalty":0.5}`)
	post(t, gw.URL, `{"model":"m","messages":[{"role":"user","content":"hi"}],"frequency_penalty":0}`)
	if !rec.jobs[0].Params.SamplingModifiers || rec.jobs[1].Params.SamplingModifiers {
		t.Fatalf("sampling modifier flags wrong")
	}
}
