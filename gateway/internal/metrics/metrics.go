// Package metrics provides a dependency-free Prometheus text exposition (GW-10).
package metrics

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type histogram struct {
	bounds []float64
	counts []atomic.Uint64
	sumNs  atomic.Int64
	count  atomic.Uint64
}

func newHistogram(bounds []float64) *histogram {
	return &histogram{bounds: bounds, counts: make([]atomic.Uint64, len(bounds)+1)}
}

func (h *histogram) observe(d time.Duration) {
	s := d.Seconds()
	i := sort.SearchFloat64s(h.bounds, s)
	h.counts[i].Add(1)
	h.sumNs.Add(int64(d))
	h.count.Add(1)
}

// Registry holds all gateway metrics.
type Registry struct {
	mu        sync.Mutex
	requests  map[string]*atomic.Uint64 // key: provider|status
	upstream  map[string]*histogram     // key: provider
	overhead  *histogram
	selected  map[string]*atomic.Uint64

	// Reservoir of the most recent overhead samples for exact quantiles.
	resMu  sync.Mutex
	res    []time.Duration
	resPos int
	resN   int

	// Gauges supplied by the audit enqueuer.
	AuditsEnqueued func() uint64
	AuditsDropped  func() uint64
}

var overheadBounds = []float64{25e-6, 50e-6, 100e-6, 250e-6, 500e-6, 1e-3, 2.5e-3, 5e-3, 10e-3}
var upstreamBounds = []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120}

// New creates a registry.
func New() *Registry {
	return &Registry{
		requests: map[string]*atomic.Uint64{},
		upstream: map[string]*histogram{},
		selected: map[string]*atomic.Uint64{},
		overhead: newHistogram(overheadBounds),
		res:      make([]time.Duration, 100_000),
	}
}

func (r *Registry) counter(m map[string]*atomic.Uint64, key string) *atomic.Uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	c, ok := m[key]
	if !ok {
		c = &atomic.Uint64{}
		m[key] = c
	}
	return c
}

// Request counts a completed request.
func (r *Registry) Request(provider string, status int) {
	r.counter(r.requests, fmt.Sprintf("%s|%d", provider, status)).Add(1)
}

// AuditSelected counts audit selections per provider.
func (r *Registry) AuditSelected(provider string) { r.counter(r.selected, provider).Add(1) }

// Upstream records upstream latency.
func (r *Registry) Upstream(provider string, d time.Duration) {
	r.mu.Lock()
	h, ok := r.upstream[provider]
	if !ok {
		h = newHistogram(upstreamBounds)
		r.upstream[provider] = h
	}
	r.mu.Unlock()
	h.observe(d)
}

// Overhead records time spent inside the gateway, excluding upstream waits.
func (r *Registry) Overhead(d time.Duration) {
	r.overhead.observe(d)
	r.resMu.Lock()
	r.res[r.resPos] = d
	r.resPos = (r.resPos + 1) % len(r.res)
	if r.resN < len(r.res) {
		r.resN++
	}
	r.resMu.Unlock()
}

// OverheadQuantiles returns exact quantiles over the reservoir.
func (r *Registry) OverheadQuantiles(qs ...float64) (n int, out []time.Duration) {
	r.resMu.Lock()
	cp := append([]time.Duration(nil), r.res[:r.resN]...)
	r.resMu.Unlock()
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	out = make([]time.Duration, len(qs))
	if len(cp) == 0 {
		return 0, out
	}
	for i, q := range qs {
		idx := int(math.Ceil(q*float64(len(cp)))) - 1
		if idx < 0 {
			idx = 0
		}
		if idx >= len(cp) {
			idx = len(cp) - 1
		}
		out[i] = cp[idx]
	}
	return len(cp), out
}

// ResetOverhead clears the reservoir (used by benchmarks between phases).
func (r *Registry) ResetOverhead() {
	r.resMu.Lock()
	r.resPos, r.resN = 0, 0
	r.resMu.Unlock()
}

func writeHist(w io.Writer, name, labels string, h *histogram) {
	var cum uint64
	sep := ""
	if labels != "" {
		sep = ","
	}
	for i, b := range h.bounds {
		cum += h.counts[i].Load()
		fmt.Fprintf(w, "%s_bucket{%s%sle=\"%g\"} %d\n", name, labels, sep, b, cum)
	}
	cum += h.counts[len(h.bounds)].Load()
	fmt.Fprintf(w, "%s_bucket{%s%sle=\"+Inf\"} %d\n", name, labels, sep, cum)
	fmt.Fprintf(w, "%s_sum{%s} %g\n", name, labels, float64(h.sumNs.Load())/1e9)
	fmt.Fprintf(w, "%s_count{%s} %d\n", name, labels, h.count.Load())
}

func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

// ServeHTTP renders the Prometheus text format.
func (r *Registry) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	r.mu.Lock()
	defer r.mu.Unlock()

	fmt.Fprintln(w, "# HELP verigate_requests_total Completed chat completion requests.")
	fmt.Fprintln(w, "# TYPE verigate_requests_total counter")
	for _, k := range sortedKeys(r.requests) {
		parts := strings.SplitN(k, "|", 2)
		fmt.Fprintf(w, "verigate_requests_total{provider=%q,status=%q} %d\n", parts[0], parts[1], r.requests[k].Load())
	}
	fmt.Fprintln(w, "# HELP verigate_upstream_latency_seconds Upstream time from request sent to response complete.")
	fmt.Fprintln(w, "# TYPE verigate_upstream_latency_seconds histogram")
	for _, k := range sortedKeys(r.upstream) {
		writeHist(w, "verigate_upstream_latency_seconds", fmt.Sprintf("provider=%q", k), r.upstream[k])
	}
	fmt.Fprintln(w, "# HELP verigate_gateway_overhead_seconds Time spent in the gateway excluding upstream waits.")
	fmt.Fprintln(w, "# TYPE verigate_gateway_overhead_seconds histogram")
	writeHist(w, "verigate_gateway_overhead_seconds", "", r.overhead)
	fmt.Fprintln(w, "# HELP verigate_audits_selected_total Requests selected for audit.")
	fmt.Fprintln(w, "# TYPE verigate_audits_selected_total counter")
	for _, k := range sortedKeys(r.selected) {
		fmt.Fprintf(w, "verigate_audits_selected_total{provider=%q} %d\n", k, r.selected[k].Load())
	}
	if r.AuditsEnqueued != nil {
		fmt.Fprintln(w, "# TYPE verigate_audits_enqueued_total counter")
		fmt.Fprintf(w, "verigate_audits_enqueued_total %d\n", r.AuditsEnqueued())
	}
	if r.AuditsDropped != nil {
		fmt.Fprintln(w, "# TYPE verigate_audits_dropped_total counter")
		fmt.Fprintf(w, "verigate_audits_dropped_total %d\n", r.AuditsDropped())
	}
}
