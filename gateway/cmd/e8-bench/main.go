// E8 Benchmark Runner: Measures gateway latency overhead and throughput
// against a mock upstream at 1, 10, 50, 100, and 200 concurrent streams.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type LatencyResult struct {
	Count       int     `json:"count"`
	DurationSec float64 `json:"duration_sec"`
	RPS         float64 `json:"rps"`
	MinMs       float64 `json:"min_ms"`
	P50Ms       float64 `json:"p50_ms"`
	P90Ms       float64 `json:"p90_ms"`
	P95Ms       float64 `json:"p95_ms"`
	P99Ms       float64 `json:"p99_ms"`
	MaxMs       float64 `json:"max_ms"`
}

type ConcurrencyBenchmark struct {
	Concurrency        int           `json:"concurrency"`
	Direct             LatencyResult `json:"direct"`
	Gateway            LatencyResult `json:"gateway"`
	AddedOverheadP50Ms float64       `json:"added_overhead_p50_ms"`
	AddedOverheadP99Ms float64       `json:"added_overhead_p99_ms"`
	InternalP50Us      float64       `json:"internal_p50_us"`
	InternalP99Us      float64       `json:"internal_p99_us"`
}

func calcLatencyResult(latencies []time.Duration, totalTime time.Duration) LatencyResult {
	n := len(latencies)
	if n == 0 {
		return LatencyResult{}
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
	pct := func(p float64) float64 {
		idx := int(float64(n) * p)
		if idx >= n {
			idx = n - 1
		}
		return ms(latencies[idx])
	}

	return LatencyResult{
		Count:       n,
		DurationSec: totalTime.Seconds(),
		RPS:         float64(n) / totalTime.Seconds(),
		MinMs:       ms(latencies[0]),
		P50Ms:       pct(0.50),
		P90Ms:       pct(0.90),
		P95Ms:       pct(0.95),
		P99Ms:       pct(0.99),
		MaxMs:       ms(latencies[n-1]),
	}
}

func runWorkerPool(targetURL string, concurrency int, reqsPerWorker int, stream bool, headers map[string]string) ([]time.Duration, time.Duration) {
	tr := &http.Transport{
		Proxy:               nil,
		DialContext:         (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:        1024,
		MaxIdleConnsPerHost: 512,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true,
	}
	client := &http.Client{Transport: tr, Timeout: 10 * time.Second}

	reqBody := []byte(`{"model":"qwen2.5-0.5b-instruct","messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0.7,"max_tokens":16}`)

	var (
		wg        sync.WaitGroup
		mu        sync.Mutex
		latencies = make([]time.Duration, 0, concurrency*reqsPerWorker)
	)

	start := time.Now()
	for c := 0; c < concurrency; c++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			local := make([]time.Duration, 0, reqsPerWorker)
			for i := 0; i < reqsPerWorker; i++ {
				req, err := http.NewRequest(http.MethodPost, targetURL, bytes.NewReader(reqBody))
				if err != nil {
					continue
				}
				req.Header.Set("Content-Type", "application/json")
				if stream {
					req.Header.Set("Accept", "text/event-stream")
				}
				for k, v := range headers {
					req.Header.Set(k, v)
				}

				t0 := time.Now()
				resp, err := client.Do(req)
				if err == nil {
					_, _ = io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
					local = append(local, time.Since(t0))
				}
			}
			mu.Lock()
			latencies = append(latencies, local...)
			mu.Unlock()
		}()
	}
	wg.Wait()
	total := time.Since(start)
	return latencies, total
}

func getInternalOverhead(debugURL string) (float64, float64) {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(debugURL)
	if err != nil || resp.StatusCode != http.StatusOK {
		return 0, 0
	}
	defer resp.Body.Close()
	var data struct {
		P50Us float64 `json:"p50_us"`
		P99Us float64 `json:"p99_us"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return 0, 0
	}
	return data.P50Us, data.P99Us
}

func resetInternalOverhead(debugURL string) {
	client := &http.Client{Timeout: 2 * time.Second}
	req, _ := http.NewRequest(http.MethodPost, debugURL, nil)
	_, _ = client.Do(req)
}

func main() {
	upstreamURL := flag.String("upstream", "http://127.0.0.1:9099/chat/completions", "mock upstream URL")
	gatewayURL := flag.String("gateway", "http://127.0.0.1:8080/v1/chat/completions", "gateway URL")
	debugURL := flag.String("debug", "http://127.0.0.1:8080/debug/overhead", "gateway overhead debug URL")
	concurrenciesStr := flag.String("concurrency", "1,10,50,100,200", "comma-separated concurrency levels")
	reqsPerWorker := flag.Int("n", 50, "requests per concurrent worker")
	stream := flag.Bool("stream", false, "test streaming SSE mode")
	outPath := flag.String("out", "results/e8_overhead.json", "output JSON path")
	flag.Parse()

	var concs []int
	for _, s := range strings.Split(*concurrenciesStr, ",") {
		if c, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && c > 0 {
			concs = append(concs, c)
		}
	}

	modeStr := "Non-streaming JSON"
	if *stream {
		modeStr = "Streaming SSE"
	}

	fmt.Println("=========================================================================================")
	fmt.Printf("E8 GATEWAY OVERHEAD & LATENCY BENCHMARK (%s)\n", modeStr)
	fmt.Println("=========================================================================================")
	fmt.Printf("%-12s | %-12s | %-12s | %-14s | %-14s | %-12s\n",
		"Concurrency", "Direct p99", "Gateway p99", "Added p99 (ms)", "Internal p99", "Target Status")
	fmt.Println("-----------------------------------------------------------------------------------------")

	var results []ConcurrencyBenchmark

	for _, c := range concs {
		// Warm-up
		runWorkerPool(*upstreamURL, 2, 5, *stream, nil)
		runWorkerPool(*gatewayURL, 2, 5, *stream, map[string]string{"X-Verigate-Provider": "mock"})

		// 1. Direct upstream
		dLat, dTot := runWorkerPool(*upstreamURL, c, *reqsPerWorker, *stream, nil)
		dRes := calcLatencyResult(dLat, dTot)

		// 2. Gateway
		resetInternalOverhead(*debugURL)
		gLat, gTot := runWorkerPool(*gatewayURL, c, *reqsPerWorker, *stream, map[string]string{"X-Verigate-Provider": "mock"})
		gRes := calcLatencyResult(gLat, gTot)

		intP50, intP99 := getInternalOverhead(*debugURL)
		addedP50 := gRes.P50Ms - dRes.P50Ms
		addedP99 := gRes.P99Ms - dRes.P99Ms
		if addedP50 < 0 {
			addedP50 = 0.001
		}
		if addedP99 < 0 {
			addedP99 = 0.001
		}

		status := "PASS (<1ms)"
		if addedP99 > 1.0 {
			status = "WARN (>1ms)"
		}

		bench := ConcurrencyBenchmark{
			Concurrency:        c,
			Direct:             dRes,
			Gateway:            gRes,
			AddedOverheadP50Ms: addedP50,
			AddedOverheadP99Ms: addedP99,
			InternalP50Us:      intP50,
			InternalP99Us:      intP99,
		}
		results = append(results, bench)

		fmt.Printf("%-12d | %-10.3fms | %-10.3fms | %-12.3fms | %-12.1fµs | %s\n",
			c, dRes.P99Ms, gRes.P99Ms, addedP99, intP99, status)
	}

	fmt.Println("=========================================================================================")

	_ = os.MkdirAll(filepath.Dir(*outPath), 0755)
	data, _ := json.MarshalIndent(results, "", "  ")
	_ = os.WriteFile(*outPath, data, 0644)
	log.Printf("Detailed E8 benchmark results saved to %s", *outPath)
}
