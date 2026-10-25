// Mock OpenAI upstream server for E8 latency and throughput benchmarking.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"
)

var (
	nonStreamResp []byte
	streamChunks  [][]byte
)

func init() {
	// Pre-generate response bodies for zero allocation in the hot loop
	resp := map[string]any{
		"id":      "chatcmpl-mock-001",
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   "mock-model",
		"choices": []map[string]any{
			{
				"index":         0,
				"finish_reason": "stop",
				"message": map[string]any{
					"role":    "assistant",
					"content": "Paris is the capital of France.",
				},
				"logprobs": map[string]any{
					"content": []map[string]any{
						{"token": "Paris", "logprob": -0.01, "bytes": []int{80, 97, 114, 105, 115}},
						{"token": " is", "logprob": -0.005, "bytes": []int{32, 105, 115}},
						{"token": " the", "logprob": -0.002, "bytes": []int{32, 116, 104, 101}},
						{"token": " capital", "logprob": -0.015, "bytes": []int{32, 99, 97, 112, 105, 116, 97, 108}},
						{"token": ".", "logprob": -0.001, "bytes": []int{46}},
					},
				},
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     12,
			"completion_tokens": 5,
			"total_tokens":      17,
		},
	}
	var err error
	nonStreamResp, err = json.Marshal(resp)
	if err != nil {
		panic(err)
	}

	// Pre-render SSE chunks
	tokens := []string{"Paris", " is", " the", " capital", "."}
	for i, t := range tokens {
		chunk := map[string]any{
			"id":      "chatcmpl-mock-chunk",
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   "mock-model",
			"choices": []map[string]any{
				{
					"index": 0,
					"delta": map[string]string{"content": t},
					"logprobs": map[string]any{
						"content": []map[string]any{
							{"token": t, "logprob": -0.01},
						},
					},
					"finish_reason": nil,
				},
			},
		}
		if i == 0 {
			chunk["choices"].([]map[string]any)[0]["delta"].(map[string]string)["role"] = "assistant"
		}
		b, _ := json.Marshal(chunk)
		streamChunks = append(streamChunks, []byte(fmt.Sprintf("data: %s\n\n", b)))
	}
	doneChunk := []byte("data: {\"id\":\"chatcmpl-mock-chunk\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	streamChunks = append(streamChunks, doneChunk)
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	isStream := r.Header.Get("Accept") == "text/event-stream"
	if isStream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		w.WriteHeader(http.StatusOK)
		if ok {
			flusher.Flush()
		}
		for _, c := range streamChunks {
			_, _ = w.Write(c)
			if ok {
				flusher.Flush()
			}
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", fmt.Sprint(len(nonStreamResp)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(nonStreamResp)
}

func main() {
	port := flag.Int("port", 9099, "port to listen on")
	flag.Parse()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", handleChat)
	mux.HandleFunc("/chat/completions", handleChat)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })

	server := &http.Server{
		Addr:         fmt.Sprintf("127.0.0.1:%d", *port),
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	log.Printf("Mock upstream server listening on 127.0.0.1:%d", *port)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}
