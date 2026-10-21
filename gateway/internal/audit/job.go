package audit

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

// TopLogprob is one alternative token with its log-probability.
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// Output is everything the gateway tees from the upstream response (GW-4).
type Output struct {
	Text          string         `json:"text"`
	Tokens        []string       `json:"tokens"`
	TokenBytes    [][]int        `json:"token_bytes,omitempty"`
	TokenLogprobs []float64      `json:"token_logprobs"`
	TopLogprobs   [][]TopLogprob `json:"top_logprobs"`
	FinishReason  string         `json:"finish_reason"`
}

// Params are the declared sampling parameters of the request.
type Params struct {
	Temperature *float64 `json:"temperature"`
	TopP        *float64 `json:"top_p"`
	TopK        *int     `json:"top_k"`
	MaxTokens   *int     `json:"max_tokens"`
	Seed        *int64   `json:"seed"`
	// SamplingModifiers is true when the request used penalties, min_p or
	// logit_bias, which the verifier cannot model exactly.
	SamplingModifiers bool `json:"sampling_modifiers,omitempty"`
}

// Usage mirrors the upstream usage block, when present.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// Job is the audit record placed on the queue (PRD §7.2).
type Job struct {
	AuditID    string            `json:"audit_id"`
	RequestID  string            `json:"request_id"`
	Provider   string            `json:"provider"`
	ModelAlias string            `json:"model_alias"`
	ReceivedAt string            `json:"received_at"`
	Messages   []json.RawMessage `json:"messages"`
	Params     Params            `json:"params"`
	Output     Output            `json:"output"`
	Usage      *Usage            `json:"usage,omitempty"`
	Trusted    bool              `json:"trusted,omitempty"`
	PromptHash string            `json:"prompt_hash"`
	OutputHash string            `json:"output_hash"`
}

// Finalize fills the content hashes.
func (j *Job) Finalize() {
	msgs, _ := json.Marshal(j.Messages)
	j.PromptHash = hashOf(msgs)
	j.OutputHash = hashOf([]byte(j.Output.Text))
}

func hashOf(b []byte) string {
	s := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(s[:])
}
