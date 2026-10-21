package proxy

import (
	"encoding/json"
	"strings"

	"verigate/gateway/internal/audit"
)

// Wire types for extracting what we need from OpenAI-format responses.

type lpEntry struct {
	Token       string             `json:"token"`
	Logprob     float64            `json:"logprob"`
	Bytes       []int              `json:"bytes"`
	TopLogprobs []audit.TopLogprob `json:"top_logprobs"`
}

type lpBlock struct {
	Content []lpEntry `json:"content"`
}

type msgPart struct {
	Content   *string         `json:"content"`
	ToolCalls json.RawMessage `json:"tool_calls"`
}

type respChoice struct {
	Index        int      `json:"index"`
	Message      *msgPart `json:"message"`
	Delta        *msgPart `json:"delta"`
	Logprobs     *lpBlock `json:"logprobs"`
	FinishReason *string  `json:"finish_reason"`
}

type respBody struct {
	Choices []respChoice `json:"choices"`
	Usage   *audit.Usage `json:"usage"`
}

// tee accumulates the first choice of a (possibly streamed) response (GW-4).
type tee struct {
	text      strings.Builder
	tokens    []string
	bytes     [][]int
	anyBytes  bool
	lps       []float64
	tops      [][]audit.TopLogprob
	finish    string
	usage     *audit.Usage
	toolCalls bool
}

func (t *tee) add(b *respBody) {
	if b.Usage != nil {
		t.usage = b.Usage
	}
	for i := range b.Choices {
		c := &b.Choices[i]
		if c.Index != 0 {
			continue
		}
		part := c.Message
		if part == nil {
			part = c.Delta
		}
		if part != nil {
			if part.Content != nil {
				t.text.WriteString(*part.Content)
			}
			if len(part.ToolCalls) > 0 && string(part.ToolCalls) != "null" && string(part.ToolCalls) != "[]" {
				t.toolCalls = true
			}
		}
		if c.Logprobs != nil {
			for _, e := range c.Logprobs.Content {
				t.tokens = append(t.tokens, e.Token)
				t.lps = append(t.lps, e.Logprob)
				t.tops = append(t.tops, e.TopLogprobs)
				t.bytes = append(t.bytes, e.Bytes)
				if e.Bytes != nil {
					t.anyBytes = true
				}
			}
		}
		if c.FinishReason != nil && *c.FinishReason != "" {
			t.finish = *c.FinishReason
		}
	}
}

func (t *tee) output() audit.Output {
	o := audit.Output{
		Text:          t.text.String(),
		Tokens:        t.tokens,
		TokenLogprobs: t.lps,
		TopLogprobs:   t.tops,
		FinishReason:  t.finish,
	}
	if t.toolCalls && o.FinishReason == "" {
		o.FinishReason = "tool_calls"
	}
	if t.anyBytes {
		o.TokenBytes = t.bytes
	}
	if o.Tokens == nil {
		o.Tokens = []string{}
		o.TokenLogprobs = []float64{}
		o.TopLogprobs = [][]audit.TopLogprob{}
	}
	return o
}

// stripLogprobs removes `logprobs` from every choice of a response or chunk
// (GW-5: injected logprobs are not shown to clients that did not ask).
func stripLogprobs(raw []byte) ([]byte, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, err
	}
	chRaw, ok := top["choices"]
	if !ok {
		return raw, nil
	}
	var choices []map[string]json.RawMessage
	if err := json.Unmarshal(chRaw, &choices); err != nil {
		return nil, err
	}
	changed := false
	for _, c := range choices {
		if _, ok := c["logprobs"]; ok {
			delete(c, "logprobs")
			changed = true
		}
	}
	if !changed {
		return raw, nil
	}
	nb, err := json.Marshal(choices)
	if err != nil {
		return nil, err
	}
	top["choices"] = nb
	return json.Marshal(top)
}
