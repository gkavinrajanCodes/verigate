// Package config loads verigate.yaml for the gateway.
package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// Gateway holds hot-path settings.
type Gateway struct {
	Listen                string   `yaml:"listen"`
	AuditSecretEnv        string   `yaml:"audit_secret_env"`
	QueueURL              string   `yaml:"queue_url"`
	QueueKey              string   `yaml:"queue_key"`
	JobTTLSeconds         int      `yaml:"job_ttl_seconds"`
	EnqueueBuffer         int      `yaml:"enqueue_buffer"`
	UpstreamTimeoutSecs   int      `yaml:"upstream_timeout_seconds"`
	AuditOptOutKeyHashes  []string `yaml:"audit_opt_out_key_hashes"`
}

// Provider is an upstream OpenAI-compatible endpoint.
type Provider struct {
	Name                string            `yaml:"-"`
	BaseURL             string            `yaml:"base_url"`
	APIKeyEnv           string            `yaml:"api_key_env"`
	Models              map[string]string `yaml:"models"` // alias -> provider model id
	RequestLogprobs     bool              `yaml:"request_logprobs"`
	AuditRate           *float64          `yaml:"audit_rate"`
	PinUpstreamProvider string            `yaml:"pin_upstream_provider"`
	Trusted             bool              `yaml:"trusted"`

	apiKey string
}

// APIKey returns the provider API key resolved from the environment.
func (p *Provider) APIKey() string { return p.apiKey }

// Rate returns the audit rate, defaulting to 0.05 (GW-7).
func (p *Provider) Rate() float64 {
	if p.AuditRate == nil {
		return 0.05
	}
	return *p.AuditRate
}

// Config is the subset of verigate.yaml used by the gateway.
type Config struct {
	Gateway   Gateway              `yaml:"gateway"`
	Providers map[string]*Provider `yaml:"providers"`

	AuditSecret []byte `yaml:"-"`
	// providerOrder is a deterministic (sorted) list of provider names.
	providerOrder []string
	optOut        map[string]struct{}
}

// Load reads and validates a config file.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

// Parse parses YAML bytes and resolves environment variables.
func Parse(raw []byte) (*Config, error) {
	var c Config
	if err := yaml.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	g := &c.Gateway
	if g.Listen == "" {
		g.Listen = "127.0.0.1:8080"
	}
	if g.AuditSecretEnv == "" {
		g.AuditSecretEnv = "VERIGATE_AUDIT_SECRET"
	}
	if g.QueueKey == "" {
		g.QueueKey = "verigate:audits"
	}
	if g.JobTTLSeconds <= 0 {
		g.JobTTLSeconds = 86400
	}
	if g.EnqueueBuffer <= 0 {
		g.EnqueueBuffer = 4096
	}
	if g.UpstreamTimeoutSecs <= 0 {
		g.UpstreamTimeoutSecs = 300
	}
	c.AuditSecret = []byte(os.Getenv(g.AuditSecretEnv))

	if len(c.Providers) == 0 {
		return nil, fmt.Errorf("config: no providers defined")
	}
	for name, p := range c.Providers {
		if p == nil {
			return nil, fmt.Errorf("config: provider %q is empty", name)
		}
		p.Name = name
		p.BaseURL = strings.TrimRight(p.BaseURL, "/")
		if p.BaseURL == "" {
			return nil, fmt.Errorf("config: provider %q has no base_url", name)
		}
		if r := p.Rate(); r < 0 || r > 1 {
			return nil, fmt.Errorf("config: provider %q audit_rate %v outside [0,1]", name, r)
		}
		if p.APIKeyEnv != "" {
			p.apiKey = os.Getenv(p.APIKeyEnv)
		}
		c.providerOrder = append(c.providerOrder, name)
	}
	sort.Strings(c.providerOrder)

	c.optOut = map[string]struct{}{}
	for _, h := range g.AuditOptOutKeyHashes {
		c.optOut[strings.ToLower(strings.TrimSpace(h))] = struct{}{}
	}
	return &c, nil
}

// Resolve picks the provider for a model alias. If explicit is non-empty it
// must name a provider that serves the alias (GW-2).
func (c *Config) Resolve(alias, explicit string) (*Provider, error) {
	if explicit != "" {
		p, ok := c.Providers[explicit]
		if !ok {
			return nil, fmt.Errorf("unknown provider %q", explicit)
		}
		if _, ok := p.Models[alias]; !ok {
			return nil, fmt.Errorf("provider %q does not serve model %q", explicit, alias)
		}
		return p, nil
	}
	for _, name := range c.providerOrder {
		p := c.Providers[name]
		if _, ok := p.Models[alias]; ok {
			return p, nil
		}
	}
	return nil, fmt.Errorf("no provider configured for model %q", alias)
}

// ProviderNames returns provider names in deterministic order.
func (c *Config) ProviderNames() []string { return c.providerOrder }

// OptedOut reports whether a client bearer token opted out of auditing.
func (c *Config) OptedOut(bearer string) bool {
	if bearer == "" || len(c.optOut) == 0 {
		return false
	}
	sum := sha256.Sum256([]byte(bearer))
	_, ok := c.optOut[hex.EncodeToString(sum[:])]
	return ok
}
