// Command verigate-gateway runs the Verigate OpenAI-compatible gateway.
package main

import (
	"context"
	"crypto/rand"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"verigate/gateway/internal/audit"
	"verigate/gateway/internal/config"
	"verigate/gateway/internal/metrics"
	"verigate/gateway/internal/proxy"
)

func main() {
	cfgPath := flag.String("config", "verigate.yaml", "path to verigate.yaml")
	listen := flag.String("listen", "", "override gateway.listen")
	flag.Parse()

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	if *listen != "" {
		cfg.Gateway.Listen = *listen
	}
	if len(cfg.AuditSecret) == 0 {
		// Still unpredictable to providers, but selection is not reproducible
		// across restarts. Never log the secret itself.
		cfg.AuditSecret = make([]byte, 32)
		_, _ = rand.Read(cfg.AuditSecret)
		log.Printf("WARNING: %s is not set; using an ephemeral random audit secret", cfg.Gateway.AuditSecretEnv)
	}

	m := metrics.New()
	var enq *audit.Enqueuer
	if cfg.Gateway.QueueURL != "" {
		pusher, err := audit.NewRedisPusher(cfg.Gateway.QueueURL, cfg.Gateway.QueueKey, cfg.Gateway.JobTTLSeconds)
		if err != nil {
			log.Fatalf("queue: %v", err)
		}
		defer pusher.Close()
		enq = audit.NewEnqueuer(pusher, cfg.Gateway.EnqueueBuffer, 2)
		m.AuditsEnqueued = func() uint64 { return enq.Enqueued.Load() }
		m.AuditsDropped = func() uint64 { return enq.Dropped.Load() }
	} else {
		log.Printf("WARNING: gateway.queue_url is empty; auditing disabled")
	}

	srv := proxy.New(cfg, enq, m)
	hs := &http.Server{
		Addr:              cfg.Gateway.Listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		log.Printf("verigate gateway listening on %s (%d providers)", cfg.Gateway.Listen, len(cfg.Providers))
		for _, name := range cfg.ProviderNames() {
			p := cfg.Providers[name]
			log.Printf("  provider %-16s audit_rate=%.3f logprobs=%v -> %s", name, p.Rate(), p.RequestLogprobs, p.BaseURL)
		}
		if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("listen: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	log.Printf("shutting down")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = hs.Shutdown(ctx)
	if enq != nil {
		enq.Close()
	}
}
