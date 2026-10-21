package audit

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Pusher stores one serialized job on the queue.
type Pusher interface {
	Push(ctx context.Context, auditID string, payload []byte) error
}

// ---------------------------------------------------------------------------
// Minimal RESP (Redis protocol) client. Works with PulseDB, Redis, Valkey,
// KeyDB, etc. Only the handful of commands the gateway needs are supported.
// ---------------------------------------------------------------------------

// RedisPusher writes each job as `SET <key>:job:<id> <json> EX <ttl>` and
// pushes the id with `LPUSH <key> <id>` in one pipelined round trip. The
// per-job key gives the 24 h safety-net TTL that a plain list cannot.
type RedisPusher struct {
	addr     string
	username string
	password string
	db       int
	key      string
	ttl      int

	mu         sync.Mutex
	conn       net.Conn
	rd         *bufio.Reader
	lastDialEr time.Time
}

// NewRedisPusher parses a redis:// URL.
func NewRedisPusher(rawURL, key string, ttlSeconds int) (*RedisPusher, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("queue url: %w", err)
	}
	if u.Scheme != "redis" {
		return nil, fmt.Errorf("queue url: unsupported scheme %q (want redis://)", u.Scheme)
	}
	p := &RedisPusher{addr: u.Host, key: key, ttl: ttlSeconds}
	if !strings.Contains(p.addr, ":") {
		p.addr += ":6379"
	}
	if u.User != nil {
		p.username = u.User.Username()
		p.password, _ = u.User.Password()
		if p.password == "" && p.username != "" {
			// redis://secret@host form: treat the user part as the password.
			p.password, p.username = p.username, ""
		}
	}
	if db := strings.Trim(u.Path, "/"); db != "" {
		if p.db, err = strconv.Atoi(db); err != nil {
			return nil, fmt.Errorf("queue url: bad db %q", db)
		}
	}
	return p, nil
}

// JobKey returns the key under which a job body is stored.
func (p *RedisPusher) JobKey(id string) string { return p.key + ":job:" + id }

func writeCmd(w *bufio.Writer, args ...string) {
	fmt.Fprintf(w, "*%d\r\n", len(args))
	for _, a := range args {
		fmt.Fprintf(w, "$%d\r\n", len(a))
		w.WriteString(a)
		w.WriteString("\r\n")
	}
}

func readReply(r *bufio.Reader) error {
	line, err := r.ReadString('\n')
	if err != nil {
		return err
	}
	if len(line) < 3 {
		return errors.New("resp: short reply")
	}
	body := line[1 : len(line)-2]
	switch line[0] {
	case '+', ':':
		return nil
	case '-':
		return fmt.Errorf("resp: %s", body)
	case '$':
		n, _ := strconv.Atoi(body)
		if n >= 0 {
			_, err = io.CopyN(io.Discard, r, int64(n+2))
		}
		return err
	case '*':
		n, _ := strconv.Atoi(body)
		for i := 0; i < n; i++ {
			if err := readReply(r); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("resp: unexpected reply %q", line)
}

func (p *RedisPusher) connect() error {
	if p.conn != nil {
		return nil
	}
	if time.Since(p.lastDialEr) < time.Second {
		return errors.New("queue unavailable (backing off)")
	}
	c, err := net.DialTimeout("tcp", p.addr, 500*time.Millisecond)
	if err != nil {
		p.lastDialEr = time.Now()
		return err
	}
	p.conn, p.rd = c, bufio.NewReader(c)
	w := bufio.NewWriter(c)
	n := 0
	if p.password != "" {
		if p.username != "" {
			writeCmd(w, "AUTH", p.username, p.password)
		} else {
			writeCmd(w, "AUTH", p.password)
		}
		n++
	}
	if p.db != 0 {
		writeCmd(w, "SELECT", strconv.Itoa(p.db))
		n++
	}
	if n > 0 {
		_ = c.SetDeadline(time.Now().Add(2 * time.Second))
		if err := w.Flush(); err != nil {
			p.closeLocked()
			return err
		}
		for i := 0; i < n; i++ {
			if err := readReply(p.rd); err != nil {
				p.closeLocked()
				p.lastDialEr = time.Now()
				return err
			}
		}
	}
	return nil
}

func (p *RedisPusher) closeLocked() {
	if p.conn != nil {
		_ = p.conn.Close()
	}
	p.conn, p.rd = nil, nil
}

// Push implements Pusher.
func (p *RedisPusher) Push(ctx context.Context, auditID string, payload []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.connect(); err != nil {
		return err
	}
	deadline := time.Now().Add(2 * time.Second)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	_ = p.conn.SetDeadline(deadline)
	w := bufio.NewWriter(p.conn)
	writeCmd(w, "SET", p.JobKey(auditID), string(payload), "EX", strconv.Itoa(p.ttl))
	writeCmd(w, "LPUSH", p.key, auditID)
	if err := w.Flush(); err != nil {
		p.closeLocked()
		return err
	}
	for i := 0; i < 2; i++ {
		if err := readReply(p.rd); err != nil {
			p.closeLocked()
			return err
		}
	}
	return nil
}

// Close closes the connection.
func (p *RedisPusher) Close() {
	p.mu.Lock()
	p.closeLocked()
	p.mu.Unlock()
}

// ---------------------------------------------------------------------------
// Enqueuer: a bounded, non-blocking hand-off from the hot path (GW-8).
// ---------------------------------------------------------------------------

// Enqueuer accepts jobs without ever blocking the caller. Jobs are dropped
// (and counted) when the buffer is full or the queue is unreachable.
type Enqueuer struct {
	ch       chan *Job
	pusher   Pusher
	wg       sync.WaitGroup
	Enqueued atomic.Uint64
	Dropped  atomic.Uint64
	closed   atomic.Bool
}

// NewEnqueuer starts `workers` background goroutines.
func NewEnqueuer(p Pusher, buffer, workers int) *Enqueuer {
	if workers <= 0 {
		workers = 1
	}
	e := &Enqueuer{ch: make(chan *Job, buffer), pusher: p}
	for i := 0; i < workers; i++ {
		e.wg.Add(1)
		go e.run()
	}
	return e
}

// Submit hands off a job; it never blocks.
func (e *Enqueuer) Submit(j *Job) bool {
	if e == nil || e.closed.Load() {
		return false
	}
	select {
	case e.ch <- j:
		return true
	default:
		e.Dropped.Add(1)
		return false
	}
}

func (e *Enqueuer) run() {
	defer e.wg.Done()
	var lastLog time.Time
	for j := range e.ch {
		j.Finalize()
		payload, err := json.Marshal(j)
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			err = e.pusher.Push(ctx, j.AuditID, payload)
			cancel()
		}
		if err != nil {
			e.Dropped.Add(1)
			if time.Since(lastLog) > 10*time.Second {
				log.Printf("audit: dropping audit job (queue error: %v)", err)
				lastLog = time.Now()
			}
			continue
		}
		e.Enqueued.Add(1)
	}
}

// Close drains the buffer and stops workers.
func (e *Enqueuer) Close() {
	if e.closed.Swap(true) {
		return
	}
	close(e.ch)
	e.wg.Wait()
}
