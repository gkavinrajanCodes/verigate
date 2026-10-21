package audit

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestSelectionRate(t *testing.T) {
	secret := []byte("test-secret")
	const n = 100_000
	for _, r := range []float64{0.01, 0.05, 0.10, 0.5} {
		hits := 0
		for i := 0; i < n; i++ {
			if Selected(secret, fmt.Sprintf("req_%08d", i), r) {
				hits++
			}
		}
		got := float64(hits) / n
		if math.Abs(got-r) > 0.005 {
			t.Errorf("rate %.2f: observed %.4f, outside ±0.5%%", r, got)
		}
	}
}

func TestSelectionDeterministicAndSecretDependent(t *testing.T) {
	a := SelectionValue([]byte("s1"), "req_x")
	if a != SelectionValue([]byte("s1"), "req_x") {
		t.Fatal("selection must be deterministic")
	}
	if a == SelectionValue([]byte("s2"), "req_x") {
		t.Fatal("selection must depend on the secret")
	}
	if a < 0 || a >= 1 {
		t.Fatalf("value %v outside [0,1)", a)
	}
}

func TestNewIDUniqueSortable(t *testing.T) {
	seen := map[string]bool{}
	prev := ""
	for i := 0; i < 10000; i++ {
		id := NewID("req_")
		if len(id) != 30 || !strings.HasPrefix(id, "req_") {
			t.Fatalf("bad id %q", id)
		}
		if seen[id] {
			t.Fatalf("duplicate id %q", id)
		}
		seen[id] = true
		if prev != "" && id[4:14] < prev[4:14] {
			t.Fatalf("timestamp prefix not monotonic: %s < %s", id, prev)
		}
		prev = id
	}
}

type failingPusher struct{}

func (failingPusher) Push(context.Context, string, []byte) error { return errors.New("down") }

func TestEnqueuerDropsOnQueueFailure(t *testing.T) {
	e := NewEnqueuer(failingPusher{}, 8, 1)
	for i := 0; i < 5; i++ {
		e.Submit(&Job{AuditID: fmt.Sprint(i)})
	}
	e.Close()
	if e.Dropped.Load() != 5 || e.Enqueued.Load() != 0 {
		t.Fatalf("dropped=%d enqueued=%d", e.Dropped.Load(), e.Enqueued.Load())
	}
}

type blockingPusher struct{ release chan struct{} }

func (b blockingPusher) Push(context.Context, string, []byte) error { <-b.release; return nil }

func TestEnqueuerNeverBlocks(t *testing.T) {
	bp := blockingPusher{release: make(chan struct{})}
	e := NewEnqueuer(bp, 2, 1)
	start := time.Now()
	for i := 0; i < 1000; i++ {
		e.Submit(&Job{AuditID: fmt.Sprint(i)})
	}
	if d := time.Since(start); d > 50*time.Millisecond {
		t.Fatalf("Submit blocked for %v", d)
	}
	if e.Dropped.Load() < 990 {
		t.Fatalf("expected overflow drops, got %d", e.Dropped.Load())
	}
	close(bp.release)
	e.Close()
}

// fakeRedis is a tiny RESP server recording commands.
type fakeRedis struct {
	ln   net.Listener
	mu   sync.Mutex
	cmds [][]string
}

func startFakeRedis(t *testing.T) *fakeRedis {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	f := &fakeRedis{ln: ln}
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go f.serve(c)
		}
	}()
	return f
}

func (f *fakeRedis) serve(c net.Conn) {
	defer c.Close()
	r := bufio.NewReader(c)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return
		}
		var n int
		fmt.Sscanf(line, "*%d", &n)
		args := make([]string, n)
		for i := 0; i < n; i++ {
			var l int
			hdr, _ := r.ReadString('\n')
			fmt.Sscanf(hdr, "$%d", &l)
			buf := make([]byte, l+2)
			if _, err := ioReadFull(r, buf); err != nil {
				return
			}
			args[i] = string(buf[:l])
		}
		f.mu.Lock()
		f.cmds = append(f.cmds, args)
		f.mu.Unlock()
		if strings.EqualFold(args[0], "LPUSH") {
			c.Write([]byte(":1\r\n"))
		} else {
			c.Write([]byte("+OK\r\n"))
		}
	}
}

func ioReadFull(r *bufio.Reader, buf []byte) (int, error) {
	n := 0
	for n < len(buf) {
		m, err := r.Read(buf[n:])
		n += m
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func TestRedisPusher(t *testing.T) {
	f := startFakeRedis(t)
	defer f.ln.Close()
	p, err := NewRedisPusher("redis://:pw@"+f.ln.Addr().String()+"/2", "verigate:audits", 86400)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Push(context.Background(), "aud_1", []byte(`{"a":1}`)); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	want := [][]string{
		{"AUTH", "pw"},
		{"SELECT", "2"},
		{"SET", "verigate:audits:job:aud_1", `{"a":1}`, "EX", "86400"},
		{"LPUSH", "verigate:audits", "aud_1"},
	}
	if fmt.Sprint(f.cmds) != fmt.Sprint(want) {
		t.Fatalf("commands:\n got %v\nwant %v", f.cmds, want)
	}
}
