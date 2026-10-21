// Package audit implements post-hoc audit selection, audit job construction
// and the non-blocking enqueue path.
package audit

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"sync"
	"time"
)

// crockford is the Crockford base32 alphabet used for ULID-style IDs.
const crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

var idMu sync.Mutex

// NewID returns a ULID-style identifier (48-bit ms timestamp + 80 random bits)
// with the given prefix, e.g. "req_01J...". IDs sort by creation time.
func NewID(prefix string) string {
	var b [16]byte
	ms := uint64(time.Now().UnixMilli())
	b[0] = byte(ms >> 40)
	b[1] = byte(ms >> 32)
	b[2] = byte(ms >> 24)
	b[3] = byte(ms >> 16)
	b[4] = byte(ms >> 8)
	b[5] = byte(ms)
	idMu.Lock()
	_, _ = rand.Read(b[6:])
	idMu.Unlock()
	return prefix + encode(b)
}

func encode(b [16]byte) string {
	// 128 bits -> 26 base32 chars (first char carries 3 bits).
	hi := binary.BigEndian.Uint64(b[:8])
	lo := binary.BigEndian.Uint64(b[8:])
	out := make([]byte, 26)
	for i := 25; i >= 0; i-- {
		out[i] = crockford[lo&31]
		lo = (lo >> 5) | (hi << 59)
		hi >>= 5
	}
	return string(out)
}

// SelectionValue maps HMAC-SHA256(secret, requestID) to a uniform value in
// [0,1) using the top 53 bits of the MAC (exactly representable in float64).
func SelectionValue(secret []byte, requestID string) float64 {
	m := hmac.New(sha256.New, secret)
	m.Write([]byte(requestID))
	sum := m.Sum(nil)
	v := binary.BigEndian.Uint64(sum[:8]) >> 11
	return float64(v) / float64(uint64(1)<<53)
}

// Selected reports whether a completed request should be audited (GW-6).
// It is evaluated only after the response has finished, so the upstream
// request is identical whether or not it is audited (defense against A7).
func Selected(secret []byte, requestID string, rate float64) bool {
	if rate <= 0 {
		return false
	}
	if rate >= 1 {
		return true
	}
	return SelectionValue(secret, requestID) < rate
}
