"""Output token-ID recovery (VF-3).

Providers return decoded token *strings* (and, in the OpenAI format, often the
raw UTF-8 `bytes` of each token). We map those back to reference vocabulary
IDs. If that is impossible we re-tokenize the output text and record which
path was used, so mismatched audits can be excluded from the tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

PROVIDER = "provider_tokens"
RETOKENIZED = "retokenized"
MISMATCH = "mismatch"


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE byte↔unicode table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_ENC = _bytes_to_unicode()
_BYTES_PREFIX = re.compile(r"^bytes:((?:\\x[0-9a-fA-F]{2})+)$")


@dataclass
class Recovered:
    ids: list[int]
    method: str
    provider_logprobs: Optional[list[Optional[float]]]  # aligned with ids, only for PROVIDER


class TokenRecovery:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.vocab: dict[str, int] = tokenizer.get_vocab()
        sample = list(self.vocab.keys())[:5000]
        self.byte_level = any("Ġ" in t or "Ċ" in t for t in sample)
        self.sentencepiece = (not self.byte_level) and any(t.startswith("▁") for t in sample)

    # ------------------------------------------------------------------
    def _token_bytes(self, tok: str, raw: Optional[list[int]]) -> bytes:
        if raw is not None:
            return bytes(raw)
        m = _BYTES_PREFIX.match(tok)
        if m:  # vLLM-style "bytes:\xe4\xbd" for partial UTF-8 tokens
            return bytes(int(h, 16) for h in m.group(1).split("\\x")[1:])
        return tok.encode("utf-8")

    def _lookup(self, b: bytes) -> Optional[int]:
        if self.byte_level:
            return self.vocab.get("".join(_BYTE_ENC[x] for x in b))
        if self.sentencepiece:
            try:
                s = b.decode("utf-8")
            except UnicodeDecodeError:
                s = None
            if s is not None:
                i = self.vocab.get(s.replace(" ", "▁"))
                if i is not None:
                    return i
            if len(b) == 1:
                return self.vocab.get(f"<0x{b[0]:02X}>")
            return None
        try:
            return self.vocab.get(b.decode("utf-8"))
        except UnicodeDecodeError:
            return None

    # ------------------------------------------------------------------
    def recover(self, output: dict) -> Recovered:
        text: str = output.get("text") or ""
        tokens: list[str] = output.get("tokens") or []
        raw_bytes = output.get("token_bytes") or None
        lps = output.get("token_logprobs") or None

        if tokens:
            ids: list[int] = []
            joined = bytearray()
            ok = True
            for i, t in enumerate(tokens):
                rb = raw_bytes[i] if raw_bytes and i < len(raw_bytes) else None
                b = self._token_bytes(t, rb)
                tid = self._lookup(b)
                if tid is None:
                    ok = False
                    break
                ids.append(tid)
                joined += b
            if ok and bytes(joined).decode("utf-8", errors="replace") == text:
                plp = list(lps[: len(ids)]) if lps else None
                return Recovered(ids, PROVIDER, plp)
            # Provider tokens could not be reconciled with the reference
            # vocabulary or the text: score the text, but flag the audit.
            return Recovered(self._retok(text), MISMATCH, None)

        return Recovered(self._retok(text), RETOKENIZED, None)

    def _retok(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))
