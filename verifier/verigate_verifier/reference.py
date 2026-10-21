"""Reference model loading, pinned chat templates, and batched prefill (VF-1, VF-2, VF-4, VF-6)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import torch

from .config import ReferenceSpec

log = logging.getLogger("verigate.reference")

_DTYPES = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}


def pick_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


class ChatTemplateError(RuntimeError):
    pass


class ReferenceModel:
    """A trusted copy of the claimed model at pinned weights and precision."""

    def __init__(self, spec: ReferenceSpec, device: str = "auto", model=None, tokenizer=None):
        self.spec = spec
        self.device = pick_device(device)
        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("loading reference %s @ %s (%s) on %s", spec.hf_repo, spec.revision, spec.dtype, self.device)
            tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, revision=spec.revision)
            dtype = _DTYPES.get(spec.dtype, torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(spec.hf_repo, revision=spec.revision, dtype=dtype)
            model.to(self.device)
        self.model = model.eval()
        self.tok = tokenizer
        self._template: Optional[str] = None
        if spec.chat_template not in ("official", "", None):
            p = Path(spec.chat_template)
            self._template = p.read_text(encoding="utf-8") if p.exists() else spec.chat_template
        elif not getattr(self.tok, "chat_template", None):
            raise ChatTemplateError(
                f"{spec.hf_repo}@{spec.revision} has no chat template; refusing to score (VF-2)"
            )
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self._base = getattr(self.model, self.model.base_model_prefix, None)
        self._head = self.model.get_output_embeddings()

    # ------------------------------------------------------------------
    def render(self, messages: Sequence[dict]) -> str:
        """Render the prompt exactly as the official serving stack does."""
        kw = {"chat_template": self._template} if self._template else {}
        return self.tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True, **kw)

    def prompt_ids(self, messages: Sequence[dict]) -> list[int]:
        text = self.render(messages)
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, items: Sequence[tuple[list[int], list[int]]]) -> list[torch.Tensor]:
        """Single forward pass over prompt+output for a batch of requests.

        Returns, per item, float32 logits [n_out, V] where row t is the
        reference distribution for output token t given prompt + y_<t.
        Only output positions are projected through the LM head, which keeps
        memory proportional to n_out × V rather than seq_len × V.
        """
        if not items:
            return []
        seqs = [p + o for p, o in items]
        L = max(len(s) for s in seqs)
        B = len(seqs)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, s in enumerate(seqs):  # right padding keeps absolute positions intact
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)

        if self._base is not None and self._head is not None:
            hidden = self._base(input_ids=input_ids, attention_mask=attn, use_cache=False).last_hidden_state
            outs = []
            for i, (p, o) in enumerate(items):
                if not o:
                    outs.append(torch.empty((0, self._head.out_features), device=self.device))
                    continue
                h = hidden[i, len(p) - 1 : len(p) - 1 + len(o)]
                outs.append(self._head(h).float())
            return outs

        logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        return [logits[i, len(p) - 1 : len(p) - 1 + len(o)].float() for i, (p, o) in enumerate(items)]
