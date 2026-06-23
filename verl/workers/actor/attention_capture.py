"""Hook-based post-RoPE Q/K capture for Qwen2.5-VL.

Replaces ``output_attentions=True`` (which forces eager attention and
materialises a full ``(B, H, T, T)`` tensor for every layer) with an
instance-level monkey patch that mirrors :class:`Qwen2_5_VLAttention.forward`
from transformers 4.57.6 byte-for-byte and additionally stashes post-RoPE
``query_states`` and ``key_states`` into ``module._captured_qk``.

The reward code can then call :func:`compute_slice_for_sample` to materialise
only the small ``(T_q, V)`` slice it needs (queries restricted to the response
range, keys restricted to visual token positions), instead of the full
``(T, T)`` matrix.
"""

from __future__ import annotations

import types
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_text_attention_modules(root: nn.Module) -> dict:
    """Return ``{layer_idx: attn_module}`` for every ``Qwen2_5_VLAttention`` in
    the tree. The vision tower's ``Qwen2_5_VLVisionAttention`` is skipped (its
    class name does not match and it has no ``layer_idx`` attribute)."""
    found: dict[int, nn.Module] = {}
    for module in root.modules():
        if module.__class__.__name__ == "Qwen2_5_VLAttention" and hasattr(module, "layer_idx"):
            found[int(module.layer_idx)] = module
    return found


def _patched_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Mirrors transformers 4.57.6 ``Qwen2_5_VLAttention.forward`` and stashes
    post-RoPE Q, K into ``self._captured_qk``."""
    # Imported lazily so this file is importable in non-transformers contexts.
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        ALL_ATTENTION_FUNCTIONS,
        apply_multimodal_rotary_pos_emb,
        eager_attention_forward,
    )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # Capture post-RoPE Q and K plus the per-layer attention metadata that the
    # slice computation needs (GQA group size and the head_dim**-0.5 scaling).
    # detach() avoids retaining the autograd graph; the calling context is
    # no_grad anyway, so this is just defensive.
    self._captured_qk = {
        "q": query_states.detach(),
        "k": key_states.detach(),
        "num_key_value_groups": int(self.num_key_value_groups),
        "scaling": float(self.scaling),
    }

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


class AttentionSliceCapturer:
    """Context manager that installs a per-instance forward patch on selected
    text decoder attention layers so they save post-RoPE Q and K.

    The actual attention kernel (FlashAttention / SDPA / eager) selected by
    the model's config is unchanged — we only inject a side-channel capture.
    """

    def __init__(self, model: nn.Module, layer_indices: Optional[Iterable[int]] = None):
        self._all_modules = find_text_attention_modules(model)
        if not self._all_modules:
            raise RuntimeError(
                "No Qwen2_5_VLAttention modules found while installing AttentionSliceCapturer."
            )
        if layer_indices is None:
            self._target_indices = sorted(self._all_modules.keys())
        else:
            self._target_indices = [int(i) for i in layer_indices if int(i) in self._all_modules]
        # Tracks (module, prior_instance_forward_or_None) for restoration.
        self._installed: list[tuple[nn.Module, Optional[object]]] = []

    @property
    def captured_layer_indices(self) -> list:
        return list(self._target_indices)

    def __enter__(self) -> "AttentionSliceCapturer":
        for idx in self._target_indices:
            module = self._all_modules[idx]
            prior = module.__dict__.pop("forward", None)
            module.forward = types.MethodType(_patched_forward, module)
            self._installed.append((module, prior))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for module, prior in self._installed:
            if "forward" in module.__dict__:
                del module.__dict__["forward"]
            if prior is not None:
                module.forward = prior
            if hasattr(module, "_captured_qk"):
                del module._captured_qk
        self._installed = []

    def get_captures(self) -> dict:
        """Returns ``{layer_idx: {'q': Tensor (B, H_q, T, D),
        'k': Tensor (B, H_kv, T, D)}}`` for layers that were actually visited
        during the most recent forward."""
        out: dict[int, dict[str, torch.Tensor]] = {}
        for idx in self._target_indices:
            module = self._all_modules[idx]
            qk = getattr(module, "_captured_qk", None)
            if qk is not None:
                out[idx] = qk
        return out


def _repeat_kv(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent to :func:`transformers...modeling_qwen2_5_vl.repeat_kv`."""
    if n_rep == 1:
        return k
    bsz, h_kv, slen, d = k.shape
    return k[:, :, None, :, :].expand(bsz, h_kv, n_rep, slen, d).reshape(bsz, h_kv * n_rep, slen, d)


def _build_additive_mask(
    query_positions: torch.Tensor,  # (T_q,) long, abs positions in [0, T_k)
    key_valid_mask: torch.Tensor,   # (T_k,) bool, True where the key is real (not padding)
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive attention bias of shape ``(T_q, T_k)``: 0 where the key may be
    attended (causal AND not padding), ``finfo(dtype).min`` otherwise.

    This handles both left- and right-padding because the validity comes from
    the explicit ``key_valid_mask`` rather than a single right-edge cutoff.
    """
    T_k = int(key_valid_mask.shape[0])
    device = key_valid_mask.device
    key_positions = torch.arange(T_k, device=device)
    causal_ok = query_positions[:, None] >= key_positions[None, :]
    allowed = causal_ok & key_valid_mask[None, :]
    neg_inf = torch.finfo(dtype).min
    return torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg_inf, dtype=dtype, device=device),
    )


def compute_slice_for_sample(
    captures: dict,
    sample_idx: int,
    query_indices: torch.Tensor,    # (T_q,) long, abs positions
    visual_indices: torch.Tensor,   # (V,) long, abs positions
    key_valid_mask: torch.Tensor,   # (T_capture,) bool, True where the key token is real
    layer_indices: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Compute layer- and head-averaged attention from ``query_indices`` to
    ``visual_indices`` for one sample, using captured post-RoPE Q/K.

    The math mirrors :func:`eager_attention_forward` from transformers:
    ``softmax((Q @ Kᵀ) * scaling + mask)`` in fp32, then slice and head-mean.

    Returns a ``(T_q, V)`` fp32 tensor. Peak memory is one layer's logits in
    fp32 — ``(1, H_q, T_q, T_capture)`` — plus the running ``(T_q, V)``
    accumulator.
    """
    assert query_indices.dim() == 1 and visual_indices.dim() == 1
    if layer_indices is None:
        layer_indices = sorted(captures.keys())
    else:
        layer_indices = [i for i in layer_indices if i in captures]
    T_q = int(query_indices.numel())
    V = int(visual_indices.numel())
    if not layer_indices or T_q == 0 or V == 0:
        device = next(iter(captures.values()))["q"].device if captures else key_valid_mask.device
        return torch.zeros((T_q, V), dtype=torch.float32, device=device)

    device = captures[layer_indices[0]]["q"].device
    T_capture = int(captures[layer_indices[0]]["q"].shape[-2])
    assert int(key_valid_mask.shape[0]) == T_capture, (
        f"key_valid_mask length {key_valid_mask.shape[0]} != capture length {T_capture}"
    )

    q_idx = query_indices.to(device=device, dtype=torch.long)
    v_idx = visual_indices.to(device=device, dtype=torch.long).clamp_(max=T_capture - 1)
    mask = key_valid_mask.to(device=device, dtype=torch.bool)

    # Built once for this sample; broadcasts over batch and head dims.
    bias = _build_additive_mask(q_idx, mask, dtype=torch.float32)  # (T_q, T_capture)

    accum: Optional[torch.Tensor] = None
    n_layers = 0
    for idx in layer_indices:
        qk = captures[idx]
        q = qk["q"][sample_idx : sample_idx + 1]  # (1, H_q, T_capture, D)
        k = qk["k"][sample_idx : sample_idx + 1]  # (1, H_kv, T_capture, D)
        n_rep = int(qk["num_key_value_groups"])
        scaling = float(qk["scaling"])

        q_sel = q.index_select(2, q_idx)  # (1, H_q, T_q, D)
        k_full = _repeat_kv(k, n_rep)  # (1, H_q, T_capture, D)

        # Eager-attention math in fp32 for softmax numerical stability.
        logits = torch.matmul(q_sel.float(), k_full.float().transpose(-1, -2)) * scaling
        logits = logits + bias  # broadcasts (T_q, T_capture) -> (1, H_q, T_q, T_capture)
        probs = F.softmax(logits, dim=-1)  # (1, H_q, T_q, T_capture)

        visual_probs = probs.index_select(-1, v_idx)  # (1, H_q, T_q, V)
        head_avg = visual_probs.mean(dim=1).squeeze(0)  # (T_q, V)

        accum = head_avg if accum is None else accum + head_avg
        n_layers += 1

        del logits, probs, visual_probs, q_sel, k_full

    return accum / max(n_layers, 1)
