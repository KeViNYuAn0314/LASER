"""Numerical correctness test for AttentionSliceCapturer + compute_slice_for_sample.

Strategy:
  1. Build one Qwen2_5_VLAttention layer with random weights.
  2. Run it with output_attentions=True (eager) -> ground-truth attn_weights.
  3. Run same forward through AttentionSliceCapturer (output_attentions=False).
  4. Compute the slice via compute_slice_for_sample.
  5. Compare against the head-averaged ground-truth slice.

Tolerance: fp32 throughout; expect ~1e-5 max abs diff.
"""

import sys
sys.path.insert(0, "/workspace/verl")  # mounted by apptainer

import torch
import torch.nn as nn

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLTextConfig,
)
from transformers import AutoConfig

from verl.workers.actor.attention_capture import (
    AttentionSliceCapturer,
    compute_slice_for_sample,
)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_cfg = AutoConfig.from_pretrained(
        "/home/uqbyuan3/data_distillation/models/huggingface/Qwen2.5-VL-7B-Instruct"
    )
    text_cfg: Qwen2_5_VLTextConfig = full_cfg.text_config
    text_cfg._attn_implementation = "eager"
    print(
        f"hidden={text_cfg.hidden_size} heads={text_cfg.num_attention_heads} "
        f"kv_heads={text_cfg.num_key_value_heads} head_dim={text_cfg.hidden_size // text_cfg.num_attention_heads} "
        f"layer_types[0]={text_cfg.layer_types[0]}"
    )

    attn = Qwen2_5_VLAttention(text_cfg, layer_idx=0).to(device).to(torch.float32).eval()

    # Wrap so AttentionSliceCapturer can find the attention module via .modules().
    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.layer = m
    wrap = Wrap(attn).to(device)

    B, T = 2, 24
    H = text_cfg.hidden_size
    hidden = torch.randn(B, T, H, device=device, dtype=torch.float32)
    # Build mrope position_ids: (3, B, T) — same positions across the three sections
    pos1d = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    position_ids = pos1d.unsqueeze(0).expand(3, -1, -1).contiguous()
    cos, sin = attn.rotary_emb(hidden, position_ids)
    position_embeddings = (cos, sin)

    # Build a 4D causal attention bias matching what the language model would
    # construct internally for eager attention: 0 where causal, -inf otherwise.
    neg_inf = torch.finfo(torch.float32).min
    causal_bool = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    causal_bias = torch.where(
        causal_bool,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), neg_inf, device=device, dtype=torch.float32),
    )
    causal_bias = causal_bias.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T).contiguous()

    # ---- 1. Reference run: output_attentions=True (eager) ----
    with torch.no_grad():
        out_ref = attn(
            hidden,
            attention_mask=causal_bias,
            position_ids=pos1d,  # used by some kernels; eager just uses attention_mask
            position_embeddings=position_embeddings,
            output_attentions=True,
        )
    attn_weights_ref = out_ref[1]  # (B, H_q, T, T)
    print(f"ref attn_weights shape: {tuple(attn_weights_ref.shape)}")

    # ---- 2. Hook run: output_attentions=False, capture Q/K via patcher ----
    with AttentionSliceCapturer(wrap) as cap:
        with torch.no_grad():
            _ = attn(
                hidden,
                attention_mask=causal_bias,
                position_ids=pos1d,
                position_embeddings=position_embeddings,
                output_attentions=False,
            )
        captures = cap.get_captures()
    assert 0 in captures, "Capture missing for layer 0"
    q_cap = captures[0]["q"]
    k_cap = captures[0]["k"]
    print(f"captured q={tuple(q_cap.shape)} k={tuple(k_cap.shape)}")

    # ---- 3. Compute slice for sample 0, queries={5,11,17}, visual={2,7,15,21} ----
    sample_idx = 0
    query_idx = torch.tensor([5, 11, 17], dtype=torch.long, device=device)
    visual_idx = torch.tensor([2, 7, 15, 21], dtype=torch.long, device=device)
    key_valid = torch.ones(T, dtype=torch.bool, device=device)

    slice_hook = compute_slice_for_sample(
        captures=captures,
        sample_idx=sample_idx,
        query_indices=query_idx,
        visual_indices=visual_idx,
        key_valid_mask=key_valid,
    )  # (T_q=3, V=4) fp32
    print(f"slice_hook shape: {tuple(slice_hook.shape)}")

    # Ground truth: head-averaged attention probs from query positions to visual
    # positions for sample 0 — single layer, so this is the layer-averaged value.
    slice_ref = attn_weights_ref[sample_idx][:, query_idx][:, :, visual_idx].float().mean(dim=0)
    print(f"slice_ref  shape: {tuple(slice_ref.shape)}")

    diff = (slice_hook - slice_ref).abs()
    print(f"max abs diff: {diff.max().item():.3e}")
    print(f"hook[0]: {slice_hook[0].tolist()}")
    print(f"ref [0]: {slice_ref[0].tolist()}")

    assert diff.max().item() < 1e-4, f"slice mismatch! max diff = {diff.max().item()}"
    print("PASS: hook slice matches output_attentions=True ground truth.")


if __name__ == "__main__":
    main()
