"""Test that compute_slice_for_sample correctly masks padding tokens.

Compares against an eager-attention reference that has the SAME 4D bias
applied (causal + padding). Padding keys should get 0 probability mass.
"""
import sys
sys.path.insert(0, "/workspace/verl")

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
from transformers import AutoConfig

from verl.workers.actor.attention_capture import (
    AttentionSliceCapturer,
    compute_slice_for_sample,
)


def main():
    torch.manual_seed(42)
    device = "cuda"

    full_cfg = AutoConfig.from_pretrained(
        "/home/uqbyuan3/data_distillation/models/huggingface/Qwen2.5-VL-7B-Instruct"
    )
    text_cfg = full_cfg.text_config
    text_cfg._attn_implementation = "eager"
    attn = Qwen2_5_VLAttention(text_cfg, layer_idx=0).to(device).to(torch.float32).eval()

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.layer = m
    wrap = Wrap(attn).to(device)

    B, T = 2, 32
    H = text_cfg.hidden_size
    hidden = torch.randn(B, T, H, device=device)
    pos1d = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    position_ids = pos1d.unsqueeze(0).expand(3, -1, -1).contiguous()
    cos, sin = attn.rotary_emb(hidden, position_ids)

    # Per-sample valid mask: sample 0 has left-pad of 3 + right-pad of 5,
    # sample 1 has left-pad of 0 + right-pad of 8.
    valid_mask_2d = torch.ones(B, T, dtype=torch.bool, device=device)
    valid_mask_2d[0, :3] = False
    valid_mask_2d[0, T - 5:] = False
    valid_mask_2d[1, T - 8:] = False

    # Build 4D additive bias matching valid mask + causal.
    neg_inf = torch.finfo(torch.float32).min
    key_pos = torch.arange(T, device=device)
    causal = key_pos[None, :] <= key_pos[:, None]  # (T_q, T_k) lower-tri
    bias = torch.zeros(B, 1, T, T, device=device)
    for b in range(B):
        allowed = causal & valid_mask_2d[b][None, :]
        bias[b, 0] = torch.where(allowed, torch.zeros((), device=device),
                                 torch.full((), neg_inf, device=device))

    # Reference run with eager
    with torch.no_grad():
        out = attn(hidden, attention_mask=bias, position_ids=pos1d,
                   position_embeddings=(cos, sin), output_attentions=True)
    attn_ref = out[1]

    # Hook run
    with AttentionSliceCapturer(wrap) as cap:
        with torch.no_grad():
            _ = attn(hidden, attention_mask=bias, position_ids=pos1d,
                     position_embeddings=(cos, sin), output_attentions=False)
        captures = cap.get_captures()

    # Probe several queries on each sample
    for b in range(B):
        # Pick valid queries within the unpadded region
        query_idx = torch.where(valid_mask_2d[b])[0][3:6]  # 3 mid queries
        visual_idx = torch.where(valid_mask_2d[b])[0][:4]  # first 4 valid keys
        slice_hook = compute_slice_for_sample(
            captures, sample_idx=b,
            query_indices=query_idx, visual_indices=visual_idx,
            key_valid_mask=valid_mask_2d[b],
        )
        slice_ref = attn_ref[b][:, query_idx][:, :, visual_idx].float().mean(dim=0)
        diff = (slice_hook - slice_ref).abs().max().item()
        print(f"sample {b}: T_q={query_idx.numel()} V={visual_idx.numel()} max abs diff = {diff:.3e}")
        assert diff < 1e-5, f"mismatch on sample {b}: {diff}"

    print("PASS: padding-aware slice matches eager reference.")


if __name__ == "__main__":
    main()
