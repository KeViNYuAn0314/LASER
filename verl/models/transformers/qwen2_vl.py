# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import _flash_attention_forward, fa_peft_integration_check
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLAttention,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
)
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
from transformers.cache_utils import Cache
from transformers.utils import is_flash_attn_greater_or_equal
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from typing import Any, Callable, Optional, Union
from transformers.modeling_outputs import BaseModelOutputWithPast

from typing import Tuple


from verl.utils.device import is_npu_available
from verl.utils.transformers_compat import is_transformers_version_in_range
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

import torch.nn.functional as F

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

if is_npu_available:
    from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
    from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask

    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_use_top_left_mask = flash_attn_supports_top_left_mask()

_flash_deterministic_enabled = os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen2-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1405
    """
    spatial_merge_size = processor.image_processor.merge_size
    tokens_per_second = 2
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)  # (3, seqlen)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0

                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


# def prepare_fa2_from_position_ids(
#     query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor
# ):
#     assert position_ids.ndim == 2  # (batch_size, seq_length)
#     query = query.contiguous().view(-1, query.size(-2), query.size(-1))
#     key = key.contiguous().view(-1, key.size(-2), key.size(-1))
#     value = value.contiguous().view(-1, value.size(-2), value.size(-1))
#     position_ids = position_ids.view(-1)
#     cu_seqlens = torch.cat(
#         (
#             (position_ids == 0).nonzero().view(-1).to(torch.int32),
#             torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
#         )
#     )
#     max_length = cu_seqlens.diff().max()  # use cu_seqlens to infer max_length for qwen2vl mrope
#     return (query, key, value, (cu_seqlens, cu_seqlens), (max_length, max_length))
def prepare_fa2_from_position_ids(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor
):
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.view(-1, key.size(-2), key.size(-1))
    value = value.view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
    cu_seqlens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )
    max_length = cu_seqlens.diff().max()  # use cu_seqlens to infer max_length for qwen2vl mrope
    return (query, key, value, indices_q, (cu_seqlens, cu_seqlens), (max_length, max_length))


def _custom_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    deterministic: Optional[bool] = None,
    **kwargs,
):
    """
    Patches flash attention forward to handle 3D position ids in mrope. (3, batch_size, seq_length)
    """
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if _flash_supports_deterministic:
        flash_kwargs["deterministic"] = deterministic if deterministic is not None else _flash_deterministic_enabled

    if kwargs.get("softcap") is not None:
        flash_kwargs["softcap"] = kwargs.pop("softcap")

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype=torch.bfloat16
    )

    # print(f"query states in custom flash attention forward is {query_states}")

    if position_ids is not None:
        assert position_ids.ndim == 2  # (batch_size, seq_length / sp_size)

    sp_size = get_ulysses_sequence_parallel_world_size()
    if sp_size > 1:
        # qkv: (batch_size, seq_length / sp_size, num_head, head_size)
        validate_ulysses_config(query_states.size(2), sp_size)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)
        position_ids_lst = [torch.empty_like(position_ids) for _ in range(sp_size)]
        position_ids = dist.all_gather(position_ids_lst, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.cat(position_ids_lst, dim=-1)  # (batch_size, seq_length)

    if position_ids is not None and query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all():
        batch_size = query_states.size(0)
        q, k, v, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids
        )
        # print(f"in custom flash attention forward q is {q}")
        attn_output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=kwargs.pop("dropout", 0.0),
            softmax_scale=kwargs.pop("softmax_scale", None),
            causal=is_causal,
            **flash_kwargs,
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=is_causal,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            deterministic=deterministic,
            **kwargs,
        )  # do not pass position_ids to old flash_attention_forward

    if sp_size > 1:
        # (batch_size, seq_length, num_head, head_size)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    return attn_output


# def qwen2_vl_attn_forward(
#     self: "Qwen2VLAttention",
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
#     **kwargs,
# ) -> tuple[torch.Tensor, None, None]:
#     from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, repeat_kv

#     bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size
#     query_states = self.q_proj(hidden_states)  # (batch_size, seq_length / sp_size, num_heads * head_size)
#     key_states = self.k_proj(hidden_states)
#     value_states = self.v_proj(hidden_states)

#     query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#     value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#     # Because the input can be padded, the absolute sequence length depends on the max position id.
#     cos, sin = position_embeddings
#     query_states, key_states = apply_multimodal_rotary_pos_emb(
#         query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
#     )
#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)
#     dropout_rate = 0.0 if not self.training else self.attention_dropout

#     sliding_window = None
#     if (
#         self.config.use_sliding_window
#         and getattr(self.config, "sliding_window", None) is not None
#         and self.layer_idx >= self.config.max_window_layers
#     ):
#         sliding_window = self.config.sliding_window


#     # ========== ADD ATTENTION WEIGHTS COMPUTATION ==========
#     attn_weights = None
#     if kwargs.get("output_attentions", False):
#         # Compute attention weights BEFORE transpose
#         # query_states, key_states, value_states are in [batch, heads, seq, dim]
#         scaling = 1.0 / (self.head_dim ** 0.5)
        
#         # Compute attention scores
#         attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling
#         # Shape: [batch, heads, seq, seq]
#         print(f"attn_scores is {attn_scores}")
        
#         # Apply causal mask if needed
#         # if getattr(self, "is_causal", True):
#         #     causal_mask = torch.triu(
#         #         torch.full((q_len, q_len), float('-inf'), 
#         #                   device=query_states.device, 
#         #                   dtype=query_states.dtype),
#         #         diagonal=1
#         #     )
#         #     print(f"causal mask is {causal_mask}")
#         #     attn_scores = attn_scores + causal_mask[None, None, :, :]
        
#         # Apply attention mask if provided
#         if attention_mask is not None:
#             if attention_mask.dim() == 2:
#                 # Expand mask: [batch, seq] -> [batch, 1, 1, seq]
#                 expanded_mask = attention_mask[:, None, None, :].to(dtype=query_states.dtype)
#                 inverted_mask = (1.0 - expanded_mask) * torch.finfo(query_states.dtype).min
#                 attn_scores = attn_scores + inverted_mask
#             elif attention_mask.dim() == 4:
#                 attn_scores = attn_scores + attention_mask
#         elif getattr(self, "is_causal", True):
#             causal_mask = torch.triu(
#                 torch.full((q_len, q_len), float('-inf'), 
#                           device=query_states.device, 
#                           dtype=query_states.dtype),
#                 diagonal=1
#             )
#             print(f"causal mask is {causal_mask}")
#             attn_scores = attn_scores + causal_mask[None, None, :, :]
        
#         # Softmax to get attention weights
#         attn_weights_full = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         # Shape: [batch, heads, seq, seq]
#         print(f"attn_weights_full is {attn_weights_full}")
        
#         # Average across heads for output
#         attn_weights = attn_weights_full.mean(dim=1)
#         # Shape: [batch, seq, seq]
#     # ========== END ATTENTION WEIGHTS COMPUTATION ==========

#     # This is before the transpose
#     q_len = query_states.shape[2]

#     # FA2 uses non-transposed inputs
#     query_states = query_states.transpose(1, 2)
#     key_states = key_states.transpose(1, 2)
#     value_states = value_states.transpose(1, 2)

#     if position_ids.ndim == 3:
#         position_ids = position_ids[0]

#     attn_output = _custom_flash_attention_forward(
#         query_states,
#         key_states,
#         value_states,
#         attention_mask,
#         query_length=q_len,
#         is_causal=getattr(self, "is_causal", True),
#         dropout=dropout_rate,
#         sliding_window=sliding_window,
#         use_top_left_mask=_flash_use_top_left_mask,
#         position_ids=position_ids,  # important: pass position ids
#     )  # (batch_size, seq_length / sp_size, num_head, head_size)
#     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
#     attn_output = self.o_proj(attn_output)
#     if is_transformers_version_in_range(min_version="4.54.0"):
#         return attn_output, attn_weights
#     else:
#         return attn_output, attn_weights, None

@torch.no_grad()
def get_full_attention(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, repeat_kv
        import math
        
        bsz, q_len, _ = hidden_states.size()
        
        # print(f"hidden_states in get_full_attention is {hidden_states}")

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_scores = torch.matmul(
            query_states,  # [batch, heads, seq_len, head_dim]
            key_states.transpose(-2, -1)  # [batch, heads, head_dim, seq_len]
        ) / math.sqrt(self.head_dim)
        # Result: [batch, heads, seq_len, seq_len]

        if attention_mask is not None and attention_mask.dim() == 2:
            # Create causal mask
            causal_mask = torch.triu(
                torch.full((q_len, q_len), float('-inf'), 
                        device=query_states.device, 
                        dtype=query_states.dtype),
                diagonal=1
            )

            expanded_mask = attention_mask[:, None, None, :].to(dtype=query_states.dtype)
            inverted_mask = (1.0 - expanded_mask) * torch.finfo(query_states.dtype).min
            
            combined_mask = causal_mask[None, None, :, :] + inverted_mask
            
            attn_scores = attn_scores + combined_mask
            
        elif attention_mask is not None and attention_mask.dim() == 4:
            attn_scores = attn_scores + attention_mask
        else:
            causal_mask = torch.triu(
                torch.full((q_len, q_len), float('-inf'), device=query_states.device),
                diagonal=1  # Mask out upper triangle (future tokens)
            )
            attn_scores = attn_scores + causal_mask
        attn_scores = attn_scores + causal_mask
        
        # print(f"attn_scores is {attn_scores}")
        
        # Apply softmax to get probabilities
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Result: [batch, heads, seq_len, seq_len]

        full_attention = attn_weights.mean(dim=1)
        # full_attention = attn_weights

        return full_attention

def get_target_dtype(query: torch.Tensor, module: torch.nn.Module) -> torch.dtype:
    """If the query is in float32, return a target dtype compatible with flash attention. Return None otherwise."""
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            # NOTE: `torch.get_autocast_dtype` is there starting from PyTorch 2.4
            return (
                torch.get_autocast_dtype("cuda")
                if hasattr(torch, "get_autocast_dtype")
                else torch.get_autocast_gpu_dtype()
            )
        # Handle the case where the model is quantized
        elif hasattr(module.config, "quantization_config"):
            return module.config.dtype
        else:
            return next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype
    return None

def qwen2_5_vl_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:

    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = get_target_dtype(query, module)

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else module.is_causal

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=flash_attn_supports_top_left_mask(),
        target_dtype=target_dtype,
        attn_implementation=module.config._attn_implementation,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        **kwargs,
    )

    return attn_output, None

def qwen2_5_vl_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, repeat_kv
        bsz, q_len, _ = hidden_states.size()
        
        # start bowen
        full_attention = None
        if output_attentions:
            full_attention = self.get_full_attention(
                hidden_states,
                attention_mask,
                past_key_values,
                cache_position,
                position_embeddings,
            )
        # end bowen

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

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
            # attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_interface = qwen2_5_vl_flash_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, full_attention

def qwen2_5_vl_text_model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # selected layer is the odd layers
        SELECTED_LAYERS = [i for i in range(len(self.layers)) if i % 3 == 0]
        # print(f"SELECTED_LAYERS for attentions are {SELECTED_LAYERS}")

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions and idx in SELECTED_LAYERS:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    deterministic: Optional[bool] = None,
    **kwargs,
):
    """
    Patches flash attention forward to handle 3D position ids in mrope. (3, batch_size, seq_length)
    """
    causal = is_causal if not use_top_left_mask else is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if is_flash_attn_greater_or_equal("2.4.1"):
        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        flash_kwargs["deterministic"] = deterministic

    if (
        flash_attn_varlen_func is not None
        and position_ids is not None
        and query_length != 1
        and not (torch.diff(position_ids[0], dim=-1) >= 0).all()
    ):
        batch_size = query_states.size(0)
        query_states, key_states, value_states, _, cu_seq_lens, max_seq_lens = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids[0]
        )  # remove channel dimension
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        flash_attn_func = flash_attn_varlen_func
        common_attn_kwargs = {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_in_batch_q,
            "max_seqlen_k": max_seqlen_in_batch_k,
            "dropout_p": kwargs.pop("dropout", 0.0),
            "softmax_scale": kwargs.pop("softmax_scale", None),
            **flash_kwargs,
        }

        if flash_attn_func is None:
            # Use transformers >= 4.54
            flash_attn_func = _flash_attention_forward
            specific_attn_kwargs = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "query_length": query_length,
                "is_causal": causal,
            }
        else:
            specific_attn_kwargs = {"causal": causal}

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            **common_attn_kwargs,
            **specific_attn_kwargs,
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
    else:
        if (
            flash_attn_varlen_func is None
            and position_ids is not None
            and query_length != 1
            and not (torch.diff(position_ids[0], dim=-1) >= 0).all()
        ):
            logger.warning_once(
                "flash_attn_varlen_func is not available; falling back to _flash_attention_forward."
                "This may be suboptimal for non-monotonic position_ids in VLM mRoPE."
            )
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=is_causal,
            sliding_window=sliding_window,
            use_top_left_mask=flash_attn_supports_top_left_mask(),
            deterministic=deterministic,
            **kwargs,
        )

    return attn_output

def qwen2_vl_attn_forward(
    self: "Qwen2VLAttention",
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    # output_attentions: bool = False,  # Added parameter
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], None]:  # Modified return type
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, repeat_kv
    
    # start bowen
    full_attention = None
    if kwargs.get("output_attentions", False):
        full_attention = self.get_full_attention(
            hidden_states,
            attention_mask,
            kwargs.get("past_key_value", None),
            kwargs.get("cache_position", None),
            position_embeddings,
        )
    # end bowen

    bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size
    query_states = self.q_proj(hidden_states)  # (batch_size, seq_length / sp_size, num_heads * head_size)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        validate_ulysses_config(self.num_heads, ulysses_sp_size)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
        # (batch_size, num_head / sp_size, seq_length, head_size)
        full_q_len = query_states.size(2)  # full_q_len = seq_length
    else:
        full_q_len = q_len

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    # print(f"first 5 query states before rope: {query_states[0,0,:5,:5]}")
    # print(f"last 5 query states before rope: {query_states[0,0,-5:,-5:]}")
    # print(f"first 5 key states before rope: {key_states[0,0,:5,:5]}")
    # print(f"last 5 key states before rope: {key_states[0,0,-5:,-5:]}")
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    
    
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        full_q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=flash_attn_supports_top_left_mask(),
        position_ids=position_ids,  # important: pass position ids
    )  # (batch_size, seq_length, num_head / sp_size, head_size)
    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    
    # Return attention weights if computed
    if is_transformers_version_in_range(min_version="4.54.0"):
        return attn_output, full_attention
    else:
        return attn_output, full_attention, None



def _get_input_embeds(
    model: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:  # handle mixed text-image data
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds += 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return inputs_embeds, attention_mask


def process_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    if position_ids.ndim != 3 or position_ids.size(0) != 4:
        # we concat the text position ids with the 3D vision position ids by default
        # see https://github.com/huggingface/transformers/pull/39447
        raise ValueError("position_ids should be a 3D tensor of shape (4, batch_size, seq_length).")

    if is_transformers_version_in_range(max_version="4.53.3"):
        # transformers < 4.54.0 only accepts vision position ids, so we discard the text position ids here
        position_ids = position_ids[1:]

    return position_ids


@dataclass
class Qwen2VLCausalLMOutputForPPO(Qwen2VLCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def qwen2_vl_base_forward(
    self: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    kwargs["inputs_embeds"], kwargs["attention_mask"] = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    return self.language_model(input_ids=None, **kwargs)


def qwen2_vl_forward(
    self: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if is_transformers_version_in_range(min_version="4.52.0"):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=process_position_ids(position_ids),
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
    else:
        inputs_embeds, attention_mask = _get_input_embeds(
            self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
        )
        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=process_position_ids(position_ids),
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


def forward_with_normal_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen2VLCausalLMOutputWithPast":
    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    output_attentions = kwargs.get("output_attentions", False)
    # print(f"here output_attentions: {output_attentions}")
    if output_attentions:
        attn_weigths = outputs[1]
    # print(f"here attnetion weights: {attn_weigths if output_attentions else 'None'}")
    logits = self.lm_head(hidden_states)

    return Qwen2VLCausalLMOutputWithPast(
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=attn_weigths if output_attentions else None,
    )


def forward_with_torch_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> tuple | Qwen2VLCausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    
    output_attentions = kwargs.get("output_attentions", False)
    # print(f"here output_attentions: {output_attentions}")
    if output_attentions:
        attn_weigths = outputs[1]
    # print(f"here attention weights: {attn_weigths if output_attentions else 'None'}")

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen2VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
        attentions=attn_weigths if output_attentions else None,
    )


def forward_with_triton_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> tuple | Qwen2VLCausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen2VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
