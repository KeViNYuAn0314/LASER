# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.actor.attention_capture import AttentionSliceCapturer, compute_slice_for_sample
from verl.workers.config import ActorConfig
from verl.utils.torch_dtypes import PrecisionType

import math
import re

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


from contextlib import contextmanager


@contextmanager
def _disabled_ulysses_sp():
    """Temporarily nulls the ulysses sequence-parallel process group so a
    standalone diagnostic forward (the attention-reward extractor) sees the
    full unsharded sequence. Restored on exit even if the body raises.

    This matters because ``patch_vlm_for_ulysses_input_slicing`` slices
    ``inputs_embeds`` and ``position_ids`` at model entry whenever
    ``get_ulysses_sequence_parallel_world_size() > 1``, and
    ``_ulysses_flash_attention_forward`` does an all-to-all on Q/K/V — both
    keyed off the global SP group. Setting the group to None makes both
    runtime checks return 1 → fall back to the unsharded path.
    """
    prev = get_ulysses_sequence_parallel_group()
    set_ulysses_sequence_parallel_group(None)
    try:
        yield
    finally:
        set_ulysses_sequence_parallel_group(prev)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm
    
    
    
    def compute_windowed_global_stability(
        self,
        attention: torch.Tensor,  # (t,)
        window_size: int = 20,
        sensitivity: float = 5.0, 
        scale: float = 0.1,     
        penalty: float = 0.02
    ) -> torch.Tensor:
        t = attention.shape[0]

        if t <= window_size:
            return torch.tensor(0.0, device=attention.device)
        
        weighted_attn = attention
        windows = weighted_attn.unfold(dimension=0, size=window_size, step=window_size // 2)  # (num_windows, window_size)
        window_means = windows.mean(dim=-1) # (num_windows,)

        target_level = window_means.max().detach() 
        denominator = target_level + 1e-12
        ratios = window_means / denominator
        squared_relative_error = (1.0 - ratios)
        
        #    Uses Gaussian kernel
        # per_window_rewards = torch.exp(-sensitivity * squared_relative_error)
        per_window_rewards = torch.exp(-sensitivity * squared_relative_error) - penalty
        # print(f"ratios: {ratios}")
        # print(f"squared_relative_error: {squared_relative_error}")
        # print(f"per_window_rewards: {per_window_rewards}")
        total_reward = per_window_rewards.sum() * scale
        # print(f"total_reward: {total_reward}")
        return total_reward

    def compute_windowed_global_stability_early_weighted(
        self,
        attention: torch.Tensor,  # (t,)
        window_size: int = 20,
        sensitivity: float = 5.0,
        scale: float = 0.1,
        penalty: float = 0.02,
        decay_rate: float = 1.5,
        weight_mode: str = "exp",
    ) -> torch.Tensor:
        # Reflects Finding 1: early-stage visual attention decay is causally
        # more harmful than late-stage decay. Per-window deviation-from-peak
        # rewards are weighted higher at earlier windows. Weights are normalized
        # to sum to num_windows so the magnitude stays comparable to the
        # uniform variant (no need to retune `scale`).
        t = attention.shape[0]

        if t <= window_size:
            return torch.tensor(0.0, device=attention.device)

        weighted_attn = attention
        windows = weighted_attn.unfold(dimension=0, size=window_size, step=window_size // 2)  # (num_windows, window_size)
        window_means = windows.mean(dim=-1)  # (num_windows,)
        num_windows = window_means.shape[0]

        target_level = window_means.max().detach()
        denominator = target_level + 1e-12
        ratios = window_means / denominator
        relative_error = 1.0 - ratios
        per_window_rewards = torch.exp(-sensitivity * relative_error) - penalty

        idx = torch.arange(num_windows, device=attention.device, dtype=per_window_rewards.dtype)
        denom = float(max(num_windows - 1, 1))
        if weight_mode == "exp":
            raw_weights = torch.exp(-decay_rate * (idx / denom))
        elif weight_mode == "linear":
            raw_weights = torch.clamp(1.0 - decay_rate * (idx / denom), min=1e-6)
        elif weight_mode == "power":
            raw_weights = (idx + 1.0).pow(-decay_rate)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode!r}; expected 'exp', 'linear', or 'power'.")

        weights = raw_weights * (num_windows / (raw_weights.sum() + 1e-12))

        total_reward = (per_window_rewards * weights).sum() * scale
        return total_reward


    def compute_sink_suppression_reward(
        self,
        attention: torch.Tensor,      # (num_output_tokens, num_visual_tokens)
        sink_indices: torch.Tensor,   # indices of sink tokens in visual token dimension
        tau: float = 0.9,             # target threshold for sink ratio
        beta: float = 0.5,            # sensitivity parameter
        scale: float = 1.0,           # reward scaling
    ) -> torch.Tensor:
        """
        Token-wise reward function that penalizes excessive attention on visual sink tokens.
        
        Computes per-output-token sink attention ratio, applies penalty for each token,
        then aggregates into a single reward value.
        
        Args:
            attention: Attention weights, shape (num_output_tokens, num_visual_tokens)
            sink_indices: Indices identifying sink tokens in the visual token dimension
            tau: Target maximum ratio for sink attention (default 1.0)
            beta: Sensitivity of penalty to excess sink attention (default 5.0)
            scale: Overall reward scaling factor
            eps: Numerical stability constant
        
        Returns:
            Scalar reward tensor (negative when sink attention is too high)
        """
        num_output_tokens, num_visual_tokens = attention.shape
        
        # Create sink mask over visual tokens
        sink_mask = torch.zeros(num_visual_tokens, dtype=torch.bool, device=attention.device)
        if sink_indices.numel() > 0:
            sink_indices = sink_indices.clamp(0, num_visual_tokens - 1)
            sink_mask[sink_indices] = True
        
        if sink_mask.sum() == 0:
            return torch.tensor(0.0, device=attention.device)
        
        # attention: (T, V), sink_mask: (V,)
        sink_attention = attention[:, sink_mask].mean(dim=-1)      # (T,)
        total_attention = attention.mean(dim=-1)                    # (T,)
        
        sink_ratio = sink_attention / (total_attention + 1e-10)      # (T,)
        # print(f"sink ratio is {sink_ratio}")
        
        # Compute per-token reward: penalize when ratio exceeds threshold
        excess = torch.clamp(sink_ratio - tau, min=0.0)            # (T,)
        per_token_rewards = torch.exp(-beta * excess)       # (T,) in range [0, 1]
        

        reward = per_token_rewards.mean()

        
        return reward * scale
    
    
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def generate_attentions_experimental(
        self,
        data,
        apply_rectification: bool = True,
        apply_sink_suppression: bool = True,
        apply_early_weighted_stability: bool = False,
        stability_reward_penalty: float = 0.015,
        stability_reward_scale: float = 0.1,
        stability_reward_sensitivity: float = 5.0,
        suppression_reward_beta: float = 0.5,
        suppression_reward_scale: float = 1.0,
    ):
        with _disabled_ulysses_sp():
            return self._generate_attentions_experimental_impl(
                data,
                apply_rectification=apply_rectification,
                apply_sink_suppression=apply_sink_suppression,
                apply_early_weighted_stability=apply_early_weighted_stability,
                stability_reward_penalty=stability_reward_penalty,
                stability_reward_scale=stability_reward_scale,
                stability_reward_sensitivity=stability_reward_sensitivity,
                suppression_reward_beta=suppression_reward_beta,
                suppression_reward_scale=suppression_reward_scale,
            )

    def _generate_attentions_experimental_impl(
        self,
        data,
        apply_rectification: bool = True,
        apply_sink_suppression: bool = True,
        apply_early_weighted_stability: bool = False,
        stability_reward_penalty: float = 0.015,
        stability_reward_scale: float = 0.1,
        stability_reward_sensitivity: float = 5.0,
        suppression_reward_beta: float = 0.5,
        suppression_reward_scale: float = 1.0,
    ):
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # batch = data.batch
        
        data_len = data.batch.batch_size[0]
        chunk_size = 8
        
        attention_all = []
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        current_index = 0
        device = data.batch["input_ids"].device
        reward_tensor = torch.zeros(data_len, dtype=torch.float32, device=device)
        suppression_reward_tensor = torch.zeros(data_len, dtype=torch.float32, device=device)
        
        # for batch_idx, micro_batch in enumerate(batch_data):
        for micro_batch in data.split(chunk_size):
            # torch.cuda.empty_cache()
            micro_batch = micro_batch.batch
            multi_modal_inputs = {}
            if "multi_modal_inputs" in micro_batch.keys():
                if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                    for key in micro_batch["multi_modal_inputs"][0].keys():
                        multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
                else:
                    for key in micro_batch["multi_modal_inputs"][0].keys():
                        multi_modal_inputs[key] = torch.cat(
                            [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                        )

            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                    input_ids = micro_batch["input_ids"]    # the input ids include both prompt and response
                    prompt_ids = micro_batch["prompts"]
                    prompt_length = prompt_ids.shape[-1]
                    batch_size, seqlen = input_ids.shape
                    
                    attention_mask = micro_batch["attention_mask"]
                    position_ids = micro_batch["position_ids"]
                    entropy = None
                    if position_ids.dim() == 3:  # qwen2vl mrope
                        position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
                        
                    last_position = 0

                    prompt_length = micro_batch["prompts"].shape[-1]
                    valid_lens = micro_batch["attention_mask"][:, prompt_length:].sum(dim=1)  # (B,)
                    last_position = (prompt_length + valid_lens.max()).item() + 5

                    
                    # last_position = desc_positions.max().item() + 5 # ensure we cover the description part including tags
                    curr_input_ids = input_ids[:, : last_position]
                    curr_attention_mask = attention_mask[:, : last_position]
                    curr_position_ids = position_ids[:, :, : last_position]
                    
                    extra_args = {}
                    extra_args["output_attentions"] = True
                    if self.use_fused_kernels:
                        extra_args["temperature"] = temperature
                        extra_args["return_dict"] = True
                    
                    output = self.actor_module(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attention_mask,
                        position_ids=curr_position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    ) 
                    
                    attentions = output.attentions
                    
                    layer_avg_attentions = torch.stack(attentions, dim=0) # shape (num_layers, batch_size, seq_len, seq_len)
                    
                    del output, attentions
                    
                    for i in range(batch_size):
                        response_ids = micro_batch["responses"][i]
                        valid_response_length = micro_batch["attention_mask"][i][prompt_length:].sum()
                        valid_response_ids = response_ids[:valid_response_length]
                        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                        
                        if not self._format_reward_think(response_str):
                            reward_tensor[current_index] = 0.0
                            current_index += 1
                            continue
                        
                        desc_start = prompt_length + 1
                        desc_end = desc_start + len(valid_response_ids) - 1
                        if_find = desc_start != -1 and desc_end != -1 and desc_end > desc_start
                        assert if_find, f"Cannot find think tags in the response! desc_start: {desc_start}, desc_end: {desc_end}, last_position: {last_position}"
                        
                        visual_token_indices = self._find_visual_token_positions(input_ids[i], image_token_id)
                        # visual_token_indices = torch.tensor(visual_token_indices, dtype=torch.long, device=input_ids.device)
                        
                        torch.cuda.empty_cache()
                        
                        desc_start = prompt_length + 1
                        
                        # compute average attention weights from instruction tokens to visual tokens
                        # instruction_start starts from end of visual tokens to prompt_length
                        # instruction_start = visual_sink_token_indices.max().item() + 2
                        # attn_weights = attentions[:, i, :, instruction_start:prompt_length, visual_token_indices].mean(dim=0).mean(dim=0).mean(dim=0)  # shape (num_visual_tokens)
                        
                        
                        # # TODO:  identify the visual sink token indices
                        bos_pos = prompt_length - 1
                        # decode the bos_token to check

                        if apply_rectification:
                            attn_weights = layer_avg_attentions[:, i, bos_pos, visual_token_indices].mean(dim=0)  # shape (num_visual_tokens)
                            sink_mask = attn_weights > attn_weights.mean() + 2 * attn_weights.std()
                            sink_indices = torch.where(sink_mask)[0]
                            non_sink_mask = ~sink_mask # shape (num_visual_tokens)
                            
                            # TODO: compute stability reward
                            avg_attentions = layer_avg_attentions[:, i, desc_start:desc_end, visual_token_indices].mean(dim=0)  # shape (desc_len, num_visual_tokens)
                            # multiply rectified visual attention weights
                            # avg_attentions = avg_attentions * visual_attentions_rectified.unsqueeze(0)  # shape (desc_len, num_visual_tokens)
                            
                            # only check the attention weights on non-sink tokens
                            avg_attentions = avg_attentions[:, non_sink_mask]  # shape (desc_len, num_non_sink_visual_tokens)
                        else:
                            avg_attentions = layer_avg_attentions[:, i, desc_start:desc_end, visual_token_indices].mean(dim=0)  # shape (desc_len, num_visual_tokens)
                        # avg_attentions = avg_attentions
                        
                        avg_attentions = avg_attentions.mean(dim=1)  # shape (desc_len,)
                        window_size = 10
                        if apply_early_weighted_stability:
                            stability_reward = self.compute_windowed_global_stability_early_weighted(
                                avg_attentions,
                                window_size=window_size,
                                sensitivity=stability_reward_sensitivity,
                                scale=stability_reward_scale,
                                penalty=stability_reward_penalty,
                                decay_rate=1.5,        # tune: larger = more early-stage emphasis
                                weight_mode="exp",
                            )
                        else:
                            stability_reward = self.compute_windowed_global_stability(
                                avg_attentions,
                                window_size=window_size,
                                sensitivity=stability_reward_sensitivity,
                                scale=stability_reward_scale,
                                penalty=stability_reward_penalty,
                            )

                        if apply_rectification and apply_sink_suppression:
                            # compute sink suppression reward
                            sink_mask = attn_weights > attn_weights.mean() + 2 * attn_weights.std()
                            sink_indices = torch.where(sink_mask)[0]
                            sink_suppression_reward = self.compute_sink_suppression_reward(
                                attention=layer_avg_attentions[:, i, desc_start:desc_end, visual_token_indices].mean(dim=0),  # shape (desc_len, num_visual_tokens)
                                sink_indices=sink_indices,
                                beta=suppression_reward_beta,
                                scale=suppression_reward_scale,
                            )

                            # stability_reward += sink_suppression_reward
                            suppression_reward_tensor[current_index] = sink_suppression_reward
                        
                        
                        reward_tensor[current_index] = stability_reward
                        current_index += 1
        torch.cuda.empty_cache()
        return reward_tensor, suppression_reward_tensor


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def generate_attentions_experimental_hooked(
        self,
        data,
        apply_rectification: bool = True,
        apply_sink_suppression: bool = True,
        apply_early_weighted_stability: bool = False,
        stability_reward_penalty: float = 0.015,
        stability_reward_scale: float = 0.1,
        stability_reward_sensitivity: float = 5.0,
        suppression_reward_beta: float = 0.5,
        suppression_reward_scale: float = 1.0,
    ):
        with _disabled_ulysses_sp():
            return self._generate_attentions_experimental_hooked_impl(
                data,
                apply_rectification=apply_rectification,
                apply_sink_suppression=apply_sink_suppression,
                apply_early_weighted_stability=apply_early_weighted_stability,
                stability_reward_penalty=stability_reward_penalty,
                stability_reward_scale=stability_reward_scale,
                stability_reward_sensitivity=stability_reward_sensitivity,
                suppression_reward_beta=suppression_reward_beta,
                suppression_reward_scale=suppression_reward_scale,
            )

    def _generate_attentions_experimental_hooked_impl(
        self,
        data,
        apply_rectification: bool = True,
        apply_sink_suppression: bool = True,
        apply_early_weighted_stability: bool = False,
        stability_reward_penalty: float = 0.015,
        stability_reward_scale: float = 0.1,
        stability_reward_sensitivity: float = 5.0,
        suppression_reward_beta: float = 0.5,
        suppression_reward_scale: float = 1.0,
    ):
        """Drop-in replacement for ``generate_attentions_experimental`` that
        captures post-RoPE Q and K via instance-level monkey patches on the
        text decoder's attention modules (``AttentionSliceCapturer``) and
        computes only the small ``(T_q, V)`` slice the reward needs. This
        avoids materialising the full ``(B, H, T, T)`` attention tensor that
        ``output_attentions=True`` forces, and keeps FlashAttention/SDPA active
        for the actual forward pass.

        Iteration mirrors :meth:`_generate_attentions_experimental_impl`
        exactly: every DP rank runs the *same* number of forward passes
        (one per micro-batch of ``chunk_size``). The format-think check is
        applied per-sample *after* the forward — failing samples just receive
        a zero reward. Pre-filtering would diverge the forward count across
        DP ranks and hang FSDP's collectives.
        """
        self.actor_module.eval()
        temperature = data.meta_info["temperature"]

        data_len = data.batch.batch_size[0]
        device = data.batch["input_ids"].device
        reward_tensor = torch.zeros(data_len, dtype=torch.float32, device=device)
        suppression_reward_tensor = torch.zeros(data_len, dtype=torch.float32, device=device)

        prompt_length = data.batch["prompts"].shape[-1]
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        chunk_size = 4
        # Pointer into the original (un-filtered) data position. Mirrors
        # ``current_index`` in :meth:`_generate_attentions_experimental_impl`.
        current_index = 0

        for micro_batch in data.split(chunk_size):
            micro_batch = micro_batch.batch
            multi_modal_inputs: dict = {}
            if "multi_modal_inputs" in micro_batch.keys():
                if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o branch
                    for key in micro_batch["multi_modal_inputs"][0].keys():
                        multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
                else:
                    for key in micro_batch["multi_modal_inputs"][0].keys():
                        multi_modal_inputs[key] = torch.cat(
                            [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                        )

            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                    input_ids = micro_batch["input_ids"]
                    batch_size, _ = input_ids.shape
                    attention_mask = micro_batch["attention_mask"]
                    position_ids = micro_batch["position_ids"]
                    if position_ids.dim() == 3:  # qwen2vl mrope: (B, 3, T) -> (3, B, T)
                        position_ids = position_ids.transpose(0, 1)

                    valid_lens = attention_mask[:, prompt_length:].sum(dim=1)  # (B,)
                    last_position = int((prompt_length + valid_lens.max()).item()) + 5
                    last_position = min(last_position, attention_mask.shape[1])

                    curr_input_ids = input_ids[:, :last_position]
                    curr_attention_mask = attention_mask[:, :last_position]
                    curr_position_ids = position_ids[:, :, :last_position]

                    extra_args = {}
                    if self.use_fused_kernels:
                        extra_args["temperature"] = temperature
                        extra_args["return_dict"] = True

                    with AttentionSliceCapturer(self.actor_module) as capturer:
                        _ = self.actor_module(
                            input_ids=curr_input_ids,
                            attention_mask=curr_attention_mask,
                            position_ids=curr_position_ids,
                            **multi_modal_inputs,
                            use_cache=False,
                            **extra_args,
                        )
                        captures = capturer.get_captures()

                    if not captures:
                        # No layers captured (shouldn't happen for Qwen2.5-VL); skip the
                        # whole micro-batch but advance the pointer correctly.
                        current_index += batch_size
                        del captures
                        torch.cuda.empty_cache()
                        continue

                    for i in range(batch_size):
                        # Format-think check identical to
                        # :meth:`_generate_attentions_experimental_impl`. We run the
                        # check *after* the forward so every rank performs the same
                        # number of forwards (and the same FSDP collectives).
                        valid_response_length_t = curr_attention_mask[i, prompt_length:last_position].sum()
                        valid_response_length = int(valid_response_length_t.item())
                        if valid_response_length <= 1:
                            current_index += 1
                            continue

                        valid_response_ids = micro_batch["responses"][i][:valid_response_length]
                        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                        if not self._format_reward_think(response_str):
                            current_index += 1
                            continue

                        desc_start = prompt_length + 1
                        desc_end = desc_start + valid_response_length - 1
                        if desc_end <= desc_start:
                            current_index += 1
                            continue

                        visual_token_indices = self._find_visual_token_positions(
                            curr_input_ids[i], image_token_id
                        )
                        if visual_token_indices.numel() == 0:
                            current_index += 1
                            continue

                        # Per-sample valid key mask (handles left + right padding rigorously).
                        sample_valid_mask = curr_attention_mask[i].to(torch.bool)

                        # --- Sink identification: bos -> visual ---
                        if apply_rectification:
                            bos_query = torch.tensor(
                                [prompt_length - 1], dtype=torch.long, device=device
                            )
                            bos_slice = compute_slice_for_sample(
                                captures=captures,
                                sample_idx=i,
                                query_indices=bos_query,
                                visual_indices=visual_token_indices,
                                key_valid_mask=sample_valid_mask,
                            )  # (1, V) fp32
                            attn_weights = bos_slice.squeeze(0)  # (V,)
                            sink_mask_visual = attn_weights > attn_weights.mean() + 2 * attn_weights.std()
                            sink_indices = torch.where(sink_mask_visual)[0]
                            non_sink_mask_visual = ~sink_mask_visual
                        else:
                            sink_indices = None
                            non_sink_mask_visual = None

                        # --- Per-step attention to visual: response token range -> visual ---
                        desc_query = torch.arange(
                            desc_start, desc_end, dtype=torch.long, device=device
                        )
                        desc_slice = compute_slice_for_sample(
                            captures=captures,
                            sample_idx=i,
                            query_indices=desc_query,
                            visual_indices=visual_token_indices,
                            key_valid_mask=sample_valid_mask,
                        )  # (T_q, V) fp32

                        if apply_rectification:
                            avg_attentions = desc_slice[:, non_sink_mask_visual]
                        else:
                            avg_attentions = desc_slice

                        avg_attentions_per_step = avg_attentions.mean(dim=1)  # (T_q,)
                        window_size = 10
                        if apply_early_weighted_stability:
                            stability_reward = self.compute_windowed_global_stability_early_weighted(
                                avg_attentions_per_step,
                                window_size=window_size,
                                sensitivity=stability_reward_sensitivity,
                                scale=stability_reward_scale,
                                penalty=stability_reward_penalty,
                                decay_rate=1.5,
                                weight_mode="exp",
                            )
                        else:
                            stability_reward = self.compute_windowed_global_stability(
                                avg_attentions_per_step,
                                window_size=window_size,
                                sensitivity=stability_reward_sensitivity,
                                scale=stability_reward_scale,
                                penalty=stability_reward_penalty,
                            )

                        reward_tensor[current_index] = stability_reward

                        if apply_rectification and apply_sink_suppression and sink_indices is not None:
                            sink_suppression_reward = self.compute_sink_suppression_reward(
                                attention=desc_slice,  # (T_q, V) full visual set
                                sink_indices=sink_indices,
                                beta=suppression_reward_beta,
                                scale=suppression_reward_scale,
                            )
                            suppression_reward_tensor[current_index] = sink_suppression_reward

                        current_index += 1

                    # Free capture tensors before the next chunk.
                    del captures
            torch.cuda.empty_cache()

        return reward_tensor, suppression_reward_tensor

    def _find_visual_token_positions(self, input_ids, image_pad_token_id):
        """
        Find positions of visual tokens (image tokens) in the input sequence.
        Returns tensor of positions.
        """
        return (input_ids == image_pad_token_id).nonzero(as_tuple=True)[0]

    def _format_reward_think(self, completion_content: str) -> float:
        """
        Reward function that checks if the response follows the correct format:
        - Image description is enclosed within <description> and </description> tags
        - Reasoning process is enclosed within <think> and </think> tags
        - Final answer is enclosed within <answer> and </answer> tags
        
        Args:
            completion_content: The model's completion string
            
        Returns:
            1.0 if format is correct, 0.0 otherwise
        """
        # pattern_description = r"^(\s*)<description>.*?</description>"
        pattern_think = r"<think>.*?</think>"
        pattern_answer = r"<answer>.*?</answer>(\s*)$"
        
        # match_description = re.findall(pattern_description, completion_content, re.MULTILINE | re.DOTALL)
        match_think = re.findall(pattern_think, completion_content, re.MULTILINE | re.DOTALL)
        match_answer = re.findall(pattern_answer, completion_content, re.MULTILINE | re.DOTALL)
        
        if (match_think is not None and match_answer is not None and 
            len(match_think) == 1 and len(match_answer) == 1):
            return 1.0
        else:
            return 0.0   

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    # loss.backward()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    # torch.cuda.empty_cache()
                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
