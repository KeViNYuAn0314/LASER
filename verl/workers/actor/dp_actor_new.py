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
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
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

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
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
    
    def chunk_iterate(batch_data, chunk_size):
        for i in range(0, len(batch_data), chunk_size):
            yield batch_data[i:i + chunk_size]

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def generate_attentions(self, data):
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        # micro_batch_size = 2
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # batch = data.batch
        
        data_len = data.batch.batch_size[0]
        chunk_size = 1
        # num_chunks = max(data_len, 1)
        # num_chunks = math.ceil(data_len / chunk_size)
        # batch_data = data.chunk(chunks=num_chunks)
        
        # print(f"computing attentions")
        
        attention_all = []
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        current_index = 0
        reward_tensor = torch.zeros(data_len, dtype=torch.float32)
        # for batch_idx, micro_batch in enumerate(batch_data):
        for micro_batch in data.split(chunk_size):
        # for batch_idx, micro_batch in enumerate(self.chunk_iterate(data, chunk_size)):
            # clear memory
            torch.cuda.empty_cache()
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
                    
            import time
            start_time = time.time()
            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                    input_ids = micro_batch["input_ids"]    # the input ids include both prompt and response
                    prompt_ids = micro_batch["prompts"]
                    prompt_length = prompt_ids.shape[-1]
                    batch_size, seqlen = input_ids.shape
                    
                    attention_mask = micro_batch["attention_mask"]
                    position_ids = micro_batch["position_ids"]
                    entropy = None
                    if position_ids.dim() == 3:  # qwen2vl mrope
                        position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
                        # only pass input_ids and position_ids to enable flash_attn_varlen
                    
                    # desc_token = self.tokenizer.encode("<description>", add_special_tokens=False)[1]
                    # desc_mask = desc_token == input_ids
                    
                    think_start_token_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
                    think_end_token_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
                    # think_end_token_id = torch.tensor([26865], device=input_ids.device)  # shape (1,)
                    
                    # find the max desc position in the batch
                    last_position = 0
                    for i in range(batch_size):
                        # last_position = max(last_position, self.find_subtensor(input_ids[i], torch.tensor(think_end_token_ids, device=input_ids.device)[1].unsqueeze(0)))
                        response_ids = micro_batch["responses"][i]
                        valid_response_length = micro_batch["attention_mask"][i][prompt_length:].sum()
                        prompt_length = micro_batch["prompts"].shape[-1]
                        last_position = max(last_position, prompt_length + valid_response_length)    # find the last response token position
                    last_position += 5  # ensure we cover the description part including tags

                    # desc_positions = torch.where(desc_mask)[1]
                    # if desc_positions.numel() == 0:
                    #     # print(f"Batch {batch_idx} has no description tokens, setting reward to 0.")
                    #     for i in range(batch_size):
                    #         reward_tensor[current_index] = 0.0
                    #         current_index += 1
                    #     continue
                    
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
                    )  # prevent model thinks we are generating
                    attentions = output.attentions # (batch_size, seq_len, seq_len)
                    del output
                    
                    for i in range(batch_size):
                        response_ids = micro_batch["responses"][i]
                        valid_response_length = micro_batch["attention_mask"][i][prompt_length:].sum()
                        valid_response_ids = response_ids[:valid_response_length]
                        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                        # if not self._format_reward_description(response_str):
                        #     reward_tensor[current_index] = 0.0
                        #     current_index += 1
                        #     continue
                        
                        if not self._format_reward_think(response_str):
                            reward_tensor[current_index] = 0.0
                            current_index += 1
                            continue
                        
                        # desc_start = self.find_subtensor(input_ids[i], torch.tensor(think_start_token_ids, device=input_ids.device))
                        desc_start = prompt_length + 1
                        desc_end = desc_start + len(valid_response_ids) - 2
                        # desc_end = self.find_subtensor(input_ids[i], torch.tensor(think_end_token_ids, device=input_ids.device)[1].unsqueeze(0))
                        # print(f"desc_start: {desc_start}, desc_end: {desc_end}, last_position: {last_position}")
                        if_find = desc_start != -1 and desc_end != -1 and desc_end > desc_start
                        assert if_find, f"Cannot find think tags in the response! desc_start: {desc_start}, desc_end: {desc_end}, last_position: {last_position}"
                        
                        visual_token_indices = self._find_visual_token_positions(input_ids[i], image_token_id)
                        visual_token_indices = torch.tensor(visual_token_indices, dtype=torch.long, device=input_ids.device)
                        
                        torch.cuda.empty_cache()
                        # print(f"")
                        # desc_start, desc_end, if_find = self.find_desc_token_section(input_ids[i], torch.where(desc_mask[i])[0], desc_token)
                        # if not if_find:
                        #     reward_tensor[current_index] = 0.0
                        #     current_index += 1
                        #     continue
                        desc_start = prompt_length + 1
                        description_indices = torch.arange(desc_start, desc_end)
                        # desc_to_visual_attentions = [attentions_avg[i, description_index, visual_token_indices].sum().item() / (attentions_avg[i, description_index, prompt_token_indices].sum().item() + 1e-8) for description_index in description_indices]
                        
                        # print(f"description indices length: {len(description_indices)}, valid response length: {valid_response_length}")
                        if len(description_indices) < 0.25 * valid_response_length or len(description_indices) <= 30:
                            reward_tensor[current_index] = 0.0
                            current_index += 1
                            continue
                        
                        desc_visual_attn = attentions[i, description_indices][:, visual_token_indices].sum(dim=-1)
                        # desc_total_attn = attentions_avg[i, description_indices][:, prompt_token_indices].sum(dim=-1)
                        valid_prompt_length = micro_batch[i]["attention_mask"][:prompt_length].sum()
                        prompt_start = prompt_length - valid_prompt_length
                        prompt_end = prompt_start + valid_prompt_length
                        prompt_token_indices = torch.arange(prompt_start, prompt_end)
                        desc_total_attn = torch.stack([
                            attentions[0, description_indices[j], torch.cat((prompt_token_indices, description_indices[:j]))].sum(dim=-1)
                            for j in range(len(description_indices))
                        ])
                        # print(f"desc_visual_attn first 5: {desc_visual_attn[:5]}")
                        # print(f"desc_visual_attn last 5: {desc_visual_attn[-5:]}")
                        # print(f"desc_total_attn first 5: {desc_total_attn[:5]}")
                        # print(f"desc_total_attn last 5: {desc_total_attn[-5:]}")
                        # print(f"desc_visual_attn shape is {desc_visual_attn.shape}, desc_total_attn shape is {desc_total_attn.shape}")

                        # Compute attention ratios (keep as tensor)
                        desc_to_visual_attentions = desc_visual_attn / (desc_total_attn + 1e-8)
                        
                        # Compute exponential moving average of the attention ratios
                        # def update_ema_inplace(ema, new_value, alpha=0.7):
                        #     with torch.no_grad():
                        #         ema.mul_(alpha).add_(new_value, alpha=1 - alpha)
                        #     return ema
                        
                        # desc_to_visual_attentions_ema = torch.zeros_like(desc_to_visual_attentions, device=desc_to_visual_attentions.device)
                        # update_ema_inplace(desc_to_visual_attentions_ema, desc_to_visual_attentions, alpha=0.7)
                        
                        # top_k = min(10, len(desc_to_visual_attentions_ema))
                        # max_retention = desc_to_visual_attentions_ema[:top_k].max()

                        # Create exponential decay curve
                        # sequence_length = len(desc_to_visual_attentions_ema)
                        # decay_rate = torch.log(torch.tensor(4.0, device=desc_to_visual_attentions_ema.device)) / sequence_length
                        # positions = torch.arange(sequence_length, device=desc_to_visual_attentions_ema.device).float()
                        # exponential_curve = max_retention * torch.exp(-decay_rate * positions)

                        # Calculate reward per token
                        # reward_per_token = (desc_to_visual_attentions_ema >= exponential_curve).float()
                        
                        window_average = []
                        window_positions = []
                        window_size = 20
                        stride = 10
                        
                        for start in range(0, len(desc_to_visual_attentions) - window_size + 1, stride):
                            end = start + window_size
                            window_avg = desc_to_visual_attentions[start:end].mean()
                            window_average.append(window_avg)
                            window_positions.append((start, end))
                            
                        max_retention = max(window_average[:5]) * 0.8
                        decay_rate = torch.log(torch.tensor(5.0, device=desc_to_visual_attentions.device)) / len(window_average)
                        positions = torch.arange(len(window_average), device=desc_to_visual_attentions.device).float()
                        exponential_curve = max_retention * torch.exp(-decay_rate * positions)
                        reward_per_token = (torch.tensor(window_average, device=desc_to_visual_attentions.device) >= exponential_curve).float()
                        
                        avg_reward = reward_per_token.mean().item()
                        
                        # max_retention = desc_to_visual_attentions_ema[:10].max()
                        # # min_retention = 1/2 * max_retention
                        # decay_rate = -torch.log(torch.tensor([1/2], device=desc_to_visual_attentions_ema.device)) / len(desc_to_visual_attentions_ema)
                        # exponential_curve = max_retention * torch.exp(-decay_rate * torch.arange(1, len(desc_to_visual_attentions_ema) + 1))
                        # reward_per_token = (desc_to_visual_attentions_ema >= exponential_curve).float()
                        # avg_reward = float(reward_per_token.mean().item())
                        
                        # print(f"variance of desc_to_visual_attentions: {desc_to_visual_attentions.var().item()}")
                        # print(f"desc_to_visual_attentions first 5: {desc_to_visual_attentions_ema[:5]}")
                        # print(f"desc_to_visual_attentions last 5: {desc_to_visual_attentions_ema[-5:]}")
                        
                        # print(f"exponential_curve first 5: {exponential_curve[:5]}")
                        # print(f"exponential_curve last 5: {exponential_curve[-5:]}")
                      
                        # target_max = desc_to_visual_attentions_ema.max() # (batch_size)
                        # target_min = target_max * 0.2   # (batch_size)

                        # reward_per_token = ((desc_to_visual_attentions_ema >= target_min) & (desc_to_visual_attentions_ema <= target_max)).float()
                        # avg_reward = float(reward_per_token.mean().item())
                        # print(f"sample avg reward: {avg_reward}")
                        
                        # reward should be penalized if response length is too short, while should capped if too long
                        reward_tensor[current_index] = avg_reward * min(1.0, valid_response_length.float().item() / 150.0)
                        current_index += 1
                        
                        # desc_attn_ratios = attentions[i, desc_start:desc_end]
                        
                        # target_max = desc_attn_ratios.max() # (batch_size)
                        # target_min = target_max * 0.2   # (batch_size)

                        # reward_per_token = ((desc_attn_ratios >= target_min) & (desc_attn_ratios <= target_max)).float()
                        # avg_reward = float(reward_per_token.mean().item())
                        # reward_tensor[current_index] = avg_reward
                        # current_index += 1
                        
                        

                    # auto regressive to get attentions
                    # print(f"input_ids shape: {input_ids.shape}, prompt_length: {prompt_length}, seqlen: {seqlen}")
                    # print(f"attentions mask shape: {attention_mask.shape}")
                    # print(f"position_ids shape: {position_ids.shape}")
                    # import time
                    # start_time = time.time()
                    # print(f"response length is {seqlen - prompt_length}")
                    # for response_idx in range(seqlen - prompt_length):
                    #     curr_input_ids = input_ids[:, : prompt_length + response_idx + 1]
                    #     curr_attention_mask = attention_mask[:, : prompt_length + response_idx + 1]
                    #     curr_position_ids = position_ids[:, :, : prompt_length + response_idx + 1]
                        
                    #     # print(f"current input ids shape: {curr_input_ids.shape}")
                    #     # print(f"current attention mask shape: {curr_attention_mask.shape}")
                    #     # print(f"current position ids shape: {curr_position_ids.shape}")
                    #     extra_args = {}
                    #     if self.use_fused_kernels:
                    #         extra_args["temperature"] = temperature
                    #         extra_args["return_dict"] = True
                        
                    #     output = self.actor_module(
                    #         input_ids=curr_input_ids,
                    #         attention_mask=curr_attention_mask,
                    #         position_ids=curr_position_ids,
                    #         **multi_modal_inputs,
                    #         use_cache=False,
                    #         **extra_args,
                    #     )  # prevent model thinks we are generating
                    #     attentions = output.attentions
                        # print(f"Batch {batch_idx}, Response idx {response_idx}, Attentions shape: {attentions.shape}")
                        
                    #     for i in range(batch_size):
                    #         visual_token_indices = self._find_visual_token_positions(input_ids[i], image_token_id)
                    #         visual_token_indices = torch.tensor(visual_token_indices, dtype=torch.long, device=input_ids.device)
                    #         attn_ratio = attentions[i, visual_token_indices].sum() / (attentions[i].sum() + 1e-8)
                    #         print(f"attn_ratio: {attn_ratio}")
                    #         # desc_start_tokens = self.tokenizer.encode("<description>", add_special_tokens=False)
                    #         # desc_end_tokens = self.tokenizer.encode("</description>", add_special_tokens=False)
                    #         # desc_start, desc_end = self.find_token_sequence_last_two(input_ids, desc_end_tokens[1])
                    #         attn_ratio_batch[i].append(attn_ratio)
                        
                    #     del attentions, output
                    #     torch.cuda.empty_cache()

                    # attn_ratio_batch = [torch.cat(attn_ratio_batch[i], dim=0).to(input_ids.device) for i in range(batch_size)]
                    # for i in range(batch_size):
                    #     response_ids = micro_batch["responses"][i]
                    #     desc_start, desc_end = self.find_token_sequence_last_two(response_ids, desc_token)
                    #     desc_attn_ratios = attn_ratio_batch[i][desc_start:desc_end]
                        
                    #     target_max = desc_attn_ratios.max() # (batch_size)
                    #     target_min = target_max * 0.2   # (batch_size)

                    #     reward_per_token = ((desc_attn_ratios >= target_min) & (desc_attn_ratios <= target_max)).float()
                    #     avg_reward = float(reward_per_token.mean().item())
                    #     reward_tensor[current_index] = avg_reward
                    #     current_index += 1

                    end_time = time.time()
                    # print(f"Time taken for processing micro batch {batch_idx}: {end_time - start_time} seconds")
                    # print(f"rewards in the batch are: {reward_tensor[current_index - micro_batch_size:current_index]}")
        return reward_tensor   
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def generate_attentions_reflection_v(self, data):
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        # micro_batch_size = 2
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # batch = data.batch
        
        data_len = data.batch.batch_size[0]
        chunk_size = 1
        
        attention_all = []
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        current_index = 0
        reward_tensor = torch.zeros(data_len, dtype=torch.float32)
        # for batch_idx, micro_batch in enumerate(batch_data):
        for micro_batch in data.split(chunk_size):
            torch.cuda.empty_cache()
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
                    
            import time
            start_time = time.time()
            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
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
                    for i in range(batch_size):
                        # last_position = max(last_position, self.find_subtensor(input_ids[i], torch.tensor(think_end_token_ids, device=input_ids.device)[1].unsqueeze(0)))
                        response_ids = micro_batch["responses"][i]
                        valid_response_length = micro_batch["attention_mask"][i][prompt_length:].sum()
                        prompt_length = micro_batch["prompts"].shape[-1]
                        last_position = max(last_position, prompt_length + valid_response_length)    # find the last response token position
                    last_position += 5  # ensure we cover the description part including tags

                    
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
                    
                    attentions = torch.stack(attentions, dim=0).mean(dim=0)  # average over layers
                    
                    
                    del output
                    
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
                        visual_token_indices = torch.tensor(visual_token_indices, dtype=torch.long, device=input_ids.device)
                        
                        torch.cuda.empty_cache()
                        
                        desc_start = prompt_length + 1
                        first_half_indices = torch.arange(desc_start, (desc_end - desc_start) // 2 + desc_start)
                        second_half_indices = torch.arange((desc_end - desc_start) // 2 + desc_start, desc_end)
                        

                        first_half_visual_attn = attentions[i, first_half_indices][:, visual_token_indices].mean(dim=-1).sum()
                        second_half_visual_attn = attentions[i, second_half_indices][:, visual_token_indices].mean(dim=-1).sum()
                        
                        attn_reward = second_half_visual_attn.item() / (first_half_visual_attn.item() + 1e-8)
                        
                        reward_tensor[current_index] = attn_reward
                        current_index += 1
        return reward_tensor   
    
    
    def visual_attention_rectification(
        self,
        attn: torch.Tensor,           # (h, w) or (h*w,) visual attention map
        sink_indices: torch.Tensor,   # 1D tensor of sink token indices (flattened)
        gamma_max: float = 0.1,       # maximum retention ratio for sinks
        normalize: bool = False,      # if True, output sums to 1 (importance scores)
        temperature: float = 1.0,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Rectify visual token attention by suppressing sinks and 
        redistributing to non-sink tokens proportionally.
        
        Args:
            attn: Visual attention map, shape (h, w) or (h*w,)
            sink_indices: Indices of sink tokens (in flattened space)
            gamma_max: Maximum retention ratio for sink tokens (default 0.2)
            normalize: If True, normalize output to sum to 1
            eps: Small constant for numerical stability
        
        Returns:
            Rectified attention map (same shape as input)
        """
        original_shape = attn.shape
        attn = attn.flatten().clone()
        n = attn.numel()
        
        # Create sink mask
        sink_mask = torch.zeros(n, dtype=torch.bool, device=attn.device)
        sink_mask[sink_indices] = True
        
        # Partition statistics
        M_S = attn[sink_mask].sum()              # sink mass
        M_S_bar = attn[~sink_mask].sum()         # non-sink mass
        M = M_S + M_S_bar                         # total visual attention mass
        
        n_sink = sink_mask.sum().float()
        n_nonsink = (~sink_mask).sum().float()
        
        # Adaptive suppression factor
        gamma = torch.clamp(
            (n_sink * M_S_bar) / (n_nonsink * M_S + eps),
            max=gamma_max
        )
        
        # Scaling factor for non-sink tokens (redistributes released mass)
        lambda_scale = (M - gamma * M_S) / (M_S_bar + eps)
        
        # Apply rectification
        attn[sink_mask] *= gamma
        attn[~sink_mask] *= lambda_scale
        
        attn = torch.clamp(attn, min=eps)
        attn = torch.pow(attn, 1.0 / temperature)

        if normalize:
            attn = attn / (attn.sum() + eps)
        
        return attn.view(original_shape)
    
    def compute_windowed_global_LSE(
        self,
        attention: torch.Tensor,  # (t,)
        window_size: int = 20,
        sensitivity: float = 3.0, 
        scale: float = 0.2,                                 
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
        
        # perform log-mean-exp on the ratios
        # lse = (torch.logsumexp(-sensitivity * ratios, dim=0) - torch.log(torch.tensor(float(windows.shape[0]), device=attention.device))) * (-1/sensitivity)
        lse = (-1/sensitivity) * torch.log(torch.mean(torch.exp(-sensitivity * ratios), dim=0))

        total_reward = lse
        return total_reward
    
    def compute_windowed_global_stability(
        self,
        attention: torch.Tensor,  # (t,)
        window_size: int = 20,
        sensitivity: float = 5.0, 
        scale: float = 0.1,     
        penalty: float = 0.01
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
    def generate_attentions_experimental(self, data, apply_rectification: bool = True, apply_sink_suppression: bool = True):
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # batch = data.batch
        
        data_len = data.batch.batch_size[0]
        chunk_size = 4
        
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
                    
            import time
            start_time = time.time()
            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
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
                    # for i in range(batch_size):
                    #     # last_position = max(last_position, self.find_subtensor(input_ids[i], torch.tensor(think_end_token_ids, device=input_ids.device)[1].unsqueeze(0)))
                    #     response_ids = micro_batch["responses"][i]
                    #     valid_response_length = micro_batch["attention_mask"][i][prompt_length:].sum()
                    #     prompt_length = micro_batch["prompts"].shape[-1]
                    #     last_position = max(last_position, prompt_length + valid_response_length)    # find the last response token position
                    # last_position += 5  # ensure we cover the description part including tags
                    
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
                        
                        # print(f"apply rectification: {apply_rectification}, apply sink suppression: {apply_sink_suppression}")
                        # Identify sink visual tokens via BOS-row attention (used in both branches)
                        attn_weights = layer_avg_attentions[:, i, bos_pos, visual_token_indices].mean(dim=0)  # shape (num_visual_tokens)
                        sink_mask = attn_weights > attn_weights.mean() + 2 * attn_weights.std()
                        sink_indices = torch.where(sink_mask)[0]

                        if apply_rectification:
                            non_sink_mask = ~sink_mask  # shape (num_visual_tokens)

                            avg_attentions = layer_avg_attentions[:, i, desc_start:desc_end, visual_token_indices].mean(dim=0)  # shape (desc_len, num_visual_tokens)

                            # Only consider the attention weights on sink visual tokens for stability
                            avg_attentions = avg_attentions[:, sink_mask]  # shape (desc_len, num_sink_visual_tokens)
                        else:
                            avg_attentions = layer_avg_attentions[:, i, desc_start:desc_end, visual_token_indices].mean(dim=0)  # shape (desc_len, num_visual_tokens)
                            # Only consider the attention weights on sink visual tokens for stability
                            avg_attentions = avg_attentions[:, sink_mask]  # shape (desc_len, num_sink_visual_tokens)
                        # avg_attentions = avg_attentions
                        
                        avg_attentions = avg_attentions.mean(dim=1)  # shape (desc_len,)
                        window_size = 10
                        stability_reward = self.compute_windowed_global_stability(
                            avg_attentions,
                            window_size=window_size,
                            scale=0.1,
                        )
                        
                        if apply_rectification and apply_sink_suppression:
                            # compute sink suppression reward (sink_mask/sink_indices already computed above)
                            sink_suppression_reward = self.compute_sink_suppression_reward(
                                attention=layer_avg_attentions[:, i, desc_start:desc_end, visual_token_indices].mean(dim=0),  # shape (desc_len, num_visual_tokens)
                                sink_indices=sink_indices,
                            )

                            # stability_reward += sink_suppression_reward
                            suppression_reward_tensor[current_index] = sink_suppression_reward
                        
                        
                        reward_tensor[current_index] = stability_reward
                        current_index += 1
        # print(f"Rewards: {reward_tensor}")
        torch.cuda.empty_cache()
        return reward_tensor, suppression_reward_tensor
    
    
    def find_subtensor(self, large_tensor, small_tensor):
        """
        Find the starting index of small_tensor in large_tensor.
        Returns -1 if not found.
        """
        if len(small_tensor) > len(large_tensor):
            return -1
        
        # Slide through the large tensor
        max_i = -1
        for i in range(len(large_tensor) - len(small_tensor) + 1):
            # Check if the slice matches
            if torch.equal(large_tensor[i:i + len(small_tensor)], small_tensor):
                max_i = i
        
        return max_i   

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def generate_attentions_old(self, data):
        self.actor_module.eval()
        
        # micro_batch_size = data.meta_info["micro_batch_size"]
        micro_batch_size = 2
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # batch = data.batch
        
        # print(f"total batch size: {data.batch['input_ids'].shape[0]}")
        # print(f"micro batch size: {micro_batch_size}")
        
        batch_size = data.batch.batch_size[0]
        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_data = data.chunk(chunks=num_chunks)
        
        # randomly select subset of batch data
        import random
        # selected_chunks = random.sample(batch_data, k=min(8, num_chunks))
        
        attention_all = []
        reward_tensor = torch.zeros(batch_size, dtype=torch.float32)
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        current_index = 0
        for batch_idx, micro_batch in enumerate(batch_data):
            # print(f"Processing micro batch {batch_idx+1}/{num_chunks}...")
            # print(f" micro batch size: {micro_batch.batch['input_ids'].shape[0]}")
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
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                    input_ids = micro_batch["input_ids"]
                    batch_size, seqlen = input_ids.shape
                    attention_mask = micro_batch["attention_mask"]
                    position_ids = micro_batch["position_ids"]
                    entropy = None
                    if position_ids.dim() == 3:  # qwen2vl mrope
                        position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
                        # only pass input_ids and position_ids to enable flash_attn_varlen
                    extra_args = {}
                    # extra_args["attn_implementation"] = "eager"
                    extra_args["output_attentions"] = True
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
                        
                attentions = output.attentions
                del output
                torch.cuda.empty_cache()
                attentions = torch.stack(attentions)
                # print(f"Batch {batch_idx}, Attentions shape: {attentions.shape}")
                attentions_avg = attentions.mean(dim=(0,2)).detach() # (batch_size, seq_len, seq_len)
                # print(f"Batch {batch_idx}, Attentions avg shape: {attentions_avg.shape}")
                
                for i in range(attentions_avg.shape[0]):
                    data_item = micro_batch[i]
                    input_ids = data_item["input_ids"]
                    prompt_ids = data_item["prompts"]
                    prompt_length = prompt_ids.shape[-1]
                    valid_prompt_length = data_item["attention_mask"][:prompt_length].sum()
                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                    response_ids = data_item["responses"]
                    valid_response_length = data_item["attention_mask"][prompt_length:].sum()
                    valid_response_ids = response_ids[:valid_response_length]
                    # find the valide response indices in terms of input_ids
                    valid_response_start_index = prompt_length
                    valid_response_indices = torch.arange(
                        valid_response_start_index, valid_response_start_index + valid_response_length, device=input_ids.device
                    )
                    # decode
                    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    # print(f"input ids shape: {input_ids.shape}")
                    # print(f"prompt ids shape: {prompt_ids.shape}")
                    # print(f"response ids shape: {response_ids.shape}")
                    # print(f"attentions avg shape per sample: {attentions_avg[i].shape}")
                    # print(f"attention mask shape per sample: {data_item['attention_mask'].shape}")
                    
                    # print(f"Length of description section: {len(description_indices)}")
                    
                    if not self._format_reward_description(response_str):
                        reward_tensor[current_index] = 0.0
                    else: 
                        visual_token_indices = self._find_visual_token_positions(input_ids, image_token_id)
                        visual_token_indices = torch.tensor(visual_token_indices, dtype=torch.long, device=attentions_avg.device)
                        desc_start_tokens = self.tokenizer.encode("<description>", add_special_tokens=False)
                        desc_end_tokens = self.tokenizer.encode("</description>", add_special_tokens=False)
                        desc_start, desc_end = self.find_token_sequence_last_two(input_ids, desc_end_tokens[1])
                        prompt_start = prompt_length - valid_prompt_length
                        prompt_end = prompt_start + valid_prompt_length
                        prompt_token_indices = torch.arange(prompt_start, prompt_end)
                        # print(f"desc_start: {desc_start}, desc_end: {desc_end}")
                        description_indices = torch.arange(desc_start, desc_end)
                        # desc_to_visual_attentions = [attentions_avg[i, description_index, visual_token_indices].sum().item() / (attentions_avg[i, description_index, prompt_token_indices].sum().item() + 1e-8) for description_index in description_indices]
                        
                        # should not give nothing in the description part
                        if len(description_indices) < 0.1 * valid_response_length or len(description_indices) <= 30:
                            reward_tensor[current_index] = 0.0
                            current_index += 1
                            continue
                        
                        desc_visual_attn = attentions_avg[i, description_indices][:, visual_token_indices].sum(dim=-1)
                        # desc_total_attn = attentions_avg[i, description_indices][:, prompt_token_indices].sum(dim=-1)
                        
                        # for j in range(len(description_indices)): 
                        desc_total_attn = torch.stack([
                            attentions_avg[0, description_indices[j], torch.cat((prompt_token_indices, description_indices[:j]))].sum(dim=-1)
                            for j in range(len(description_indices))
                        ])
                        # print(f"desc_visual_attn shape is {desc_visual_attn.shape}, desc_total_attn shape is {desc_total_attn.shape}")

                        # Compute attention ratios (keep as tensor)
                        desc_to_visual_attentions = desc_visual_attn / (desc_total_attn + 1e-8)
                        # print(f"desc_to_visual_attentions shape is {desc_to_visual_attentions.shape}")
                        # print(f"desc_to_visual_attentions: {desc_to_visual_attentions}")
                        # print(f"description to visual attention mean: {desc_to_visual_attentions.mean().item()}, std: {desc_to_visual_attentions.std().item()}")

                        # target_max = 0.05
                        # target_min = 0.005
                        # normalized_attentions = (desc_to_visual_attentions - target_min) / (target_max - target_min)
                        # normalized_attentions = torch.clamp(normalized_attentions, 0.0, 1.0)
                        # reward_per_token = normalized_attentions
                        target_max = desc_to_visual_attentions.max() # (batch_size)
                        target_min = target_max * 0.2   # (batch_size)

                        reward_per_token = ((desc_to_visual_attentions >= target_min) & (desc_to_visual_attentions <= target_max)).float()
                        avg_reward = float(reward_per_token.mean().item())
                        # print(f"reward per token: {reward_per_token}", flush=True)
                        # print(f"avg reward: {avg_reward}", flush=True)
                        reward_tensor[current_index] = avg_reward
                    current_index += 1


                del attentions
                torch.cuda.empty_cache()
                # attention_all.append(attentions_avg)
                # print(f"length of attention_all: {len(attention_all)}")
        # attentions_avg = torch.cat(attention_all, dim=0)  # (total_batch_size, seq_len, seq_len)
        # print(f"final attentions avg shape: {attentions_avg.shape}")
        # return attentions_avg
        # print(f"reward tensor shape is {reward_tensor.shape}")
        return reward_tensor
    
    # def find_token_sequence(self, input_ids, sequence):
    #     """Find the starting index of a token sequence"""
    #     seq_len = len(sequence)
    #     for i in range(len(input_ids) - seq_len + 1):
    #         if torch.all(input_ids[i:i+seq_len] == torch.tensor(sequence)):
    #             return i
    #     return -1

    def find_desc_token_section(self, input_ids, desc_mask, token_id):
        """Find the starting index of a token sequence. Reversely iterate over desc_mask, and check the positions of description tags if exist."""
        close_tag_idx = None
        open_tag_idx = None
        for desc_idx in reversed(desc_mask):
            # assert input_ids[desc_idx] == token_id
            # decode desc_mask strings in the input ids
            decoded_str = "".join(self.tokenizer.decode(input_ids[desc_idx - 2: desc_idx + 3], skip_special_tokens=True))
            # print(f"decoded str at position {desc_idx.item()}: {decoded_str}")
            if "</description>" in decoded_str:
                if close_tag_idx is None:
                    close_tag_idx = desc_idx.item()
            elif "<description>" in decoded_str:
                if open_tag_idx is None:
                    open_tag_idx = desc_idx.item()
        
        return open_tag_idx, close_tag_idx, (open_tag_idx is not None and close_tag_idx is not None and open_tag_idx < close_tag_idx)
        
        # second_last_index = -1
        # last_index = -1
        # for i in range(len(input_ids)):
        #     if input_ids[i] == token_id:
        #         second_last_index = last_index
        #         last_index = i
        # return second_last_index, last_index

    
    # def _find_visual_token_positions(self, input_ids, image_pad_token_id):
    #     """
    #     Find positions of visual tokens (image tokens) in the input sequence.
    #     """
    #     visual_positions = []
    #     for idx in range(input_ids.shape[0]):
    #         token_id = input_ids[idx].item()
    #         # Check if this is an image token
    #         # Qwen2.5-VL uses special image tokens
    #         if token_id == image_pad_token_id:
    #             visual_positions.append(idx)
    #     return visual_positions
    
    def _find_visual_token_positions(self, input_ids, image_pad_token_id):
        """
        Find positions of visual tokens (image tokens) in the input sequence.
        Returns tensor of positions.
        """
        return (input_ids == image_pad_token_id).nonzero(as_tuple=True)[0]

    def _format_reward_description(self, completion_content: str) -> float:
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
        pattern_description = r"^(\s*)<description>.*?</description>"
        pattern_think = r"<think>.*?</think>"
        pattern_answer = r"<answer>.*?</answer>(\s*)$"
        
        match_description = re.findall(pattern_description, completion_content, re.MULTILINE | re.DOTALL)
        match_think = re.findall(pattern_think, completion_content, re.MULTILINE | re.DOTALL)
        match_answer = re.findall(pattern_answer, completion_content, re.MULTILINE | re.DOTALL)
        
        if (match_description is not None and match_think is not None and match_answer is not None and 
            len(match_description) == 1 and len(match_think) == 1 and len(match_answer) == 1):
            return 1.0
        else:
            return 0.0   
        
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

    # @GPUMemoryLogger(role="dp actor", logger=logger)
    # def update_policy(self, data: DataProto):
    #     # make sure we are in training mode
    #     self.actor_module.train()

    #     temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

    #     select_keys = [
    #         "responses",
    #         "response_mask",
    #         "input_ids",
    #         "attention_mask",
    #         "position_ids",
    #         "old_log_probs",
    #         "advantages",
    #     ]
    #     if self.config.use_kl_loss:
    #         select_keys.append("ref_log_prob")
    #     if self.config.tis_imp_ratio_cap > 0:
    #         assert "rollout_log_probs" in data.batch.keys(), (
    #             "Truncated Importance Sampling (TIS) requires to configure "
    #             "`actor_rollout_ref.rollout.calculate_log_probs=True` "
    #             "and is not currently supported in Server mode (agent loop)."
    #         )
    #         select_keys.append("rollout_log_probs")

    #     has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
    #     non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

    #     data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

    #     # Split to make minibatch iterator for updating the actor
    #     # See PPO paper for details. https://arxiv.org/abs/1707.06347
    #     mini_batches = data.split(self.config.ppo_mini_batch_size)

    #     on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

    #     metrics = {}
    #     for _ in range(self.config.ppo_epochs):
    #         for batch_idx, mini_batch in enumerate(mini_batches):
    #             if self.config.use_dynamic_bsz:
    #                 max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
    #                 micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
    #             else:
    #                 self.gradient_accumulation = (
    #                     self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
    #                 )
    #                 micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

    #             self.actor_optimizer.zero_grad()

    #             for micro_batch in micro_batches:
    #                 micro_batch = micro_batch.to(get_device_id())
    #                 micro_batch_metrics = {}
    #                 model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    #                 response_mask = model_inputs["response_mask"]
    #                 old_log_prob = model_inputs["old_log_probs"]
    #                 rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
    #                 advantages = model_inputs["advantages"]

    #                 entropy_coeff = self.config.entropy_coeff
    #                 loss_agg_mode = self.config.loss_agg_mode

    #                 if self.config.use_dynamic_bsz:
    #                     loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
    #                 else:
    #                     loss_scale_factor = 1 / self.gradient_accumulation

    #                 # all return: (bsz, response_length)
    #                 calculate_entropy = False
    #                 if entropy_coeff != 0:
    #                     calculate_entropy = True
    #                 entropy, log_prob = self._forward_micro_batch(
    #                     model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
    #                 )

    #                 if on_policy:
    #                     old_log_prob = log_prob.detach()
    #                 else:
    #                     old_log_prob = model_inputs["old_log_probs"]

    #                 loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
    #                 # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
    #                 # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
    #                 # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
    #                 policy_loss_fn = get_policy_loss_fn(loss_mode)
    #                 pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
    #                     old_log_prob=old_log_prob,
    #                     log_prob=log_prob,
    #                     advantages=advantages,
    #                     response_mask=response_mask,
    #                     loss_agg_mode=loss_agg_mode,
    #                     config=self.config,
    #                     rollout_log_probs=rollout_log_probs,
    #                 )

    #                 if entropy_coeff != 0:
    #                     entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    #                     # compute policy loss
    #                     policy_loss = pg_loss - entropy_loss * entropy_coeff
    #                 else:
    #                     policy_loss = pg_loss

    #                 if self.config.use_kl_loss:
    #                     ref_log_prob = model_inputs["ref_log_prob"]
    #                     # compute kl loss
    #                     kld = kl_penalty(
    #                         logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
    #                     )
    #                     kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    #                     policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
    #                     micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
    #                     micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

    #                 if self.config.use_dynamic_bsz:
    #                     # relative to the dynamic bsz
    #                     loss = policy_loss * loss_scale_factor
    #                 else:
    #                     loss = policy_loss * loss_scale_factor
    #                 loss.backward()

    #                 micro_batch_metrics.update(
    #                     {
    #                         "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
    #                         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    #                         "actor/ppo_kl": ppo_kl.detach().item(),
    #                         "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    #                     }
    #                 )
    #                 append_to_dict(metrics, micro_batch_metrics)

    #             grad_norm = self._optimizer_step()
    #             mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
    #             append_to_dict(metrics, mini_batch_metrics)
    #     self.actor_optimizer.zero_grad()
    #     return metrics