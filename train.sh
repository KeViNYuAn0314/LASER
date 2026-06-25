#!/usr/bin/env bash
# ============================================================================
# LASER: A Corrective Lens for LVLMs via Visual Attention Preservation and
#        Sink Suppression
#
# GRPO post-training stage. During rollout the actor extracts the model's
# visual self-attention and converts it into two complementary rewards:
#   * Visual Grounding Reward  (R_vis)  - sustains attention on informative,
#     non-sink visual tokens across decoding steps.
#   * Sink Suppression Reward  (R_supp) - penalizes attention concentration
#     on task-irrelevant visual "sink" tokens.
# Both are gated by answer correctness and added to the accuracy reward.
#
# Built on the verl RL framework. Reference model: Qwen2.5-VL-7B-Instruct.
#
# Usage:
#   1. Edit the paths in the "REQUIRED" block below (or pass them as env vars).
#   2. bash train.sh
#   Any field can be overridden on the command line, e.g.
#      N_GPUS=8 ACTOR_LR=1e-6 bash train.sh
#   Extra Hydra overrides can be appended:
#      bash train.sh actor_rollout_ref.rollout.gpu_memory_utilization=0.5
# ============================================================================
set -x

# Resolve repo root so the custom reward function points at this repo's copy.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------- REQUIRED: set these --------------------------
# Path to the policy model. Use the cold-start checkpoint to reproduce the
# paper, or Qwen/Qwen2.5-VL-7B-Instruct to train from the instruct model.
MODEL_PATH=${MODEL_PATH:-"/path/to/Qwen2.5-VL-7B-Instruct"}
# Parquet files in verl's multimodal RL format (column `images` for images).
TRAIN_FILES=${TRAIN_FILES:-"/path/to/train.parquet"}
VAL_FILES=${VAL_FILES:-"/path/to/val.parquet"}
# Where checkpoints and logs are written.
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_DIR}/checkpoints"}

PROJECT_NAME=${PROJECT_NAME:-"laser"}
EXP_NAME=${EXP_NAME:-"qwen2_5_vl_7b_laser"}

# ----------------------------- resources ------------------------------------
N_GPUS=${N_GPUS:-4}              # GPUs per node (paper uses 4x H100)
NNODES=${NNODES:-1}
INFER_TP=${INFER_TP:-2}          # vLLM tensor-parallel size (must divide N_GPUS)
NUM_CPUS=${NUM_CPUS:-20}         # ray init cpus

# ----------------------------- LASER switches -------------------------------
ENABLE_ATTENTION=${ENABLE_ATTENTION:-True}              # master switch for the attention rewards
APPLY_HOOK_ATTENTION=${APPLY_HOOK_ATTENTION:-True}      # memory-efficient hook-based attention capture
APPLY_RECTIFICATION=${APPLY_RECTIFICATION:-True}        # identify visual sink tokens
APPLY_SINK_SUPPRESSION=${APPLY_SINK_SUPPRESSION:-True}  # R_supp  (requires rectification)
APPLY_EARLY_WEIGHTED_STABILITY=${APPLY_EARLY_WEIGHTED_STABILITY:-True}  # R_vis with early-stage emphasis
ATTENTION_START_STEP=${ATTENTION_START_STEP:-20}        # begin the attention rewards at this global step

# ----------------------------- training hyperparameters ---------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
ACTOR_LR=${ACTOR_LR:-2e-6}
ROLLOUT_N=${ROLLOUT_N:-8}        # rollouts per prompt (GRPO group size)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=dapo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=20 \
    data.truncation=error \
    data.image_key=images \
    +data.gen_batch_size=${TRAIN_BATCH_SIZE} \
    custom_reward_function.path="${REWARD_FN_PATH:-${REPO_DIR}/verl/utils/reward_score/openr1_verl.py}" \
    custom_reward_function.name=compute_score_vanilla \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${MAX_RESPONSE_LENGTH} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.apply_hook_attention=${APPLY_HOOK_ATTENTION} \
    +actor_rollout_ref.actor.apply_rectification=${APPLY_RECTIFICATION} \
    +actor_rollout_ref.actor.apply_sink_suppression=${APPLY_SINK_SUPPRESSION} \
    +actor_rollout_ref.actor.apply_early_weighted_stability=${APPLY_EARLY_WEIGHTED_STABILITY} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.log_val_generations=5 \
    +trainer.enable_attention=${ENABLE_ATTENTION} \
    +trainer.attention_start_step=${ATTENTION_START_STEP} \
    ray_kwargs.ray_init.num_cpus=${NUM_CPUS} \
    +ray_kwargs.ray_init.object_store_memory=80000000000 \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@"
