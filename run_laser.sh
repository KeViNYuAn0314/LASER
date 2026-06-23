#!/usr/bin/env bash
# ============================================================================
# LASER: attention-guided RL for VLMs (built on verl).
# During training the actor retrieves Qwen2.5-VL self-attention and turns the
# description->visual-token attention into stability / rectification /
# sink-suppression reward signals that reweight the RL objective.
#
# All machine-specific paths are parameters below; override via environment,
# e.g.   MODEL_PATH=/path/to/model N_GPUS=4 bash run_laser.sh
# A small smoke test is available with   DEBUG=True bash run_laser.sh
# ============================================================================
set -x

# Resolve the repo root from this script's location so the custom reward
# function always points at THIS repo's copy (no hardcoded absolute path).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------- paths (override me) --------------------------
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-7B-Instruct"}
TRAIN_FILES=${TRAIN_FILES:-"${REPO_DIR}/data/train.parquet"}
VAL_FILES=${VAL_FILES:-"${REPO_DIR}/data/val.parquet"}
REWARD_FN_PATH=${REWARD_FN_PATH:-"${REPO_DIR}/verl/utils/reward_score/openr1_verl.py"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_DIR}/checkpoints"}

PROJECT_NAME=${PROJECT_NAME:-"laser"}
EXP_NAME=${EXP_NAME:-"qwen2_5_vl_7b_laser"}

# ----------------------------- resources (override me) ----------------------
N_GPUS=${N_GPUS:-2}
NNODES=${NNODES:-1}
INFER_TP=${INFER_TP:-2}          # vllm tensor parallel size
NUM_CPUS=${NUM_CPUS:-20}         # ray init cpus

# ----------------------------- LASER method switches ------------------------
ENABLE_ATTENTION=${ENABLE_ATTENTION:-True}             # master switch for the LASER attention reward
APPLY_HOOK_ATTENTION=${APPLY_HOOK_ATTENTION:-True}     # hook-based capture (memory-efficient) vs output_attentions
APPLY_RECTIFICATION=${APPLY_RECTIFICATION:-True}       # sink-token rectification
APPLY_SINK_SUPPRESSION=${APPLY_SINK_SUPPRESSION:-True} # sink-suppression reward (requires rectification)
APPLY_EARLY_WEIGHTED_STABILITY=${APPLY_EARLY_WEIGHTED_STABILITY:-True}  # early-weighted vs uniform stability
ATTENTION_START_STEP=${ATTENTION_START_STEP:-20}       # begin attention reward at this global step

# ----------------------------- training hyperparameters ---------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
ACTOR_LR=${ACTOR_LR:-2e-6}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

# ----------------------------- debug / smoke test ---------------------------
# DEBUG=True shrinks everything for a quick end-to-end runnability check and
# turns the attention reward on from step 0.
if [ "${DEBUG:-False}" = "True" ]; then
    echo "===== DEBUG (smoke test) ====="
    TRAIN_BATCH_SIZE=8
    PPO_MINI_BATCH_SIZE=8
    ROLLOUT_N=2
    MAX_RESPONSE_LENGTH=512
    TOTAL_EPOCHS=1
    SAVE_FREQ=-1
    TEST_FREQ=-1
    VAL_BEFORE_TRAIN=False
    ATTENTION_START_STEP=0
fi

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
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score_vanilla \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
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
