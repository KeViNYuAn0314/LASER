"""Reward functions for verl GRPO training.

This module provides reward functions compatible with the verl framework.
The main entry point is compute_score_vanilla(), which follows verl's expected signature.
"""

import re
from typing import Optional

from verl.utils.reward_score.acc_reward import r1v_accuracy_reward

# LASER reward weights
FORMAT_REWARD_WEIGHT = 0.3
ATTENTION_SCORE_WEIGHT = 0.05
SUPPRESSION_SCORE_WEIGHT = 0.1


def format_reward(completion_content: str) -> float:
    """
    Reward function that checks if the reasoning process is enclosed within
    <think> and </think> tags, while the final answer is enclosed within
    <answer> and </answer> tags. Each tag pair must occur exactly once,
    and there must be exactly one \\boxed{} within the answer.
    
    Args:
        completion_content: The model's completion string
        
    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    # Count tag occurrences
    think_open = len(re.findall(r"<think>", completion_content))
    think_close = len(re.findall(r"</think>", completion_content))
    answer_open = len(re.findall(r"<answer>", completion_content))
    answer_close = len(re.findall(r"</answer>", completion_content))
    
    # Check exactly one of each tag
    if not (think_open == 1 and think_close == 1 and 
            answer_open == 1 and answer_close == 1):
        return 0.0
    
    # Check structure: starts with <think>, ends with </answer>
    pattern_structure = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    if not re.search(pattern_structure, completion_content, re.DOTALL):
        return 0.0
    
    # Extract answer content and count \boxed{} occurrences
    answer_match = re.search(r"<answer>(.*?)</answer>", completion_content, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1)
        boxed_count = len(re.findall(r"\\boxed\{", answer_content))
        if boxed_count != 1:
            return 0.0
    else:
        return 0.0
    
    return 1.0


def repetition_penalty_reward(
    solution_str: str,
    ngram_sizes: list[int] = [8, 15, 40],
    max_penalty: float = -5
) -> float:
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")
    
    text = solution_str.strip()
    words = text.lower().split()
    
    if len(words) < min(ngram_sizes):
        return 0.0
    
    worst_scaling = 0.0
    
    for n in ngram_sizes:
        if len(words) < n:
            continue
        
        ngrams = set()
        total = 0
        for ng in zip(*[words[i:] for i in range(n)]):
            ngrams.add(ng)
            total += 1
        
        if total == 0:
            continue
        
        scaling = 1 - len(ngrams) / total
        worst_scaling = max(worst_scaling, scaling)
    
    return worst_scaling * max_penalty


def compute_score_vanilla(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None
) -> dict:
    """
    Main reward function compatible with verl framework.
    
    IMPORTANT: This function now returns a dictionary instead of a float!
    verl's RewardManager will extract the 'score' key as the main reward,
    and log all other keys as separate metrics to WandB.
    
    This function follows the verl signature:
    - data_source: The source/dataset name
    - solution_str: The model's generated solution
    - ground_truth: The ground truth answer
    - extra_info: Optional dictionary with additional info
    
    Returns:
        Dictionary containing:
        - 'score': The total combined reward (required by verl)
        - 'accuracy': Accuracy component
        - 'format': Format component
        - 'cosine_len': Cosine length component
        - 'repetition': Repetition penalty component
    """
    # Extract configuration from extra_info if provided
    if extra_info is None:
        extra_info = {}

    # Compute accuracy reward
    acc_reward = r1v_accuracy_reward(solution_str, ground_truth)

    # Compute format reward
    format_reward_val = format_reward(solution_str)

    # Compute repetition penalty reward
    repetition_reward = repetition_penalty_reward(
        solution_str=solution_str,
    )

    attention_score = extra_info.get('attention_score', 0)
    suppression_score = extra_info.get('suppression_attention_score', 0)

    # Combine all rewards
    total_reward = (
        acc_reward +
        FORMAT_REWARD_WEIGHT * format_reward_val +
        ATTENTION_SCORE_WEIGHT * attention_score * acc_reward * format_reward_val +
        SUPPRESSION_SCORE_WEIGHT * suppression_score * acc_reward * format_reward_val +
        repetition_reward
    )

    # Return dictionary with all components
    # verl will use 'score' as the main reward and log the rest
    return {
        'score': total_reward,
        'accuracy': acc_reward,
        'format': format_reward_val,
        'attention_score': attention_score,
        'suppression_attention_score': suppression_score,
        "repetition": repetition_reward,
    }