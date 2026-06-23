"""Reward functions for verl GRPO training.

This module provides reward functions compatible with the verl framework.
The main entry point is compute_score(), which follows verl's expected signature.
"""

import math
import re
from typing import Optional
from mathruler.grader import grade_answer

# from latex2sympy2_extended import NormalizationConfig
# from math_verify import LatexExtractionConfig, parse, verify

# from verl.utils.reward_score.r1v import r1v_accuracy_reward, r1v_format_reward
from verl.utils.reward_score.acc_reward import r1v_accuracy_reward
# from .r1v import r1v_accuracy_reward, r1v_format_reward


# def format_reward(completion_content: str) -> float:
#     """
#     Reward function that checks if the reasoning process is enclosed within 
#     <think> and </think> tags, while the final answer is enclosed within 
#     <answer> and </answer> tags.
    
#     Args:
#         completion_content: The model's completion string
        
#     Returns:
#         1.0 if format is correct, 0.0 otherwise
#     """
#     pattern_think = r"^(\s*)<think>.*?</think>"
#     pattern_answer = r"<answer>.*?</answer>(\s*)$"
#     pattern_boxed = r"<answer>.*?\\boxed\{.*?\}.*?</answer>"
    
#     match_think = re.findall(pattern_think, completion_content, re.MULTILINE | re.DOTALL)
#     match_answer = re.findall(pattern_answer, completion_content, re.MULTILINE | re.DOTALL)
#     match_boxed = re.search(pattern_boxed, completion_content, re.MULTILINE | re.DOTALL)
    
#     if (match_think is not None and match_answer is not None and 
#         len(match_think) == 1 and len(match_answer) == 1 and
#         match_boxed is not None):
#         return 1.0
#     else:
#         return 0.0

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

def format_reward_description(completion_content: str) -> float:
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


def cosine_scaled_reward(
    completion: str,
    is_correct: bool,
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.0,
    max_value_correct: float = 0.0,
    max_len: int = 2048,
) -> float:
    """
    Reward function that scales based on completion length using a cosine schedule.
    
    Shorter correct solutions are rewarded more than longer ones.
    Longer incorrect solutions are penalized less than shorter ones.
    
    Args:
        completion: Model completion string
        is_correct: Whether the answer is correct
        min_value_wrong: Minimum reward for wrong answers
        max_value_wrong: Maximum reward for wrong answers
        min_value_correct: Minimum reward for correct answers
        max_value_correct: Maximum reward for correct answers
        max_len: Maximum length for scaling
        
    Returns:
        Scaled reward value
    """
    gen_len = len(completion)
    
    # Apply cosine scaling based on length
    progress = gen_len / max_len
    cosine = math.cos(progress * math.pi)
    
    if is_correct:
        min_value = min_value_correct
        max_value = max_value_correct
    else:
        # Swap min/max for incorrect answers
        min_value = max_value_wrong
        max_value = min_value_wrong
    
    reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
    return float(reward)


# def repetition_penalty_reward(
#     completion: str,
#     ngram_size: int = 40,
#     max_penalty: float = -0.5
# ) -> float:
#     """
#     Computes N-gram repetition penalty as described in Appendix C.2 of 
#     https://arxiv.org/abs/2502.03373.
    
#     Args:
#         completion: Model completion string
#         ngram_size: Size of the n-grams
#         max_penalty: Maximum (negative) penalty for wrong answers
        
#     Returns:
#         Penalty reward (0.0 if no repetition, negative otherwise)
#     """
#     if max_penalty > 0:
#         raise ValueError(f"max_penalty {max_penalty} should not be positive")
    
#     if completion == "" or len(completion.split()) < ngram_size:
#         return 0.0
    
#     def zipngram(text: str, n: int):
#         words = text.lower().split()
#         return zip(*[words[i:] for i in range(n)])
    
#     ngrams = set()
#     total = 0
#     for ng in zipngram(completion, ngram_size):
#         ngrams.add(ng)
#         total += 1
    
#     scaling = 1 - len(ngrams) / total
#     reward = scaling * max_penalty
#     return reward

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


# def efficient_length_reward(response_length: int, target_length: int, alpha: float=0.005, delta: float = 0.5):
#     return max(0.0, min(1.0, alpha * (target_length - response_length) + delta))

# def efficient_length_reward(
#     response_length: int,
#     target_length: int = 1500,
#     min_length: int = 400,   # don't go below this
#     alpha_upper: float = 0.005,
#     alpha_lower: float = 0.01,  # steeper penalty for being too short
#     delta: float = 0.5
# ):
#     if response_length < min_length:
#         # Hard penalty for collapsing
#         return max(0.0, delta - alpha_lower * (min_length - response_length))
#     elif response_length <= target_length:
#         # Reward for being in [min_length, target_length]
#         return min(1.0, alpha_upper * (target_length - response_length) + delta)
#     else:
#         # Penalty for being too long
#         return max(0.0, alpha_upper * (target_length - response_length) + delta)


def efficient_length_reward(
    response_length: int,
    target_length: int = 1200,
    sigma_short: float = 500.0,   # lenient on shorter responses
    sigma_long: float = 400.0,    # stricter on longer responses  
    min_length: int = 200,        # hard floor — anything below gets 0
) -> float:
    """
    Asymmetric Gaussian length reward.
    - Peaks at target_length
    - Tolerant of shorter responses (300 tokens gets ~0.4 reward)
    - Stricter on longer responses
    - Hard zero below min_length
    """
    if response_length < min_length:
        return 0.0
    
    if response_length <= target_length:
        sigma = sigma_short
    else:
        sigma = sigma_long
    
    return math.exp(-0.5 * ((response_length - target_length) / sigma) ** 2)

def accuracy_reward(
    predict_str: str,
    ground_truth: str,
    response_length: Optional[int] = None
) -> float:
    """
    Calculate accuracy reward by comparing prediction with ground truth.
    
    This is a simplified version. You should replace this with your actual
    r1v_accuracy_reward implementation or similar logic.
    
    Args:
        predict_str: Model's prediction
        ground_truth: Ground truth answer
        response_length: Maximum response length (unused in this simplified version)
        
    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Extract answer from <answer> tags if present
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if answer_match:
        answer_str = answer_match.group(1).strip()
    else:
        answer_str = predict_str.strip()
    
    # Extract boxed content if present
    if "boxed" in answer_str or "\\boxed" in answer_str:
        answer_str = extract_boxed_content(answer_str).strip()
    
    # Simple string matching (you may want to use more sophisticated matching)
    # Normalize both strings for comparison
    answer_normalized = answer_str.lower().strip()
    ground_truth_normalized = ground_truth.lower().strip()
    
    if answer_normalized == ground_truth_normalized or grade_answer(answer_normalized, ground_truth_normalized) == 1.0:
        return 1.0
    else:
        return 0.0


def compute_score(
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
    
    # Get configuration parameters with defaults
    cos_len_config = extra_info.get('cos_len_reward_config', {
        'min_value_wrong': -1.0,
        'max_value_wrong': -0.5,
        'min_value_correct': 0.0,
        'max_value_correct': 0.0,
    })
    response_length = extra_info.get('response_length', 2048)
    ngram_size = extra_info.get('ngram_size', 40)
    max_penalty = extra_info.get('max_penalty', -0.5)
    
    # Compute accuracy reward
    acc_reward = accuracy_reward(solution_str, ground_truth, response_length)
    
    # Compute format reward
    format_reward_val = format_reward(solution_str)
    
    # Compute cosine length reward
    is_correct = (acc_reward == 1.0)
    cosine_len_reward = cosine_scaled_reward(
        completion=solution_str,
        is_correct=is_correct,
        min_value_wrong=cos_len_config['min_value_wrong'],
        max_value_wrong=cos_len_config['max_value_wrong'],
        min_value_correct=cos_len_config['min_value_correct'],
        max_value_correct=cos_len_config['max_value_correct'],
        max_len=response_length,
    )
    
    # Compute repetition penalty reward
    repetition_reward = repetition_penalty_reward(
        solution_str=solution_str,
    )
    
    # Combine all rewards
    total_reward = (
        acc_reward + 
        format_reward_val + 
        cosine_len_reward + 
        repetition_reward
    )
    
    # Return dictionary with all components
    # verl will use 'score' as the main reward and log the rest
    return {
        'score': total_reward,
        'accuracy': acc_reward,
        'format': format_reward_val,
        'cosine_len': cosine_len_reward,
        'repetition': repetition_reward,
    }

def compute_score_wo_repetition(
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
    
    # Get configuration parameters with defaults
    cos_len_config = extra_info.get('cos_len_reward_config', {
        'min_value_wrong': -1.0,
        'max_value_wrong': -0.5,
        'min_value_correct': 0.0,
        'max_value_correct': 0.0,
    })
    response_length = extra_info.get('response_length', 2048)
    ngram_size = extra_info.get('ngram_size', 40)
    max_penalty = extra_info.get('max_penalty', -0.5)
    
    # Compute accuracy reward
    acc_reward = accuracy_reward(solution_str, ground_truth, response_length)
    
    # Compute format reward
    format_reward_val = format_reward(solution_str)
    
    # Compute cosine length reward
    is_correct = (acc_reward == 1.0)
    cosine_len_reward = cosine_scaled_reward(
        completion=solution_str,
        is_correct=is_correct,
        min_value_wrong=cos_len_config['min_value_wrong'],
        max_value_wrong=cos_len_config['max_value_wrong'],
        min_value_correct=cos_len_config['min_value_correct'],
        max_value_correct=cos_len_config['max_value_correct'],
        max_len=response_length,
    )
    
    # Compute repetition penalty reward
    # repetition_reward = repetition_penalty_reward(
    #     completion=solution_str,
    #     ngram_size=ngram_size,
    #     max_penalty=max_penalty
    # )
    
    # Combine all rewards
    total_reward = (
        acc_reward + 
        format_reward_val + 
        cosine_len_reward
        # repetition_reward
    )
    
    # Return dictionary with all components
    # verl will use 'score' as the main reward and log the rest
    return {
        'score': total_reward,
        'accuracy': acc_reward,
        'format': format_reward_val,
        'cosine_len': cosine_len_reward,
        'repetition': 0.0,
    }
    
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
    
    # Get configuration parameters with defaults
    cos_len_config = extra_info.get('cos_len_reward_config', {
        'min_value_wrong': -1.0,
        'max_value_wrong': -0.5,
        'min_value_correct': 0.0,
        'max_value_correct': 0.0,
    })
    # response_length = extra_info.get('response_length', 2048)
    # print(f"response length from extra_info: {extra_info['response_length']}")
    response_length = extra_info['response_length']
    ngram_size = extra_info.get('ngram_size', 40)
    max_penalty = extra_info.get('max_penalty', -0.5)
    
    # Compute accuracy reward
    # acc_reward = accuracy_reward(solution_str, ground_truth, None)
    acc_reward = r1v_accuracy_reward(solution_str, ground_truth)
    
    # Compute format reward
    format_reward_val = format_reward(solution_str)
    
    # Compute cosine length reward
    is_correct = (acc_reward == 1.0)
    # cosine_len_reward = cosine_scaled_reward(
    #     completion=solution_str,
    #     is_correct=is_correct,
    #     min_value_wrong=cos_len_config['min_value_wrong'],
    #     max_value_wrong=cos_len_config['max_value_wrong'],
    #     min_value_correct=cos_len_config['min_value_correct'],
    #     max_value_correct=cos_len_config['max_value_correct'],
    #     max_len=response_length,
    # )
    
    length_reward = efficient_length_reward(response_length, target_length=2000)
    
    # Compute repetition penalty reward
    repetition_reward = repetition_penalty_reward(
        solution_str=solution_str,
    )
    
    attention_score = extra_info.get('attention_score', 0)
    suppression_score = extra_info.get('suppression_attention_score', 0)
    # print(f"attention score from extra_info: {extra_info.get('attention_score', None)}")
    
    # Combine all rewards
    total_reward = (
        acc_reward + 
        0.3 * format_reward_val +
        0.05 * attention_score * acc_reward * format_reward_val +
        0.1 * suppression_score * acc_reward * format_reward_val +
        repetition_reward
        # 0.1 * length_reward +
        # cosine_len_reward
        # repetition_reward
    )
    
    # Return dictionary with all components
    # verl will use 'score' as the main reward and log the rest
    return {
        'score': total_reward,
        'accuracy': acc_reward,
        'format': format_reward_val,
        'attention_score': attention_score,
        'suppression_attention_score': suppression_score,
        # "length": length_reward,
        "repetition": repetition_reward,
        # 'cosine_len': 0.0,
    }



def compute_score_with_breakdown(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None
) -> dict:
    """
    Alternative reward function that returns a breakdown of all reward components.
    
    This can be useful for logging and analysis, though verl expects a float.
    You can log this information separately if needed.
    
    Returns:
        Dictionary with overall reward and individual components
    """
    # Extract configuration from extra_info if provided
    if extra_info is None:
        extra_info = {}
    
    # Get configuration parameters with defaults
    cos_len_config = extra_info.get('cos_len_reward_config', {
        'min_value_wrong': -1.0,
        'max_value_wrong': -0.5,
        'min_value_correct': 0.0,
        'max_value_correct': 0.0,
    })
    response_length = extra_info.get('response_length', 2048)
    ngram_size = extra_info.get('ngram_size', 40)
    max_penalty = extra_info.get('max_penalty', -0.5)
    
    # Compute accuracy reward
    acc_reward = accuracy_reward(solution_str, ground_truth, response_length)
    
    # Compute format reward
    format_reward_val = format_reward(solution_str)
    
    # Compute cosine length reward
    is_correct = (acc_reward == 1.0)
    cosine_len_reward = cosine_scaled_reward(
        completion=solution_str,
        is_correct=is_correct,
        min_value_wrong=cos_len_config['min_value_wrong'],
        max_value_wrong=cos_len_config['max_value_wrong'],
        min_value_correct=cos_len_config['min_value_correct'],
        max_value_correct=cos_len_config['max_value_correct'],
        max_len=response_length,
    )
    
    # Compute repetition penalty reward
    repetition_reward = repetition_penalty_reward(
        completion=solution_str,
        ngram_size=ngram_size,
        max_penalty=max_penalty
    )
    
    # Return breakdown
    return {
        "overall": acc_reward + format_reward_val + cosine_len_reward + repetition_reward,
        "accuracy": acc_reward,
        "format": format_reward_val,
        "cosine_len": cosine_len_reward,
        "repetition_penalty": repetition_reward,
    }


# Example usage and testing
if __name__ == "__main__":
    # Test example
    example_solution = """<think>
To find the maximum rating of statistical capacity in the Maldives across all years, 
I need to examine the data points for the Maldives in the graph. The graph shows the 
Maldives' rating in green. Observing the graph, the highest point for the Maldives is 
at 70, which is in 2005 and 2006. There are no other green points that exceed 70. 
Therefore, the maximum rating of statistical capacity in the Maldives is 70.

Reflection: The graph clearly indicates the ratings for the Maldives, and there are 
no inconsistencies or errors in the data points. The answer is straightforward once 
the highest point is identified.
</think>
<answer>
The maximum rating of statistical capacity in the Maldives across all years is \\boxed{70}.
</answer>"""
    
    ground_truth = "70"
    
    # Test the reward function
    reward = compute_score(
        data_source="test",
        solution_str=example_solution,
        ground_truth=ground_truth,
        extra_info={
            'response_length': 2048,
            'cos_len_reward_config': {
                'min_value_wrong': -1.0,
                'max_value_wrong': -0.5,
                'min_value_correct': 0.0,
                'max_value_correct': 0.0,
            }
        }
    )
    
    print(f"Total reward: {reward}")
    
    # Test with breakdown
    breakdown = compute_score_with_breakdown(
        data_source="test",
        solution_str=example_solution,
        ground_truth=ground_truth,
        extra_info={
            'response_length': 2048,
            'cos_len_reward_config': {
                'min_value_wrong': -1.0,
                'max_value_wrong': -0.5,
                'min_value_correct': 0.0,
                'max_value_correct': 0.0,
            }
        }
    )
    
    print("\nReward breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.4f}")