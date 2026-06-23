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

import re
from typing import Dict

from math_verify import parse, verify
from mathruler.grader import extract_boxed_content
from sympy import pi


# Pre-compiled regex patterns for performance optimization
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# BOXED_PATTERN = re.compile(r"\boxed\{([^}]+)\}", re.DOTALL)
# BOXED_PATTERN1 = re.compile(r"\\boxed\{([^}]+)\}", re.DOTALL)
BOXED_PATTERN = re.compile(r"(?:\\{1,2}boxed|\x08oxed)\{([^}]+)\}", re.DOTALL)
CHOICE_PATTERN = re.compile(r"^\(?([A-Z])\)?(?:\.\s*|$|\s)")
SENTENCE_SPLIT_PATTERN = re.compile(r'[.!?]\s+')
PUNCTUATION_END_PATTERN = re.compile(r'[.!?;,]+$')
WHITESPACE_PATTERN = re.compile(r'\s+')
MIXED_NUMBER_PATTERN = re.compile(r"([0-9]) +([0-9])")

# Answer extraction patterns (ordered by priority)
ANSWER_EXTRACTION_PATTERNS = [
    re.compile(r"(?:the answer is|therefore|thus|so|hence|result is)\s*:?\s*([^.\n]+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:equals?|=)\s*([^.\n]+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:final answer|conclusion|solution)\s*:?\s*([^.\n]+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:answer|result)\s*:?\s*([^.\n]+)", re.IGNORECASE | re.DOTALL)
]

# Normalization patterns
NORMALIZATION_PATTERNS = [
    (re.compile(r"\\text\{(.*?)\}"), r"\1"),  # Remove \text{}
    (re.compile(r"\^ *\\circ"), ""),  # Remove degree symbols
    (re.compile(r"\^ *{\\circ}"), ""),
    (re.compile(r"\\circ"), ""),
    (re.compile(r"°"), ""),
    (re.compile(r",\\\\! *"), ""),  # Remove comma artifacts
    (re.compile(r"- *"), "-"),  # Normalize negative signs
    (re.compile(r"\(\)"), ""),  # Remove empty parentheses
    (re.compile(r"\{\}"), "")   # Remove empty braces
]

# Static word-to-number dictionary
WORD_TO_NUM = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
    'million': '1000000', 'billion': '1000000000'
}

# Pre-compiled word replacement pattern (more efficient than multiple regex calls)
WORD_NUM_PATTERN = re.compile(r'\b(' + '|'.join(WORD_TO_NUM.keys()) + r')\b')

# Static fraction-to-decimal dictionary
FRACTION_TO_DECIMAL = {
    '1/2': '0.5', '1/3': '0.333', '2/3': '0.667', '1/4': '0.25',
    '3/4': '0.75', '1/5': '0.2', '2/5': '0.4', '3/5': '0.6',
    '4/5': '0.8', '1/8': '0.125', '3/8': '0.375', '5/8': '0.625',
    '7/8': '0.875'
}

# Common equivalent expressions
EQUIVALENCE_MAP = {
    'yes': 'true', 'no': 'false', 'correct': 'true', 'incorrect': 'false',
    'right': 'true', 'wrong': 'false', 'million': '*10^6',
    'billion': '*10^9', 'trillion': '*10^12'
}

# Unit removal pattern (combined for efficiency)
UNIT_LIST = [
    "degree", "cm", "centimeter", "dm", "meter", "mile", "gram", "kilo",
    "kilogram", "kg", "liter", "second", "minute", "hour", "day", "week",
    "month", "year", "foot", "feet", "inch", "in", "yard", "square",
    "cell", "unit", "yuan", "time", "米", "厘米", "克", "千克", "公斤",
    "升", "秒", "分钟", "分", "小时", "天", "周", "月", "年", "元"
]
UNIT_REMOVAL_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(unit) for unit in UNIT_LIST) + 
    r')(es)?(s)? *(\^({*)[0-9]+(}*))?([\u00B2\u00B3\u2070-\u2079]+)?\b',
    re.IGNORECASE
)

# Simple cache for normalized strings (LRU-like with size limit)
_normalization_cache = {}
_CACHE_SIZE_LIMIT = 1000


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    return MIXED_NUMBER_PATTERN.sub(r"\1+\2", step)


def fix_frac(expr: str) -> str:
    # frac{xxx}{xxx} -> \frac{xxx}{xxx}
    expr = re.sub(r"(?<!\\)frac", r"\\frac", expr)
    # \fracab, \frac{a}b, \fraca{b}, \frac(a)b, \fraca(b), \frac(a)(b) -> \frac{a}{b}
    expr = re.sub(r"\\frac([^{\s])([^{\s])", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac(\{[^{}]+\})([^{\s])", r"\\frac\1{\2}", expr)
    expr = re.sub(r"\\frac([^{\s])(\{[^{}]+\})", r"\\frac{\1}\2", expr)
    expr = re.sub(r"\\frac\(([^()]+)\)\(([^()]+)\)", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac([^{\s])\(([^()]+)\)", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac\(([^()]+)\)([^{\s])", r"\\frac{\1}{\2}", expr)
    return expr


def fix_sqrt(expr: str) -> str:
    # sqrt{xxx} -> \sqrt{xxx}
    expr = re.sub(r"(?<!\\)sqrt", r"\\sqrt", expr)
    # \sqrt(xxx) -> \sqrt{xxx}
    expr = re.sub(r"\\sqrt\((.*?)\)", r"\\sqrt{\1}", expr)
    # \sqrtxxxx -> \sqrt{x}xxx
    expr = re.sub(r"\\sqrt(?!\{)(.)", r"\\sqrt{\1}", expr)
    return expr


def fix_pi(expr: str) -> str:
    # pi -> \pi
    expr = re.sub(r"(?<!\\)pi", r"\\pi", expr)
    expr = expr.replace("π", "\\pi")
    return expr


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_decimal_places(s):
    match = re.search(r"\.(\d+)", s)
    return len(match.group(1)) if match else 0


def replace_circled_numbers(text: str) -> str:
    def circled_to_digit(match):
        char = match.group(0)
        return str(ord(char) - 0x2460 + 1)

    pattern = r"[\u2460-\u2473]"
    if re.search(pattern, text) is not None:
        text = text.replace(",", "")
        return re.sub(pattern, circled_to_digit, text)
    return text


def normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    # m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    # if m is not None:
    #     expr = m.group("text")
    # Remove enclosing `\text{}`. Execute twice to account for two levels of nesting.
    expr = re.sub(r"\\text\{(.*?)\}", r"\1", expr)
    expr = re.sub(r"\\text\{(.*?)\}", r"\1", expr)

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "dm",
        "meter",
        "mile",
        "gram",
        "kilo",
        "kilogram",
        "kg",
        "liter",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "in",
        "yard",
        "square",
        "cell",
        "unit",
        "yuan",
        "time",
        "米",
        "厘米",
        "克",
        "千克",
        "公斤",
        "升",
        "秒",
        "分钟",
        "分",
        "小时",
        "天",
        "周",
        "月",
        "年",
        "元",
    ]:
        # end by es/s, ^d or unicode superscript
        expr = re.sub(
            rf"{unit}(es)?(s)? *(\^({{*)[0-9]+(}}*))?([\u00B2\u00B3\u2070-\u2079]+)?",
            "",
            expr,
        )

    # delete \cric or ^\cric or ^{\cric} or unicode format degree
    expr = re.sub(r"\^ *\\circ", "", expr)
    expr = re.sub(r"\^ *{\\circ}", "", expr)
    expr = re.sub(r"\\circ", "", expr)
    expr = re.sub(r"°", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    # if _is_float(expr) and _is_int(float(expr)):
    #     expr = str(int(round(float(expr))))
    # if "\\" in expr:
    #     try:
    #         expr = _parse_latex(expr)
    #     except:
    #         pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    # expr = expr.replace("{", "")
    # expr = expr.replace("}", "")

    # if we somehow still have blank (), drop them
    expr = expr.replace("()", "")
    expr = expr.replace("{}", "")

    expr = expr.replace("√", "\\sqrt")
    expr = fix_frac(expr)
    expr = fix_sqrt(expr)
    expr = fix_pi(expr)

    # don't be case sensitive for text answers
    expr = expr.lower()

    # if _str_is_int(expr):
    #     expr = str(_str_to_int(expr))
    # Geometry
    expr = expr.replace("\\parallel", "//")
    expr = expr.replace("平行", "//")
    expr = expr.replace("⊥", "\\perp")
    expr = expr.replace("△", "\\triangle")
    expr = expr.replace("Δ", "\\triangle")
    expr = expr.replace("∠", "\\angle")
    expr = expr.replace("∽", "\\sim")
    expr = expr.replace("角", "\\angle")
    expr = expr.replace("平面", "plane")
    expr = expr.replace("且", "and")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("正确", "correct")
    expr = expr.replace("错误", "incorrect")
    expr = expr.replace("notlessthan", "\\geq")
    expr = expr.replace("notmorethan", "\\leq")
    expr = replace_circled_numbers(expr)
    if "不够" in expr:
        expr = "no"
    elif "够" in expr:
        expr = "yes"
    if "notenough" in expr:
        expr = "no"
    elif "enough" in expr:
        expr = "yes"
    if "not" in expr:
        expr = "no"

    return expr


def is_choice_format(expr):
    """Check if expression is in multiple choice format using pre-compiled pattern."""
    if not expr:
        return False
    expr = expr.strip().upper()
    return bool(CHOICE_PATTERN.match(expr))


# Enhanced verification functions
def extract_answer_candidates(predict_str: str, max_candidates: int = 5) -> list[str]:
    """Extract answer candidates using pre-compiled patterns, prioritized by confidence."""
    candidates = []
    
    # Priority 1: <answer> tags (highest confidence) - use pre-compiled pattern
    answer_match = ANSWER_TAG_PATTERN.search(predict_str)
    if answer_match:
        # candidates.append((answer_match.group(1).strip(), 1))  # (candidate, priority)
        answer = answer_match.group(1).strip()
        boxed_matches = BOXED_PATTERN.search(answer)
        if boxed_matches:
            candidates.append((boxed_matches.group(1).strip(), 1))
    
    
    # Sort by priority and remove duplicates
    candidates.sort(key=lambda x: x[1])  # Sort by priority (lower is better)
    unique_candidates = []
    seen = set()
    
    for candidate, priority in candidates:
        clean_candidate = candidate.strip()
        if clean_candidate and clean_candidate not in seen and len(clean_candidate) > 0:
            unique_candidates.append(clean_candidate)
            seen.add(clean_candidate)
            if len(unique_candidates) >= max_candidates:  # Limit candidates for efficiency
                break
    
    return unique_candidates


def normalize_heavy(expr: str) -> str:
    """Optimized enhanced normalization for natural language answers."""
    if not expr:
        return ""
    
    # Create safe cache key from original input (limit length to prevent memory issues)
    original_expr = str(expr)
    cache_key = original_expr[:500] if len(original_expr) > 500 else original_expr
    if cache_key in _normalization_cache:
        return _normalization_cache[cache_key]
    
    # Manage cache size
    if len(_normalization_cache) >= _CACHE_SIZE_LIMIT:
        # Simple cache eviction - remove first 100 items
        keys_to_remove = list(_normalization_cache.keys())[:100]
        for key in keys_to_remove:
            del _normalization_cache[key]
    
    # Start processing - work on a copy
    expr = original_expr.strip().lower()
    
    # Remove common punctuation at the end (using pre-compiled pattern)
    expr = PUNCTUATION_END_PATTERN.sub('', expr)
    
    # Efficient word-to-number replacement using single regex
    def word_replacer(match):
        return WORD_TO_NUM[match.group(0)]
    
    expr = WORD_NUM_PATTERN.sub(word_replacer, expr)
    
    # Batch replace equivalent expressions (more efficient than individual replaces)
    for old, new in EQUIVALENCE_MAP.items():
        if old in expr:  # Only replace if present
            expr = expr.replace(old, new)
    
    # Batch replace fractions (only if fractions are present)
    if '/' in expr:
        for frac, dec in FRACTION_TO_DECIMAL.items():
            if frac in expr:
                expr = expr.replace(frac, dec)
    
    # Remove units using pre-compiled pattern
    expr = UNIT_REMOVAL_PATTERN.sub('', expr)
    
    # Apply pre-compiled normalization patterns for mathematical expressions
    if any(char in expr for char in ['\\', '{', '}', '^', '_', '°', '(']):
        for pattern, replacement in NORMALIZATION_PATTERNS:
            expr = pattern.sub(replacement, expr)
    
    # Normalize whitespace using pre-compiled pattern
    expr = WHITESPACE_PATTERN.sub(' ', expr).strip()
    
    # Apply existing normalization as final step for complex mathematical expressions
    if any(char in expr for char in ['\\', '{', '}', '^', '_']):
        try:
            expr = normalize(expr)
        except:
            pass
    
    # Cache result
    _normalization_cache[cache_key] = expr
    return expr



def r1v_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0
    # if format_match:
    #     content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL).group(1).strip()
    #     boxed_pattern = re.compile(r"\\boxed\{.*?\}", re.DOTALL)
    #     boxed_match = re.search(boxed_pattern, content_match)
    #     if boxed_match is not None:
    #         return 1.0
    # return 0.0





def r1v_accuracy_reward(
    predict_str: str, ground_truth: str, response_length=None
) -> float:
    try:
        ground_truth = ground_truth.strip()
        
        # Enhanced answer extraction with limited candidates for efficiency
        answer_candidates = extract_answer_candidates(predict_str, max_candidates=3)
        if not answer_candidates:
            # Fallback to original extraction method using pre-compiled pattern
            content_match = ANSWER_TAG_PATTERN.search(predict_str)
            if content_match:
                answer_candidates = [content_match.group(1).strip()]
            else:
                answer_candidates = [predict_str.strip()[:200]]  # Limit length for efficiency
        
        # Try verification with each candidate
        for candidate in answer_candidates:
            # pred_answer = extract_boxed_content(candidate).strip()
            pred_answer = candidate.strip()
            
            # Handle multiple choice questions using pre-compiled pattern
            if is_choice_format(pred_answer) or is_choice_format(ground_truth):
                try:
                    pred_choice = CHOICE_PATTERN.match(pred_answer.strip().upper())
                    gt_choice = CHOICE_PATTERN.match(ground_truth.strip().upper())
                    
                    if pred_choice and gt_choice:
                        if pred_choice.group(1) == gt_choice.group(1):
                            return 1.0
                        else:
                            continue  # Try next candidate
                except:
                    continue
            
            # Direct string match (existing logic)
            if pred_answer.lower() == ground_truth.lower():
                return 1.0
            
            # Normalized match (existing logic)
            pred_normalized = normalize(pred_answer).strip()
            gt_normalized = normalize(ground_truth).strip()
            if pred_normalized == gt_normalized:
                return 1.0
            
            
            # Mathematical verification (existing logic)
            try:
                if _is_float(pred_normalized) and _is_float(gt_normalized):
                    float_rounding_limit = min(
                        len(pred_normalized.split(".")[-1]), len(gt_normalized.split(".")[-1])
                    )
                elif "pi" in pred_normalized or "pi" in gt_normalized:
                    float_rounding_limit = 2
                else:
                    float_rounding_limit = 4
                
                pred_parsed = parse(f"\\boxed{{{pred_normalized}}}")
                gt_parsed = parse(f"\\boxed{{{gt_normalized}}}")
                
                # consider the constant pi
                pred_parsed[0] = pred_parsed[0].subs(pi, 3.14)
                gt_parsed[0] = gt_parsed[0].subs(pi, 3.14)
                
                if verify(pred_parsed, gt_parsed, float_rounding=float_rounding_limit):
                    return 1.0
                
                # print(f"candidate is {candidate}, gt is {ground_truth}, pred_normalized is {pred_normalized}, gt_normalized is {gt_normalized}")
            except:
                # If mathematical parsing fails, continue to next candidate
                continue
        
        
        return 0.0
        
    except Exception as e:
        if isinstance(e, ValueError):
            print(
                f"Error occurred when computing reward!\n[[predict_str]] {predict_str}\n[[ground_truth]] {ground_truth}"
            )
        return 0.0



def progressive_verify(pred_answer: str, ground_truth: str) -> bool:
    """Optimized progressive verification ordered by computational cost."""
    if not pred_answer or not ground_truth:
        return False
    
    # Normalize both once and reuse (caching handles repeated normalizations)
    pred_norm = normalize_heavy(pred_answer)
    gt_norm = normalize_heavy(ground_truth)
    
    # Layer 1: Direct string comparison (cheapest operation)
    if pred_norm == gt_norm:
        return True
    
    # Layer 2: Simple containment checks (cheap string operations)
    pred_len = len(pred_norm)
    gt_len = len(gt_norm)
    
    # For very short ground truth (likely specific answers), check containment
    if gt_len <= 10 and pred_len > gt_len:
        if gt_norm in pred_norm:
            return True
    elif pred_len <= 10 and gt_len > pred_len:
        if pred_norm in gt_norm:
            return True
    elif abs(pred_len - gt_len) < 20:
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return True
    
    # Layer 3: Token overlap (more expensive due to splitting and set operations)
    # Only proceed if strings have reasonable token overlap potential
    if pred_len > 2 and gt_len > 2:  # Skip very short strings
        pred_tokens = set(pred_norm.split())
        gt_tokens = set(gt_norm.split())
        
        if not gt_tokens:
            return False
            
        overlap_ratio = len(pred_tokens & gt_tokens) / len(gt_tokens)
        
        # High threshold check first
        if overlap_ratio >= 0.8:
            return True
            
        # Lower threshold for very short ground truth
        if len(gt_tokens) <= 3 and overlap_ratio >= 0.6:
            return True
    
    return False

def r1v_accuracy_only_reward(
    predict_str: str, ground_truth: str, response_length=None
) -> float:
    try:
        ground_truth = ground_truth.strip()
        
        # Enhanced answer extraction with limited candidates for efficiency
        answer_candidates = extract_answer_candidates(predict_str, max_candidates=3)
        if not answer_candidates:
            # Fallback to original extraction method using pre-compiled pattern
            content_match = ANSWER_TAG_PATTERN.search(predict_str)
            if content_match:
                answer_candidates = [content_match.group(1).strip()]
            else:
                answer_candidates = [predict_str.strip()[:200]]  # Limit length for efficiency
        
        # Try verification with each candidate
        for candidate in answer_candidates:
            pred_answer = extract_boxed_content(candidate).strip()
            
            # Handle multiple choice questions using pre-compiled pattern
            if is_choice_format(pred_answer) or is_choice_format(ground_truth):
                try:
                    pred_choice = CHOICE_PATTERN.match(pred_answer.strip().upper())
                    gt_choice = CHOICE_PATTERN.match(ground_truth.strip().upper())
                    
                    if pred_choice and gt_choice:
                        if pred_choice.group(1) == gt_choice.group(1):
                            return 1.0
                        else:
                            continue  # Try next candidate
                except:
                    continue
            
            # Direct string match
            if pred_answer == ground_truth:
                return 1.0
            
            # Normalized match
            pred_normalized = normalize(pred_answer).strip()
            gt_normalized = normalize(ground_truth).strip()
            if pred_normalized == gt_normalized:
                return 1.0
            
            # Enhanced verification
            if progressive_verify(pred_answer, ground_truth):
                return 1.0
            
            # Mathematical verification
            try:
                if _is_float(pred_normalized) and _is_float(gt_normalized):
                    float_rounding_limit = min(
                        len(pred_normalized.split(".")[-1]), len(gt_normalized.split(".")[-1])
                    )
                elif "pi" in pred_normalized or "pi" in gt_normalized:
                    float_rounding_limit = 2
                else:
                    float_rounding_limit = 4
                
                pred_parsed = parse(f"\\boxed{{{pred_normalized}}}")
                gt_parsed = parse(f"\\boxed{{{gt_normalized}}}")
                
                # consider the constant pi
                pred_parsed[0] = pred_parsed[0].subs(pi, 3.14)
                gt_parsed[0] = gt_parsed[0].subs(pi, 3.14)
                
                if verify(pred_parsed, gt_parsed, float_rounding=float_rounding_limit):
                    return 1.0
            except:
                # If mathematical parsing fails, continue to next candidate
                continue
        
        # Enhanced fallback verification for natural language answers
        if progressive_verify(predict_str, ground_truth):
            return 0.1  # Partial credit for matching in full text
        
        return 0.0
        
    except Exception as e:
        if isinstance(e, ValueError):
            print(
                f"Error occurred when computing reward!\n[[predict_str]] {predict_str}\n[[ground_truth]] {ground_truth}"
            )
        return 0.0


def r1v_compute_score_(
    predict_str: str, ground_truth: str, validation: bool = False, response_length=None
) -> float:
    acc_reward = r1v_accuracy_reward(predict_str, ground_truth, response_length)
    format_reward = r1v_format_reward(predict_str)
    if validation:
        reward = acc_reward if acc_reward == 1.0 else 0.0
    else:
        train_acc_reward = acc_reward
        reward = train_acc_reward * 0.9 + format_reward * 0.1
    # reward /= 2
    return reward


def r1v_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format = r1v_format_reward(predict_str)
    accuracy = r1v_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.5 * accuracy + 0.5 * format,
        "format": format,
        "accuracy": accuracy,
    }
