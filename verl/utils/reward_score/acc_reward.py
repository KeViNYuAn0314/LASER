"""
Improved accuracy reward for GRPO training of VLMs.

Key improvements over the original:
1. Balanced-brace boxed extraction (handles nested LaTeX like \\frac{\\sqrt{2}}{3})
2. Handles \\b (backspace) escape when Python interprets \\boxed as \\x08oxed
3. Uses math_verify as primary verifier with proper \\boxed{} wrapping for both pred and GT
4. Takes LAST \\boxed{} match (handles model self-correction patterns)
5. Fixes dangerous "not" catch-all in normalize()
6. Adds MCQ handling as a separate path
7. Adds timeout protection for math_verify
8. Fixes float_rounding_limit bug for integers
9. Proper empty-parse safety checks
"""

import re
import signal
from typing import Optional

from math_verify import parse, verify
try:
    from math_verify.errors import TimeoutException as _MVTimeout
except ImportError:
    class _MVTimeout(Exception):
        pass
from sympy import pi

# ============================================================
# Regex Patterns
# ============================================================

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Balanced-brace boxed pattern:
#   - Handles \\boxed, \boxed (single backslash), and \x08oxed (\\b escape)
#   - Supports up to 3 levels of nested braces (covers \\frac{\\sqrt{x^{2}+1}}{3})
BOXED_PATTERN = re.compile(
    r"(?:\\{1,2}boxed|\x08oxed)\{"
    r"((?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*)"
    r"\}",
    re.DOTALL,
)

# MCQ pattern: matches A, (A), A., etc.
MCQ_PATTERN = re.compile(r"^\(?([A-Za-z])\)?\.?\s*$")

# Format check pattern
FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
MIXED_NUMBER_PATTERN = re.compile(r"([0-9]) +([0-9])")

# ============================================================
# Helper: timeout wrapper (for math_verify hangs)
# ============================================================

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("math_verify timed out")


def _safe_verify(gt_parsed, pred_parsed, float_rounding: int = 4, timeout_sec: int = 5) -> bool:
    """Wrap math_verify's verify() with a timeout to prevent hangs."""
    if not gt_parsed or not pred_parsed:
        return False

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return verify(gt_parsed, pred_parsed, float_rounding=float_rounding)
    except (_MVTimeout, TimeoutError):
        return False
    except Exception:
        return False
    finally:
        # Defensively cancel any alarm math_verify may have left armed
        # (signal-based timeouts can race and fire after the call returns,
        # surfacing as "Exception ignored in WeakSet._remove" later).
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _safe_parse(expr: str) -> list:
    """Wrap math_verify's parse() with timeout + error handling."""
    try:
        return parse(expr)
    except (_MVTimeout, TimeoutError):
        return []
    except Exception:
        return []
    finally:
        signal.alarm(0)


# ============================================================
# Answer Extraction
# ============================================================

def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content from \\boxed{} with balanced brace matching.
    Takes the LAST match to handle self-correction patterns like:
        "\\boxed{99} is wrong. The answer is \\boxed{42}"
    
    Returns None if no boxed content found.
    """
    matches = list(BOXED_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_answer(predict_str: str) -> Optional[str]:
    """
    Extract the model's final answer from the full generation.
    
    Priority:
    1. \\boxed{} inside <answer> tags (last match)
    2. Raw content inside <answer> tags  
    3. None (no answer found)
    """
    # Step 1: Find <answer> content
    answer_match = ANSWER_TAG_PATTERN.search(predict_str)
    if not answer_match:
        return None
    
    answer_content = answer_match.group(1).strip()
    
    # Step 2: Extract from \\boxed{} (last match for self-correction)
    boxed_content = extract_boxed_content(answer_content)
    if boxed_content is not None:
        return boxed_content
    
    # Step 3: Fallback to raw <answer> content
    return answer_content if answer_content else None


# ============================================================
# MCQ Handling
# ============================================================

def is_mcq(text: str) -> bool:
    """Check if text is a single multiple-choice letter."""
    return bool(MCQ_PATTERN.match(text.strip()))


def extract_mcq_letter(text: str) -> Optional[str]:
    """Extract the choice letter from MCQ format."""
    m = MCQ_PATTERN.match(text.strip())
    return m.group(1).upper() if m else None


def verify_mcq(pred: str, gt: str) -> bool:
    """Compare MCQ answers."""
    pred_letter = extract_mcq_letter(pred)
    gt_letter = extract_mcq_letter(gt)
    if pred_letter and gt_letter:
        return pred_letter == gt_letter
    return False


# ============================================================
# Float rounding helper
# ============================================================

# Matches a string that contains a decimal point number (e.g., "0.33", "-1.67", "3.14")
_HAS_DECIMAL_PATTERN = re.compile(r'-?\d+\.\d+')

# Matches a string that is a plain fraction like "1/3", "-2/3", "5/3"
_PLAIN_FRACTION_PATTERN = re.compile(r'^-?\d+\s*/\s*\d+$')

# Indicators that a string is an exact/symbolic expression (not a rounded decimal)
_EXACT_INDICATORS = [
    '\\frac', '\\sqrt', '\\pi', '\\log', '\\ln', '\\sin', '\\cos',
    '\\tan', '\\exp', '\\infty', 'frac{', 'sqrt{', 'pi',
]


def _has_decimal_float(s: str) -> bool:
    """Check if string contains a decimal point number."""
    return bool(_HAS_DECIMAL_PATTERN.search(s.strip()))


def _is_exact_form(s: str) -> bool:
    """
    Check if string represents an exact mathematical form (not a rounded decimal).
    Covers: LaTeX fractions, plain fractions (1/3), sqrt, pi, etc.
    """
    s_stripped = s.strip()
    # Plain fraction like "1/3", "-2/3"
    if _PLAIN_FRACTION_PATTERN.match(s_stripped):
        return True
    # LaTeX symbolic expressions
    s_lower = s_stripped.lower()
    return any(ind in s_lower for ind in _EXACT_INDICATORS)


def _get_decimal_places(s: str) -> int:
    """
    Get the number of decimal places from a decimal string.
    
    Does NOT strip trailing zeros — they indicate the precision the GT author
    intended. "0.33" means 2dp precision, "0.330" means 3dp.
    """
    m = re.search(r'\.(\d+)', s)
    return len(m.group(1)) if m else 0


def _compute_float_rounding(pred_str: str, gt_str: str) -> int:
    """
    Compute float_rounding for math_verify to handle fraction-vs-rounded-decimal.
    
    Problem: verify() defaults to float_rounding=6. When comparing an exact form
    like 1/3 (=0.333333) against a rounded decimal "0.33", they differ at 6dp.
    We detect this mismatch and use the decimal side's precision as the rounding.
    
    Cases handled:
    - Exact vs decimal:  \\frac{1}{3} vs 0.33, 1/3 vs 0.33, \\sqrt{2} vs 1.41
    - Decimal vs exact:  0.33 vs \\frac{1}{3}, 0.67 vs 2/3
    - Both decimals:     0.333 vs 0.33 → min(3, 2) = 2
    - Both exact:        1/3 vs \\frac{1}{3} → default 6 (SymPy handles exactly)
    - Both integers:     42 vs 42 → default 6
    """
    pred_has_float = _has_decimal_float(pred_str)
    gt_has_float = _has_decimal_float(gt_str)
    pred_is_exact = _is_exact_form(pred_str)
    gt_is_exact = _is_exact_form(gt_str)
    
    dp_pred = _get_decimal_places(pred_str) if pred_has_float else 0
    dp_gt = _get_decimal_places(gt_str) if gt_has_float else 0
    
    # Case 1: Neither has decimals → exact comparison (SymPy handles it)
    if not pred_has_float and not gt_has_float:
        return 6
    
    # Case 2: Exact form vs decimal (the critical case)
    #   e.g., pred="1/3" or "\\frac{1}{3}" vs gt="0.33"
    #   e.g., pred="\\sqrt{2}" or "\\pi" vs gt="1.41" or "3.14"
    if pred_is_exact and gt_has_float:
        return max(dp_gt, 1)
    if gt_is_exact and pred_has_float:
        return max(dp_pred, 1)
    
    # Case 3: Both have decimals → use the lower precision
    if pred_has_float and gt_has_float:
        return max(min(dp_pred, dp_gt), 1)
    
    # Case 4: One has decimal but neither is obviously exact
    #   Use the decimal side's precision as a safe bet
    if pred_has_float:
        return max(dp_pred, 1)
    if gt_has_float:
        return max(dp_gt, 1)
    
    return 6  # Default fallback


# ============================================================
# Normalization (lightweight, for string fallback only)
# ============================================================
def replace_circled_numbers(text: str) -> str:
    def circled_to_digit(match):
        char = match.group(0)
        return str(ord(char) - 0x2460 + 1)
    pattern = r"[\u2460-\u2473]"
    if re.search(pattern, text) is not None:
        text = text.replace(",", "")
        return re.sub(pattern, circled_to_digit, text)
    return text

def _inject_implicit_mixed_number(step: str):
    """Convert mixed numbers: 7 3/4 => 7+3/4"""
    return MIXED_NUMBER_PATTERN.sub(r"\1+\2", step)


def fix_frac(expr: str) -> str:
    expr = re.sub(r"(?<!\\)frac", r"\\frac", expr)
    expr = re.sub(r"\\frac([^{\s])([^{\s])", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac(\{[^{}]+\})([^{\s])", r"\\frac\1{\2}", expr)
    expr = re.sub(r"\\frac([^{\s])(\{[^{}]+\})", r"\\frac{\1}\2", expr)
    expr = re.sub(r"\\frac\(([^()]+)\)\(([^()]+)\)", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac([^{\s])\(([^()]+)\)", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac\(([^()]+)\)([^{\s])", r"\\frac{\1}{\2}", expr)
    return expr


def fix_sqrt(expr: str) -> str:
    expr = re.sub(r"(?<!\\)sqrt", r"\\sqrt", expr)
    expr = re.sub(r"\\sqrt\((.*?)\)", r"\\sqrt{\1}", expr)
    expr = re.sub(r"\\sqrt(?!\{)(.)", r"\\sqrt{\1}", expr)
    return expr


def fix_pi(expr: str) -> str:
    expr = re.sub(r"(?<!\\)pi", r"\\pi", expr)
    expr = expr.replace("π", "\\pi")
    return expr


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False

def normalize(expr: str) -> str:
    """
    Normalize answer expressions.

    Ported from original with one bug fix:
    - "not" catch-all now uses word boundary to avoid corrupting
      words like "notation", "nothing", "note"
    """
    if expr is None:
        return None

    # Remove enclosing \text{}. Execute twice for two levels of nesting.
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
        "degree", "cm", "centimeter", "dm", "meter", "mile", "gram", "kilo",
        "kilogram", "kg", "liter", "second", "minute", "hour", "day", "week",
        "month", "year", "foot", "feet", "inch", "in", "yard", "square",
        "cell", "unit", "yuan", "time", "米", "厘米", "克", "千克", "公斤",
        "升", "秒", "分钟", "分", "小时", "天", "周", "月", "年", "元",
    ]:
        expr = re.sub(
            rf"{unit}(es)?(s)? *(\^({{*)[0-9]+(}}*))?([\u00B2\u00B3\u2070-\u2079]+)?",
            "", expr,
        )

    # Remove degree symbols
    expr = re.sub(r"\^ *\\circ", "", expr)
    expr = re.sub(r"\^ *{\\circ}", "", expr)
    expr = re.sub(r"\\circ", "", expr)
    expr = re.sub(r"°", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)

    # Normalize negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)

    # BUG FIX: Check "not" BEFORE space removal, using word boundary.
    # Old code did: if "not" in expr: expr = "no" AFTER space removal,
    # which incorrectly converted "notation", "nothing", "note" to "no".
    # We check here (before spaces are stripped) so word boundaries work correctly.
    # Exclusion: skip if the expression contains digits (likely a math expression).
    _NOT_EXCLUSIONS = {"notation", "nothing", "note", "notice", "notion", "notify", "notch"}
    _expr_lower_words = expr.lower().split()
    if ("not" in _expr_lower_words or expr.lower().startswith("not ")) \
            and not any(c.isdigit() for c in expr) \
            and expr.lower().strip() not in _NOT_EXCLUSIONS:
        expr = "no"
        return expr

    expr = expr.replace(" ", "")

    # Remove empty parens/braces
    expr = expr.replace("()", "")
    expr = expr.replace("{}", "")

    expr = expr.replace("√", "\\sqrt")
    expr = fix_frac(expr)
    expr = fix_sqrt(expr)
    expr = fix_pi(expr)

    # Case insensitive
    expr = expr.lower()

    # Geometry symbols
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

    # Chinese text
    expr = expr.replace("正确", "correct")
    expr = expr.replace("错误", "incorrect")
    expr = expr.replace("notlessthan", "\\geq")
    expr = expr.replace("notmorethan", "\\leq")

    # Circled numbers
    expr = replace_circled_numbers(expr)

    # Chinese enough/not enough
    if "不够" in expr:
        expr = "no"
    elif "够" in expr:
        expr = "yes"
    if "notenough" in expr:
        expr = "no"
    elif "enough" in expr:
        expr = "yes"

    return expr

def _normalize_for_comparison(expr: str) -> str:
    """
    Lightweight normalization for string-level fallback comparison.
    Only used when math_verify fails. Kept minimal to avoid the bugs
    in the original heavy normalize().
    """
    if not expr:
        return ""
    
    expr = expr.strip().lower()
    
    # Remove common LaTeX wrappers
    expr = re.sub(r"\\text\{(.*?)\}", r"\1", expr)
    expr = re.sub(r"\\text\{(.*?)\}", r"\1", expr)  # Nested
    
    # Remove dollar signs and percent escapes
    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "")
    
    # Remove outer braces
    if len(expr) > 1 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    
    # Normalize whitespace
    expr = re.sub(r'\s+', '', expr)
    
    return expr


# ============================================================
# Main Reward Functions
# ============================================================

def r1v_format_reward(predict_str: str) -> float:
    """Check if model output follows <think>...</think><answer>...</answer> format."""
    return 1.0 if FORMAT_PATTERN.fullmatch(predict_str) else 0.0


def r1v_accuracy_reward(
    predict_str: str,
    ground_truth: str,
    timeout_sec: int = 5,
) -> float:
    """
    Compute binary accuracy reward by comparing model prediction to ground truth.
    
    Pipeline:
    1. Extract answer from <answer>\\boxed{...}</answer> (balanced brace, last match)
    2. MCQ path: if GT is a single letter, do letter comparison
    3. math_verify path: wrap both in \\boxed{}, parse, verify (with timeout)
    4. String fallback: normalized string comparison
    
    Args:
        predict_str: Full model output including <think> and <answer> tags
        ground_truth: Ground truth answer string
        timeout_sec: Timeout for math_verify operations
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    try:
        ground_truth = ground_truth.strip()
        
        # ---- Step 1: Extract the predicted answer ----
        pred_answer = extract_answer(predict_str)
        if pred_answer is None:
            return 0.0
        
        # ---- Step 2: MCQ fast path ----
        if is_mcq(ground_truth):
            if is_mcq(pred_answer):
                return 1.0 if verify_mcq(pred_answer, ground_truth) else 0.0
            # GT is MCQ but pred isn't in MCQ format - still try letter extraction
            pred_letter = extract_mcq_letter(pred_answer)
            gt_letter = extract_mcq_letter(ground_truth)
            if pred_letter and gt_letter:
                return 1.0 if pred_letter == gt_letter else 0.0
            # Fall through to math_verify (handles cases like "A" parsed as symbol)
        
        # ---- Step 3: Direct string match ----
        if pred_answer.strip().lower() == ground_truth.strip().lower():
            return 1.0
        
        # ---- Step 4: Normalized string match ----
        pred_normalized = normalize(pred_answer).strip()
        gt_normalized = normalize(ground_truth).strip()

        if pred_normalized == gt_normalized or _safe_verify(pred_normalized, gt_normalized, float_rounding=2, timeout_sec=timeout_sec):
            return 1.0
        
        # ---- Step 5: math_verify path ----
        # Wrap both in \boxed{} for consistent parsing
        # This ensures math_verify uses the correct extraction priority
        # and handles complex LaTeX (sets, fracs, etc.) correctly
        pred_wrapped = f"\\boxed{{{pred_answer}}}"
        gt_wrapped = f"\\boxed{{{ground_truth}}}"
        
        pred_parsed = _safe_parse(pred_wrapped)
        gt_parsed = _safe_parse(gt_wrapped)
        
        if pred_parsed and gt_parsed:
            # Compute appropriate float rounding
            float_rounding = _compute_float_rounding(pred_answer, ground_truth)
            
            if _safe_verify(gt_parsed, pred_parsed, float_rounding=float_rounding, timeout_sec=timeout_sec):
                return 1.0
            
        if not gt_parsed:
            gt_parsed = _safe_parse(ground_truth)
            if gt_parsed and pred_parsed:
                float_rounding = _compute_float_rounding(pred_answer, ground_truth)
                if _safe_verify(gt_parsed, pred_parsed, float_rounding=float_rounding, timeout_sec=timeout_sec):
                    return 1.0
        
        if not pred_parsed:
            pred_parsed = _safe_parse(pred_answer)
            if gt_parsed and pred_parsed:
                float_rounding = _compute_float_rounding(pred_answer, ground_truth)
                if _safe_verify(gt_parsed, pred_parsed, float_rounding=float_rounding, timeout_sec=timeout_sec):
                    return 1.0
        
        # # ---- Step 6: Fallback - try GT without \boxed{} wrapper ----
        # # Some GT formats parse better without wrapping (e.g., plain "42")
        # if not gt_parsed:
        #     gt_parsed = _safe_parse(ground_truth)
        #     if gt_parsed and pred_parsed:
        #         float_rounding = _compute_float_rounding(pred_answer, ground_truth)
        #         if _safe_verify(gt_parsed, pred_parsed, float_rounding=float_rounding, timeout_sec=timeout_sec):
        #             return 1.0
        
        # ---- Step 6: Normalized string fallback ----
        pred_norm = _normalize_for_comparison(pred_answer)
        gt_norm = _normalize_for_comparison(ground_truth)
        if pred_norm and gt_norm and pred_norm == gt_norm:
            return 1.0
        
        return 0.0
    
    except Exception as e:
        # Log for debugging and alerting, not print
        # print(f"Error in r1v_accuracy_reward: {e}")
        signal.alarm(0)  # Ensure no lingering alarms
        
        return 0.0