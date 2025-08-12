from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple
from decimal import Decimal, InvalidOperation


DEFAULT_ANSWER_TAG_PREFIX = "<answer>"
DEFAULT_ANSWER_TAG_SUFFIX = "</answer>"


@dataclass
class MatchResult:
    is_match: bool
    pred_canonical: str
    truth_canonical: str


def extract_between_tags(
    text: str,
    tag_prefix: str = DEFAULT_ANSWER_TAG_PREFIX,
    tag_suffix: str = DEFAULT_ANSWER_TAG_SUFFIX,
) -> Optional[str]:
    """Extract the first substring between tag_prefix and tag_suffix; return None if not found."""
    if not isinstance(text, str) or not text:
        return None
    try:
        start = text.find(tag_prefix)
        if start == -1:
            return None
        start += len(tag_prefix)
        end = text.find(tag_suffix, start)
        if end == -1:
            return None
        return text[start:end].strip()
    except Exception:
        return None


def remove_tags(
    text: str,
    tag_prefix: str = DEFAULT_ANSWER_TAG_PREFIX,
    tag_suffix: str = DEFAULT_ANSWER_TAG_SUFFIX,
) -> str:
    """Remove all occurrences of the tag markers but keep inner content.

    Example: "abc <answer>42</answer> xyz" -> "abc 42 xyz"
    """
    if not isinstance(text, str) or not text:
        return text
    # Replace opening/closing tags only; keep the content
    text = text.replace(tag_prefix, "")
    text = text.replace(tag_suffix, "")
    return text


def _canonicalize_numeric(s: str) -> Optional[str]:
    """Extract the last numeric value (int/float, with optional sign) from a string.

    Returns canonical numeric string (no thousands separators) or None if not found.
    """
    if not isinstance(s, str):
        return None
    # Find all numbers (integers or decimals, possibly negative)
    # Accept forms like -12, 3.5, .75, 1,234.56 (will normalize commas later)
    cleaned = s.replace(",", "")
    matches = re.findall(r"[-+]?\d*\.?\d+", cleaned)
    if not matches:
        return None
    num_str = matches[-1].lstrip("+")
    try:
        d = Decimal(num_str)
        # normalize removes trailing zeros; format as plain string (no exponent)
        normalized = format(d.normalize(), 'f')
        # handle cases like '-0.0' -> '0'
        if normalized.startswith('-0'):
            normalized = normalized[1:]
        # remove trailing .0
        if '.' in normalized:
            normalized = normalized.rstrip('0').rstrip('.')
            if normalized == '':
                normalized = '0'
        return normalized
    except InvalidOperation:
        return num_str


def _canonicalize_text(s: str) -> str:
    """Lowercase and collapse whitespace; strip punctuation except signs and dot."""
    if not isinstance(s, str):
        return ""
    # Keep digits, letters, whitespace, sign and dot
    s = s.lower()
    s = re.sub(r"[^a-z0-9\-\+\.\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_answers(pred: str, truth: str, numeric_only: bool = True) -> Tuple[str, str]:
    """Canonicalize answers for robust comparison.

    - If numeric_only, try to extract the last number from each; if any is missing, fall back to text mode.
    - In text mode, normalize text by lowercasing and collapsing whitespace/punct.
    Returns the canonical (possibly numeric) strings.
    """
    if numeric_only:
        pred_num = _canonicalize_numeric(pred)
        truth_num = _canonicalize_numeric(truth)
        if pred_num is not None and truth_num is not None:
            return pred_num, truth_num
    # text fallback
    return _canonicalize_text(pred), _canonicalize_text(truth)


def answers_match(pred: str, truth: str, numeric_only: bool = True) -> MatchResult:
    pred_c, truth_c = canonicalize_answers(pred, truth, numeric_only=numeric_only)
    return MatchResult(is_match=(pred_c == truth_c), pred_canonical=pred_c, truth_canonical=truth_c)


def extract_final_answer(
    model_output: str,
    tag_prefix: str = DEFAULT_ANSWER_TAG_PREFIX,
    tag_suffix: str = DEFAULT_ANSWER_TAG_SUFFIX,
) -> Optional[str]:
    """Try tag-based extraction first; if not present, fall back to numeric extraction of last number."""
    if not isinstance(model_output, str) or not model_output:
        return None
    inner = extract_between_tags(model_output, tag_prefix=tag_prefix, tag_suffix=tag_suffix)
    if inner is not None and inner != "":
        return inner.strip()
    # Fallback: last number in the output
    num = _canonicalize_numeric(model_output)
    return num


def build_mismatch_hint(
    truth_final: str,
    pred_final: Optional[str],
    template: str = "Oops, I made an error, the final answer is ### {truth} but I got {pred}",
) -> str:
    pred_disp = pred_final if (pred_final is not None and pred_final != "") else "(empty)"
    return template.format(truth=truth_final, pred=pred_disp)


