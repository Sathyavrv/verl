# Copyright 2025
#
# Custom reward function for GSM8K that prefers answers inside <answer>...</answer>
# and falls back to default GSM8K parsing (#### number) when the tag is absent.

from __future__ import annotations

import re
from typing import Any

from verl.utils.reward_score import default_compute_score as _default_compute_score


# Capture full numeric tokens (e.g., 18, -2, 1,234, 3.14). Avoid lazy matching which caused '18' -> '8'.
_NUM_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_TAG_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)


def _extract_number(text: str | None) -> str | None:
    if not text:
        return None
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    # use the last number (most robust for solutions that include intermediate numbers)
    last = nums[-1]
    # normalize formatting
    return last.replace(",", "").replace("$", "")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    # kwargs below are optional and can be passed via custom_reward_function.reward_kwargs.*
    fallback_to_default: bool = True,
) -> float | dict[str, Any]:
    """Score function that reads final answers from <answer>...</answer> for GSM8K.

    If data_source != "openai/gsm8k", we delegate to the default scorer.
    If the tag is missing (or parsing fails), we optionally fall back to the default scorer.
    """

    if data_source != "openai/gsm8k":
        return _default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

    # Prefer the content inside <answer> ... </answer>
    match_iter = list(_TAG_RE.finditer(solution_str))
    used_answer_tag = False
    parsed_answer = None
    if match_iter:
        # choose the last tag occurrence
        used_answer_tag = True
        parsed_answer = _extract_number(match_iter[-1].group(1))

    if parsed_answer is None and fallback_to_default:
        # fall back to default GSM8K scorer (which reads after ####)
        score = _default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return {
            "score": float(score),
            "used_answer_tag": used_answer_tag,
            "parsed_answer": parsed_answer,
            "fallback": True,
        }

    if parsed_answer is None:
        return {"score": 0.0, "used_answer_tag": used_answer_tag, "parsed_answer": parsed_answer, "fallback": False}

    score = 1.0 if parsed_answer == ground_truth else 0.0
    return {"score": float(score), "used_answer_tag": used_answer_tag, "parsed_answer": parsed_answer, "fallback": False}


