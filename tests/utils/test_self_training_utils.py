import pytest
import runpy
import pathlib

SELF_TRAINING_PATH = pathlib.Path(__file__).resolve().parents[2] / "verl" / "utils" / "self_training.py"
ns = runpy.run_path(str(SELF_TRAINING_PATH))

DEFAULT_ANSWER_TAG_PREFIX = ns["DEFAULT_ANSWER_TAG_PREFIX"]
DEFAULT_ANSWER_TAG_SUFFIX = ns["DEFAULT_ANSWER_TAG_SUFFIX"]
MatchResult = ns["MatchResult"]
answers_match = ns["answers_match"]
build_mismatch_hint = ns["build_mismatch_hint"]
canonicalize_answers = ns["canonicalize_answers"]
extract_between_tags = ns["extract_between_tags"]
extract_final_answer = ns["extract_final_answer"]
remove_tags = ns["remove_tags"]


def test_extract_between_tags_basic():
    s = "abc <answer>42</answer> xyz"
    assert extract_between_tags(s) == "42"


def test_extract_between_tags_missing():
    s = "no tags here"
    assert extract_between_tags(s) is None


def test_remove_tags_keeps_inner():
    s = "Hello <answer>42</answer> World"
    assert remove_tags(s) == "Hello 42 World"


@pytest.mark.parametrize(
    "pred,truth,exp",
    [
        ("the result is 42", "42", True),
        ("answer: 3.50", "3.5", True),
        ("-12", "-12", True),
        ("12", "+12", True),
        ("abc", "42", False),
    ],
)
def test_answers_match_numeric(pred, truth, exp):
    res: MatchResult = answers_match(pred, truth, numeric_only=True)
    assert res.is_match is exp


def test_canonicalize_text_fallback():
    a, b = canonicalize_answers("Hello, World!", "hello world", numeric_only=True)
    assert a == b == "hello world"


def test_extract_final_answer_prefers_tags():
    s = "thinking... <answer> 13 </answer> tail"
    assert extract_final_answer(s) == "13"


def test_extract_final_answer_numeric_fallback():
    s = "some calc -> -7 and done"
    assert extract_final_answer(s) == "-7"


def test_build_mismatch_hint():
    hint = build_mismatch_hint(truth_final="42", pred_final="41")
    assert "42" in hint and "41" in hint


