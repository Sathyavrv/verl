import unittest

from recipe.reward.gsm8k_answer_tag import compute_score


class TestGSM8KAnswerTag(unittest.TestCase):
    def test_parse_simple_integer(self):
        sol = """
We compute 9 + 9. Therefore the result is <answer>18</answer>.
"""
        res = compute_score(
            data_source="openai/gsm8k", solution_str=sol, ground_truth="18", extra_info=None, fallback_to_default=False
        )
        assert isinstance(res, dict)
        assert res["used_answer_tag"] is True
        assert res["parsed_answer"] == "18"
        assert res["score"] == 1.0

    def test_parse_negative_and_decimal(self):
        sol = "<answer>-12.5</answer>"
        res = compute_score(
            data_source="openai/gsm8k", solution_str=sol, ground_truth="-12.5", extra_info=None, fallback_to_default=False
        )
        assert res["parsed_answer"] == "-12.5"
        assert res["score"] == 1.0

    def test_parse_with_commas(self):
        sol = "Result is <answer>1,234</answer>"
        res = compute_score(
            data_source="openai/gsm8k", solution_str=sol, ground_truth="1234", extra_info=None, fallback_to_default=False
        )
        assert res["parsed_answer"] == "1234"
        assert res["score"] == 1.0

    def test_fallback_when_no_tag(self):
        sol = "We finally get #### 7"
        res = compute_score(
            data_source="openai/gsm8k", solution_str=sol, ground_truth="7", extra_info=None, fallback_to_default=True
        )
        assert res["fallback"] is True
        assert res["score"] == 1.0

    def test_no_fallback_and_no_tag(self):
        sol = "No answer tag and no #### format"
        res = compute_score(
            data_source="openai/gsm8k", solution_str=sol, ground_truth="5", extra_info=None, fallback_to_default=False
        )
        assert res["used_answer_tag"] is False
        assert res["parsed_answer"] is None
        assert res["score"] == 0.0


if __name__ == "__main__":
    unittest.main()




