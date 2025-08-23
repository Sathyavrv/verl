import unittest

from verl.trainer.ppo.metric_utils import process_validation_metrics


class TestProcessValidationMetrics(unittest.TestCase):
    def test_ignore_none_and_strings(self):
        data_sources = ["src1", "src1", "src1", "src1"]
        prompts = ["p"] * 4
        infos = {
            "score": [1.0, None, 0.0, 2.0],
            "pred": ["A", "B", "A", "A"],
            "parsed_answer": ["18", None, "7", "2"],  # will be ignored as str in numeric branch
        }
        res = process_validation_metrics(data_sources, prompts, infos)
        # Only [1.0, 0.0, 2.0] are considered
        mean_key = next(iter(res["src1"]["score"]))
        # mean@3 should exist and equal (1 + 0 + 2)/3 = 1.0
        assert res["src1"]["score"]["mean@3"] == 1.0


if __name__ == "__main__":
    unittest.main()








