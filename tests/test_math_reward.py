import unittest
from concurrent.futures import ThreadPoolExecutor

from rewards.math_reward import compute_score
from rewards.math_utils import (
    MathVerifier,
    StrictMathVerifier,
    extract_math_answers,
    extract_strict_math_answers,
    is_equiv,
    verify_math_prediction,
)

# uv run --frozen python -B -m unittest discover -s tests -p 'test_*.py' # run tests


class MathRewardTest(unittest.TestCase):
    def test_boxed_answer_scores_and_returns_extracted_pred(self):
        result = compute_score(
            data_source="math",
            solution_str="We simplify the expression and get \\boxed{42}.",
            ground_truth="42",
        )

        self.assertEqual(result["score"], 10.0)
        self.assertTrue(result["acc"])
        self.assertEqual(result["pred"], "42")

    def test_extract_math_answers_uses_open_instruct_order(self):
        self.assertEqual(
            extract_math_answers("Final Answer: The final answer is 100,000. "),
            ["100000"],
        )
        self.assertEqual(extract_math_answers("The value is $7$."), ["7"])

    def test_math_verifier_matches_helper(self):
        prediction = "The answer is \\boxed{\\frac{1}{2}}."

        helper_result = verify_math_prediction(prediction, "1/2")
        verifier_result = MathVerifier()([], prediction, "1/2")

        self.assertEqual(verifier_result, helper_result)
        self.assertEqual(verifier_result.pred, "\\frac{1}{2}")

    def test_strict_verifier_only_uses_minerva_then_fallback(self):
        prediction = "The answer is \\boxed{7}."

        self.assertEqual(extract_strict_math_answers(prediction), ["Theansweris7."])
        self.assertEqual(StrictMathVerifier()([], prediction, "7").score, 0.0)

    def test_sympy_equivalence_works_in_reward_worker_thread(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(is_equiv, "\\frac{1}{2}", "1/2").result()

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
