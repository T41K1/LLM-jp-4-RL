"""Tests for rewards.ground_truth_utils.MathVerifier.

Run with:
    uv run --frozen python -B -m unittest tests.test_ground_truth_utils
or via discovery:
    uv run --frozen python -B -m unittest discover -s tests -p 'test_*.py'
"""

import dataclasses
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from rewards.ground_truth_utils import (
    MathVerifier,
    VerificationResult,
    VerifierConfig,
    VerifierFunction,
)


# Module-level so it is picklable for `forkserver` start method.
def _hang_worker(x1, x2, result_queue):
    """Simulated worker that blocks far past any reasonable test timeout."""
    time.sleep(60)


class VerificationResultTest(unittest.TestCase):
    def test_default_fields(self):
        result = VerificationResult(score=1.0)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.cost, 0.0)
        self.assertIsNone(result.reasoning)
        self.assertIsNone(result.pred)

    def test_field_names(self):
        names = {f.name for f in dataclasses.fields(VerificationResult)}
        self.assertEqual(names, {"score", "cost", "reasoning", "pred"})


class VerifierConfigTest(unittest.TestCase):
    def test_from_args_ignores_non_matching_fields(self):
        @dataclasses.dataclass
        class Source:
            unrelated: int = 7

        config = VerifierConfig.from_args(Source())
        self.assertIsInstance(config, VerifierConfig)

    def test_from_args_accepts_none_sources(self):
        config = VerifierConfig.from_args(None, None)
        self.assertIsInstance(config, VerifierConfig)


class MathVerifierMetadataTest(unittest.TestCase):
    def test_name_and_weight_defaults(self):
        verifier = MathVerifier()
        self.assertEqual(verifier.name, "math")
        self.assertEqual(verifier.weight, 1.0)

    def test_repr_includes_name_and_weight(self):
        self.assertEqual(repr(MathVerifier()), "MathVerifier(name=math, weight=1.0)")

    def test_get_config_class(self):
        self.assertIs(VerifierFunction.get_config_class(), VerifierConfig)
        self.assertIs(MathVerifier.get_config_class(), VerifierConfig)


class MathVerifierScoringTest(unittest.TestCase):
    """Exercise each extraction path in MathVerifier.__call__."""

    def setUp(self):
        self.verifier = MathVerifier()

    def _score(self, prediction: str, label: str) -> float:
        return self.verifier([], prediction, label).score

    # --- boxed extraction path -------------------------------------------------
    def test_boxed_with_space_extracts_integer(self):
        self.assertEqual(self._score(r"\boxed 42$", "42"), 1.0)

    def test_boxed_with_space_wrong_answer_scores_zero(self):
        self.assertEqual(self._score(r"\boxed 41$", "42"), 0.0)

    def test_boxed_brace_form_extracts_integer(self):
        # Regression: last_boxed_only_string previously fell through and
        # returned None for `\boxed{...}`, so this typical training output
        # was silently scored 0.0 even when correct.
        self.assertEqual(self._score(r"The answer is \boxed{42}.", "42"), 1.0)

    def test_boxed_brace_form_handles_nested_braces(self):
        self.assertEqual(
            self._score(r"So the answer is \boxed{\frac{1}{2}}.", "1/2"),
            1.0,
        )

    # --- Minerva "Final Answer: ... I hope it is correct." path ---------------
    def test_minerva_format_correct(self):
        prediction = "Final Answer: The final answer is 42. I hope it is correct."
        self.assertEqual(self._score(prediction, "42"), 1.0)

    def test_minerva_format_wrong(self):
        prediction = "Final Answer: The final answer is 42. I hope it is correct."
        self.assertEqual(self._score(prediction, "99"), 0.0)

    def test_minerva_format_normalizes_comma_separated_int(self):
        prediction = "Final Answer: The final answer is 100,000. I hope it is correct."
        self.assertEqual(self._score(prediction, "100000"), 1.0)

    def test_minerva_format_strips_units(self):
        prediction = "Final Answer: The final answer is 5 dollars. I hope it is correct."
        self.assertEqual(self._score(prediction, "5"), 1.0)

    def test_minerva_format_fraction_matches_via_sympy(self):
        prediction = (
            r"Final Answer: The final answer is \frac{1}{2}. I hope it is correct."
        )
        self.assertEqual(self._score(prediction, "1/2"), 1.0)

    def test_minerva_format_different_fraction_scores_zero(self):
        prediction = (
            r"Final Answer: The final answer is \frac{1}{3}. I hope it is correct."
        )
        self.assertEqual(self._score(prediction, "1/2"), 0.0)

    # --- last-LaTeX-between-dollars fallback ---------------------------------
    def test_latex_between_dollar_signs(self):
        # When boxed + Minerva both miss, the verifier looks at the substring
        # between the last two `$` characters.
        self.assertEqual(self._score("The value is $7$.", "7"), 1.0)


class MathVerifierReturnTypeTest(unittest.TestCase):
    def test_returns_verification_result_instance(self):
        result = MathVerifier()([], r"\boxed 1$", "1")
        self.assertIsInstance(result, VerificationResult)
        self.assertEqual(result.cost, 0.0)
        self.assertIsNone(result.reasoning)


class MathVerifierRobustnessTest(unittest.TestCase):
    """Inputs / call-sites that previously crashed the verifier."""

    def test_unparseable_prediction_does_not_raise(self):
        # Regression: math_utils.is_equiv used to reference an undefined
        # `eval_logger` in its exception path, turning a parse failure into a
        # NameError that bubbled out of MathVerifier.__call__.
        result = MathVerifier()([], "completely unparseable %%%", "42")
        self.assertEqual(result.score, 0.0)

    def test_subprocess_kills_hanging_worker_within_timeout(self):
        # Regression: previous run had a single compute_score call hang for
        # 2326s because SIGALRM cannot interrupt sympy's C extensions. The
        # subprocess-based timeout in is_equiv must hard-kill the worker.
        from rewards import math_utils

        original = math_utils._is_equiv_worker
        math_utils._is_equiv_worker = _hang_worker
        try:
            start = time.monotonic()
            result = math_utils.is_equiv("1", "1", timeout_sec=1)
            elapsed = time.monotonic() - start
        finally:
            math_utils._is_equiv_worker = original

        self.assertFalse(result)
        # 1s timeout + up to 1s for SIGTERM grace + slack for forkserver/teardown.
        self.assertLess(elapsed, 5.0)

    def test_long_input_returns_fast_without_sympy(self):
        # Regression: a previous training run had a single compute_score call
        # take ~2326s because the fallback path passed the full prediction
        # (thousands of tokens) to sympy.simplify, which can hang in C code
        # even with SIGALRM set. The length cap in is_equiv must short-circuit
        # before sympy is invoked.
        import time

        from rewards.math_utils import IS_EQUIV_MAX_LEN, is_equiv

        huge = "\\frac{1}{2}" * (IS_EQUIV_MAX_LEN // 2)
        self.assertGreater(len(huge), IS_EQUIV_MAX_LEN)
        start = time.monotonic()
        self.assertFalse(is_equiv(huge, "1/2"))
        self.assertLess(time.monotonic() - start, 0.5)

    def test_verifier_runs_in_worker_thread(self):
        # Regression: math_utils.timeout used signal.SIGALRM unconditionally,
        # which raises ValueError outside the main thread. verl reward workers
        # can call us off-thread, so the verifier must stay alive there.
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                MathVerifier(), [], r"The answer is \boxed{42}.", "42"
            ).result()
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.pred, "42")


class MathVerifierPredTest(unittest.TestCase):
    """`pred` should expose the extracted answer used for scoring."""

    def setUp(self):
        self.verifier = MathVerifier()

    def test_pred_on_match_is_extracted_boxed_value(self):
        result = self.verifier([], r"\boxed 42$", "42")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.pred, "42")

    def test_pred_on_minerva_match_is_extracted_value(self):
        prediction = "Final Answer: The final answer is 42. I hope it is correct."
        result = self.verifier([], prediction, "42")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.pred, "42")

    def test_pred_on_mismatch_is_first_candidate(self):
        prediction = "Final Answer: The final answer is 41. I hope it is correct."
        result = self.verifier([], prediction, "42")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.pred, "41")


class MathRewardComputeScoreTest(unittest.TestCase):
    """End-to-end check for the entry point verl calls during training."""

    def test_compute_score_match(self):
        from rewards.math_reward import compute_score

        result = compute_score(
            data_source="math",
            solution_str=r"\boxed 42$",
            ground_truth="42",
        )
        self.assertEqual(result["score"], 10.0)
        self.assertTrue(result["acc"])
        self.assertEqual(result["pred"], "42")

    def test_compute_score_mismatch(self):
        from rewards.math_reward import compute_score

        result = compute_score(
            data_source="math",
            solution_str=r"\boxed{41}",
            ground_truth="42",
        )
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["acc"])
        self.assertEqual(result["pred"], "41")


if __name__ == "__main__":
    unittest.main()
