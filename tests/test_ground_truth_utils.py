"""Tests for rewards.ground_truth_utils.MathVerifier.

Run with:
    uv run --frozen python -B -m unittest tests.test_ground_truth_utils
or via discovery:
    uv run --frozen python -B -m unittest discover -s tests -p 'test_*.py'
"""

import dataclasses
import unittest

from rewards.ground_truth_utils import (
    MathVerifier,
    VerificationResult,
    VerifierConfig,
    VerifierFunction,
)


class VerificationResultTest(unittest.TestCase):
    def test_default_fields(self):
        result = VerificationResult(score=1.0)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.cost, 0.0)
        self.assertIsNone(result.reasoning)

    def test_field_names(self):
        names = {f.name for f in dataclasses.fields(VerificationResult)}
        self.assertEqual(names, {"score", "cost", "reasoning"})


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
        # `last_boxed_only_string` only succeeds for the `\boxed ` (space) form,
        # which is the dominant Minerva-trained format.
        self.assertEqual(self._score(r"\boxed 42$", "42"), 1.0)

    def test_boxed_with_space_wrong_answer_scores_zero(self):
        self.assertEqual(self._score(r"\boxed 41$", "42"), 0.0)

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


if __name__ == "__main__":
    unittest.main()
