"""Tests for rewards.math_verify_verifier.verify_answer.

docs/reward-refactor.md の「通すべき / 通してはいけない」ケースを検証する。

Run with:
    uv run --frozen python -B -m unittest tests.test_math_verify_verifier
or:
    .venv/bin/python -m unittest tests.test_math_verify_verifier
"""

import unittest

from rewards.math_verify_verifier import verify_answer


def _boxed(ans: str) -> str:
    """モデル出力想定: \\boxed{} で囲んだフル出力。"""
    return r"Therefore the final answer is \boxed{" + ans + "}."


# (model_answer, ground_truth) — model_answer は \boxed{} に入れて渡す
PASS_CASES = [
    (r"\sqrt2", r"\sqrt{2}"),               # sqrt 表記ゆれ
    ("0.5", r"\frac{1}{2}"),                # 小数 vs 分数
    (r"\frac{1}{2}", "1/2"),               # 分数表記ゆれ
    ("2+3", "5"),                            # 式の simplify
    ("x^2-y^2=12", r"x^{2}-y^{2}=12"),     # 方程式の表記ゆれ
    (
        r"(-\infty,-4)\cup(-4,-2)\cup(-2,\infty)",
        r"(-\infty,-4)\cup(-4,-2)\cup(-2,\infty)",
    ),                                       # 区間 (∞/∪)
    ("33", "33"),                            # 整数 (AIME 型)
    ("70", "70"),
    ("mystery", "Mystery"),                  # 単語回答 (大文字小文字)
    ("Friday", "Friday"),
]

# 通してはいけない (false positive 禁止)
FAIL_CASES = [
    ("15", "5"),                             # substring 不一致
    ("101", "10"),
    ("3", r"\frac{3\sqrt{2}}{2}"),           # 値が違う
    ("25", "17"),                            # 別の整数
]


class VerifyAnswerPassTest(unittest.TestCase):
    def test_should_pass(self):
        for ans, gt in PASS_CASES:
            with self.subTest(ans=ans, gt=gt):
                res = verify_answer(_boxed(ans), gt)
                self.assertTrue(
                    res.ok,
                    f"期待 True だが False: pred={ans!r} gt={gt!r} method={res.method}",
                )


class VerifyAnswerFailTest(unittest.TestCase):
    def test_should_fail(self):
        for ans, gt in FAIL_CASES:
            with self.subTest(ans=ans, gt=gt):
                res = verify_answer(_boxed(ans), gt)
                self.assertFalse(
                    res.ok,
                    f"期待 False だが True (false positive): "
                    f"pred={ans!r} gt={gt!r} method={res.method}",
                )


class VerifyResultShapeTest(unittest.TestCase):
    def test_method_field(self):
        res = verify_answer(_boxed("42"), "42")
        self.assertTrue(res.ok)
        self.assertIn(res.method, {"math_verify", "text"})

    def test_uses_last_boxed_answer_only(self):
        response = r"One failed branch gives \boxed{41}. Therefore the final answer is \boxed{42}."
        res = verify_answer(response, "42")
        self.assertTrue(res.ok)
        self.assertEqual(res.pred, "42")

    def test_ignores_correct_unboxed_intermediate_answer(self):
        response = r"The value 42 appears in the derivation, but the final answer is \boxed{41}."
        res = verify_answer(response, "42")
        self.assertFalse(res.ok)
        self.assertEqual(res.method, "none")

    def test_no_boxed_is_not_matched(self):
        # boxed が無く正解も含まれない出力は False
        res = verify_answer("I am not sure about this problem.", "42")
        self.assertFalse(res.ok)
        self.assertEqual(res.method, "none")


class FulltextFallbackTest(unittest.TestCase):
    """boxed 未検出時の全文 math_verify フォールバック (method=math_verify_fulltext)。"""

    def test_unboxed_final_answer_is_rescued(self):
        # boxed は無いが、文中に最終式が出ているケースは全文フォールバックで拾う。
        response = "After simplifying everything, the final answer is 42."
        res = verify_answer(response, "42")
        self.assertTrue(res.ok)
        self.assertEqual(res.method, "math_verify_fulltext")

    def test_fulltext_not_used_when_boxed_present(self):
        # boxed があるときは全文を見ない: 本文中の正解 42 では救済されない。
        response = r"The value 42 appears here, but the final answer is \boxed{41}."
        res = verify_answer(response, "42")
        self.assertFalse(res.ok)
        self.assertEqual(res.method, "none")


class SolutionClipTest(unittest.TestCase):
    """末尾クリップ: boxed が末尾にあれば長大な前置きがあっても採点できる。"""

    def test_long_prefix_then_boxed_answer(self):
        response = "padding " * 5000 + r"Therefore the final answer is \boxed{42}."
        res = verify_answer(response, "42")
        self.assertTrue(res.ok)
        self.assertEqual(res.pred, "42")


if __name__ == "__main__":
    unittest.main()
