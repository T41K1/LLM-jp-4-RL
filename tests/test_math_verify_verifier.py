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

    def test_no_boxed_is_not_matched(self):
        # boxed が無く正解も含まれない出力は False
        res = verify_answer("I am not sure about this problem.", "42")
        self.assertFalse(res.ok)
        self.assertEqual(res.method, "none")


if __name__ == "__main__":
    unittest.main()
