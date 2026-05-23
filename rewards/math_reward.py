"""
Custom reward function for math GRPO training with verl.

Usage in training script:
    reward.custom_reward_function.path="rewards/math_reward.py" \
    reward.custom_reward_function.name="compute_score"

Verifier 切り替え (環境変数 REWARD_VERIFIER):
    "math_verify" (default) : math_verify ベース (rewards/math_verify_verifier.py)
    "legacy"                : 旧 MathVerifier (rewards/ground_truth_utils.py)

参照: docs/reward-refactor.md / issue #13
"""

import os

from rewards.ground_truth_utils import MathVerifier
from rewards.math_verify_verifier import verify_answer

_legacy_verifier = MathVerifier()


_VERIFIER = os.environ.get("REWARD_VERIFIER", "math_verify").lower()


# 今後math以外のdomainに対してもreward functionを作成する


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Compute reward score for a math solution.

    Args:
        data_source: Dataset identifier (used for logging, not routing)
        solution_str: Model's generated response
        ground_truth: Expected answer string
        extra_info: Additional metadata (optional)

    Returns:
        dict with "score" (10.0 or 0.0), "acc" (bool), "pred" (extracted answer),
        "method" (どの判定方式で一致したか: math_verify / text / none / legacy)
    """
    if _VERIFIER == "legacy":
        result = _legacy_verifier([], solution_str, ground_truth)
        matched = result.score > 0
        # scoreに対して10倍をするのはOlmo3の設定値を踏襲
        return {
            "score": result.score * 10.0,
            "acc": matched,
            "pred": result.pred or solution_str,
            "method": "legacy",
        }

    res = verify_answer(solution_str, ground_truth)
    return {
        "score": (1.0 if res.ok else 0.0) * 10.0,
        "acc": res.ok,
        "pred": res.pred or solution_str,
        "method": res.method,
    }
