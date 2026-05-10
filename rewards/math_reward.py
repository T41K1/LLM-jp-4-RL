"""
Custom reward function for math GRPO training with verl.

Usage in training script:
    reward.custom_reward_function.path="rewards/math_reward.py" \
    reward.custom_reward_function.name="compute_score"

Uses MathVerifier from math_utils for answer extraction and comparison.
"""

from rewards.math_utils import MathVerifier

_verifier = MathVerifier()


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
        dict with "score" (1.0 or -1.0), "acc" (bool), "pred" (extracted answer)
    """
    result = _verifier([], solution_str, ground_truth)
    matched = result.score > 0
    # scoreに対して10倍をするのはOlmo3の設定値を踏襲
    score = result.score  # 1.0 or 0.0
    return {"score": score * 10.0, "acc": matched, "pred": result.pred or solution_str}
