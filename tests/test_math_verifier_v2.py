import pytest

from rewards.math_verify_verifier import (
    _extract_boxed_text,
    _normalize_text,
    verify_answer,
)


def _plain_final(text: str) -> str:
    return "assistant final " + text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"The answer is \boxed{42}.", "42"),
        (r"First \boxed{1}, final \boxed{2}.", "2"),
        (r"No boxed answer here.", None),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    ],
)
def test_extract_boxed_text(text, expected):
    assert _extract_boxed_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" Median ", "median"),
        ("New   York", "new york"),
        ("twenty−one", "twenty-one"),
        ("twenty—one", "twenty-one"),
    ],
)
def test_normalize_text(text, expected):
    assert _normalize_text(text) == expected


@pytest.mark.parametrize(
    ("solution", "ground_truth", "expected_pred"),
    [
        (
            r"We compute it. Therefore \boxed{\frac{1}{2}}.",
            r"\frac{1}{2}",
            r"\frac{1}{2}",
        ),
        (r"Final answer: \boxed{2}.", "1+1", "2"),
        (r"Final answer: \boxed{\sqrt{2}}.", r"\sqrt{2}", r"\sqrt{2}"),
        (r"Final answer: \boxed{0}.", "0", "0"),
    ],
)
def test_math_verify_boxed_true(solution, ground_truth, expected_pred):
    result = verify_answer(_plain_final(solution), ground_truth)

    assert result.ok is True
    assert result.method == "math_verify"
    assert result.pred == expected_pred


@pytest.mark.parametrize(
    ("solution", "ground_truth"),
    [
        (r"Final answer: \boxed{3}.", "4"),
        (r"Final answer: \boxed{\frac{1}{3}}.", r"\frac{1}{2}"),
        (r"Final answer: \boxed{x+1}.", "x+2"),
    ],
)
def test_math_verify_boxed_false(solution, ground_truth):
    result = verify_answer(_plain_final(solution), ground_truth)

    assert result.ok is False
    assert result.method == "none"


@pytest.mark.parametrize(
    ("solution", "ground_truth", "expected_pred"),
    [
        (r"Final answer: \boxed{Median}.", "Median", "Median"),
        (r"Final answer: \boxed{friday}.", "Friday", "friday"),
        (r"Final answer: \boxed{twenty−one}.", "twenty-one", "twenty−one"),
    ],
)
def test_text_fallback_for_word_answers(solution, ground_truth, expected_pred):
    result = verify_answer(_plain_final(solution), ground_truth)

    assert result.ok is True
    assert result.method == "text"
    assert result.pred == expected_pred


def test_text_fallback_does_not_accept_math_gold():
    result = verify_answer(_plain_final(r"Final answer: \boxed{two}."), "2")

    assert result.ok is False
    assert result.method == "none"


def test_fulltext_fallback_only_when_no_boxed():
    result = verify_answer(_plain_final(r"After solving, the answer is $42$."), "42")

    assert result.ok is True
    assert result.method == "math_verify_fulltext"
    assert result.pred is None


def test_fulltext_fallback_not_used_when_boxed_exists_even_if_body_contains_gold():
    solution = r"The correct value might be $42$, but final answer is \boxed{41}."
    result = verify_answer(_plain_final(solution), "42")

    assert result.ok is False
    assert result.method == "none"
    assert result.pred == "41"


def test_last_boxed_is_used_not_first_boxed():
    solution = r"First I got \boxed{1}. After correction, final is \boxed{2}."
    result = verify_answer(_plain_final(solution), "2")

    assert result.ok is True
    assert result.method == "math_verify"
    assert result.pred == "2"


def test_too_long_boxed_candidate_is_rejected():
    very_long = "1" * 600
    result = verify_answer(_plain_final(rf"Final answer: \boxed{{{very_long}}}."), very_long)

    assert result.ok is False
    assert result.method == "none"
    assert result.pred is None


def test_solution_is_tail_clipped_and_final_boxed_survives():
    long_prefix = "noise " * 3000
    solution = long_prefix + r" final answer is \boxed{7}."
    result = verify_answer(_plain_final(solution), "7")

    assert result.ok is True
    assert result.method == "math_verify"
    assert result.pred == "7"


def test_missing_harmony_or_plain_final_marker_is_format_violation(monkeypatch):
    monkeypatch.setenv("MATH_VERIFY_POOL_ENABLED", "0")
    result = verify_answer(r"The answer is \boxed{42}.", "42")

    assert result.ok is False
    assert result.method == "format_violation"
    assert result.scored_channel == "final"


def test_plain_assistant_final_without_space_before_answer(monkeypatch):
    monkeypatch.setenv("MATH_VERIFY_POOL_ENABLED", "0")
    result = verify_answer(
        r"analysis A draft gets \boxed{41}. assistantfinalThe answer is \boxed{42}.",
        "42",
    )

    assert result.ok is True
    assert result.method == "math_verify"
    assert result.pred == "42"
    assert result.scored_channel == "final"
    assert result.analysis_ok is False


def test_harmony_parse_failure_is_not_raw_scored(monkeypatch):
    monkeypatch.setenv("MATH_VERIFY_POOL_ENABLED", "0")
    result = verify_answer(r"<|start|>not_assistant The answer is \boxed{42}.", "42")

    assert result.ok is False
    assert result.method == "harmony_parse_failed"
    assert result.scored_channel == "final"
    assert result.has_harmony is True


def test_harmony_final_channel_only_is_rewarded(monkeypatch):
    monkeypatch.setenv("MATH_VERIFY_POOL_ENABLED", "0")
    solution = (
        r"<|channel|>analysis<|message|>A draft branch gets \boxed{42}.<|end|>"
        r"<|start|>assistant<|channel|>final<|message|>The answer is \boxed{41}.<|return|>"
    )
    result = verify_answer(solution, "42")

    assert result.ok is False
    assert result.method == "none"
    assert result.scored_channel == "final"
    assert result.has_harmony is True
    assert result.has_harmony_final is True
    assert result.analysis_ok is True


def test_harmony_missing_final_is_not_rewarded(monkeypatch):
    monkeypatch.setenv("MATH_VERIFY_POOL_ENABLED", "0")
    solution = r"<|channel|>analysis<|message|>A draft branch gets \boxed{42}.<|end|>"
    result = verify_answer(solution, "42")

    assert result.ok is False
    assert result.method == "no_final"
    assert result.scored_channel == "final"
    assert result.has_harmony is True
    assert result.has_harmony_final is False
    assert result.analysis_ok is True


def test_math_reward_prefers_special_token_response_from_extra_info(monkeypatch):
    monkeypatch.setenv("MATH_VERIFY_POOL_ENABLED", "0")
    from rewards.math_reward import compute_score

    raw_without_special_tokens = r"analysis A draft branch gets \boxed{42}. assistant final The answer is \boxed{41}."
    raw_with_special_tokens = (
        r"<|channel|>analysis<|message|>A draft branch gets \boxed{42}.<|end|>"
        r"<|start|>assistant<|channel|>final<|message|>The answer is \boxed{41}.<|return|>"
    )
    result = compute_score(
        data_source="test",
        solution_str=raw_without_special_tokens,
        ground_truth="42",
        extra_info={"response_str_with_special_tokens": raw_with_special_tokens},
    )

    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["method"] == "none"
    assert result["scored_channel"] == "final"
    assert result["analysis_ok"] is True
