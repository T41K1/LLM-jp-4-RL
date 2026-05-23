"""
math_verify ベースの数学 reward verifier。

旧 `MathVerifier` (rewards/ground_truth_utils.py) を置き換える PoC。
HuggingFace math_verify (latex2sympy2_extended) を主エンジンにし、
記号・数値・分数・区間・集合・行列の等価判定を堅牢に行う。

設計方針 (reward 向け):
  0. 採点前に solution_str を末尾基準でクリップする (暴走生成/長文対策)。
     boxed は通常末尾付近に出るため、先頭ではなく末尾を残す。
  1. gold (ground_truth) は `$...$` で包んで LaTeX として parse する。
     裸の `\\sqrt{2}` や区間はそのままでは parse に失敗するため。
  2. prediction はモデル出力の最後の `\\boxed{}` から抽出する
     (学習プロンプトに boxed 指示を付与済みなので boxed が出る前提)。
  3. math_verify が扱わない「単語回答」(Median, Friday 等) は、
     gold が text 型のときのみ厳しめの文字列一致でフォールバック。
  4. boxed が抽出できなかった場合に限り、出力全文に math_verify をかけて
     式抽出で救済する (method="math_verify_fulltext")。boxed があるときは
     誤一致防止のため全文は見ない。
  5. どの method で一致したかを返し、false positive 監視に使う。

参照: docs/reward-refactor.md / issue #13
"""

import logging
import re
from dataclasses import dataclass

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from rewards.math_utils import last_boxed_only_string, remove_boxed

logger = logging.getLogger(__name__)

# 比較対象 (抽出後 boxed) の上限長。長すぎる候補は誤一致と遅延の原因になるため弾く。
MAX_LEN = 512

# 採点前に solution_str 全体を末尾基準でクリップする上限長。
# 暴走生成や極端に長い出力で parser が遅延/OOM するのを防ぐ。boxed や最終式は
# 末尾付近に出るため、先頭ではなく末尾 _SOLUTION_CLIP_CHARS 文字を残す。
_SOLUTION_CLIP_CHARS = 8192

# math_verify の parse()/verify() は内部で signal.SIGALRM ベースの timeout を使うが、
# signal はメインスレッドでしか登録できない。verl は reward (compute_score) を
# ワーカースレッドで呼ぶため、timeout を有効にすると毎回
# "ValueError: signal only works in main thread" で落ち、全件 0 点になる。
# math_verify 自体は十分速い (~12ms/件) ので signal timeout を無効化する。
_PARSE_TIMEOUT = None
_VERIFY_TIMEOUT = None

# gold parse 用の config (LaTeX 優先、式抽出もフォールバックで許可)
_GOLD_CFG = [LatexExtractionConfig(), ExprExtractionConfig()]

# gold が「単語回答」かどうかの判定。数字・LaTeX 記号を含まず英字主体ならテキスト扱い。
_TEXT_GT_RE = re.compile(r"^[A-Za-z][A-Za-z \-]*$")


@dataclass
class VerifyResult:
    ok: bool
    method: str  # "math_verify" | "text" | "math_verify_fulltext" | "none"
    pred: str | None = None


def _normalize_text(s: str) -> str:
    """単語回答用の軽い正規化: lowercase / trim / 連続空白を 1 つに / ダッシュ統一。"""
    s = s.strip().lower()
    s = re.sub(r"[‐-―−]", "-", s)  # 各種ハイフン/マイナスを '-' に
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_boxed_text(solution_str: str) -> str | None:
    """モデル出力から最後の \\boxed{...} の中身を生文字列で取り出す (text 比較用)。"""
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except AssertionError:
        return None


def _parse_latex_answer(answer: str):
    """裸の answer 文字列を math_verify 用の LaTeX として parse する。"""
    return parse(f"${answer}$", extraction_config=_GOLD_CFG, parsing_timeout=_PARSE_TIMEOUT)


def _parse_freeform(text: str):
    """モデル出力の生テキストから math_verify に式・数値を抽出させる。

    boxed が取れなかったときの救済用。`$...$` で包まず生のまま渡すことで、
    math_verify の抽出器が本文中の LaTeX/式/数値を拾えるようにする。
    """
    return parse(text, extraction_config=_GOLD_CFG, parsing_timeout=_PARSE_TIMEOUT)


def verify_answer(solution_str: str, ground_truth: str) -> VerifyResult:
    """solution_str (モデル出力) を ground_truth と照合する。

    Returns:
        VerifyResult(ok, method, pred)
    """
    gt = str(ground_truth).strip()

    # --- 0. 末尾クリップ (暴走生成/長文対策) ------------------------------
    # boxed や最終式は末尾付近に出るため、超過分は先頭側を捨てて末尾を残す。
    solution_str = str(solution_str)
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    boxed_text = _extract_boxed_text(solution_str)
    pred_candidate = boxed_text if boxed_text is not None and len(boxed_text) <= MAX_LEN else None

    # --- 1. text フォールバック (gold が単語回答型のときのみ) -------------
    # math_verify が単語を文字列として一致させる場合もあるが、単語回答は監視上
    # "text" として記録したいので math_verify より先に判定する。
    if _TEXT_GT_RE.match(gt):
        if pred_candidate is not None:
            if _normalize_text(pred_candidate) == _normalize_text(gt):
                return VerifyResult(True, "text", pred=pred_candidate)

    # gold を $...$ で包んで LaTeX として parse (裸の latex GT 対策)。
    try:
        gold = _parse_latex_answer(gt)
    except Exception:
        logger.warning("math_verify gold parse failed (gt=%r)", gt, exc_info=True)
        gold = []

    # --- 1. math_verify 主経路 (最終 boxed のみ採点) ---------------------
    if gold:
        try:
            # 出力全文には途中式や中間の boxed が含まれうるため、最終 boxed のみを採点する。
            target = _parse_latex_answer(pred_candidate) if pred_candidate is not None else []
        except Exception:
            logger.warning("math_verify target parse failed", exc_info=True)
            target = []
        if target:
            try:
                # verify(gold, target): gold が先。順序に注意。
                if verify(gold, target, timeout_seconds=_VERIFY_TIMEOUT):
                    return VerifyResult(True, "math_verify", pred=pred_candidate)
            except Exception:
                logger.warning("math_verify verify failed (gt=%r)", gt, exc_info=True)

    # --- 3. 全文 math_verify フォールバック (boxed 未検出時のみ) ----------
    # boxed があるのに全文を見ると途中式・本文中の数値に誤一致しやすいので、
    # boxed がそもそも取れなかったケースに限定して救済する。
    if gold and boxed_text is None:
        try:
            target = _parse_freeform(solution_str)
        except Exception:
            logger.warning("math_verify fulltext parse failed", exc_info=True)
            target = []
        if target:
            try:
                if verify(gold, target, timeout_seconds=_VERIFY_TIMEOUT):
                    return VerifyResult(True, "math_verify_fulltext", pred=None)
            except Exception:
                logger.warning("math_verify fulltext verify failed (gt=%r)", gt, exc_info=True)

    return VerifyResult(False, "none", pred=pred_candidate)
