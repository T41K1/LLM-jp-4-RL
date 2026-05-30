"""
math_verify ベースの数学 reward verifier。

設計方針 (reward 向け):
  0. 採点前に solution_str を末尾基準でクリップする (暴走生成/長文対策)。
  1. OpenAI Harmony special token が残っている場合は、assistant の final
     channel だけを reward 判定に使う。
  2. analysis channel にだけ正答がある場合は reward せず、diagnostic として
     analysis_ok / analysis_method に記録する。
  3. final 内に boxed がある場合は最後の `\\boxed{}` のみを採点する。
  4. final 内に boxed が無い場合は、final text に限って math_verify の
     freeform 抽出で救済する (method="math_verify_fulltext")。

参照: docs/reward-refactor.md / issue #13
"""

import atexit
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import re
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from rewards.math_utils import last_boxed_only_string, remove_boxed

logger = logging.getLogger(__name__)

# 比較対象 (抽出後 boxed) の上限長。長すぎる候補は誤一致と遅延の原因になるため弾く。
MAX_LEN = 512

# 採点前に solution_str 構造抽出後の採点対象を末尾基準でクリップする上限長。
# 暴走生成や極端に長い出力で parser が遅延/OOM するのを防ぐ。boxed や最終式は
# 末尾付近に出るため、先頭ではなく末尾 _SOLUTION_CLIP_CHARS 文字を残す。
_SOLUTION_CLIP_CHARS = 8192


_PARSE_TIMEOUT = None
_VERIFY_TIMEOUT = None

# gold parse 用の config (LaTeX 優先、式抽出もフォールバックで許可)
_GOLD_CFG = [LatexExtractionConfig(), ExprExtractionConfig()]

# gold が「単語回答」かどうかの判定。数字・LaTeX 記号を含まず英字主体ならテキスト扱い。
_TEXT_GT_RE = re.compile(r"^[A-Za-z][A-Za-z \-]*$")


# verifierの判定結果を
@dataclass
class VerifyResult:
    ok: bool
    method: str  # math_verify | text | math_verify_fulltext | none | no_final | harmony_parse_failed | format_violation | timeout/error states
    pred: str | None = None
    scored_channel: str = "raw"  # "raw" | "final" |
    has_harmony: bool = False
    has_harmony_final: bool = False
    analysis_ok: bool = False
    analysis_method: str = "none"
    boxed_found: bool = False
    boxed_missing: bool = False
    boxed_too_long: bool = False
    boxed_parse_ok: bool = False
    boxed_match: bool = False
    fulltext_fallback_used: bool = False
    fulltext_fallback_match: bool = False
    gold_parse_ok: bool = False


@dataclass(frozen=True)
class _HarmonyTextMessage:
    end: str
    role: str | None = None
    channel: str | None = None
    constrain: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class _HarmonyScope:
    final_text: str | None
    analysis_text: str | None
    has_final: bool


# process child processの管理
@dataclass
class _WorkerHandle:
    index: int
    process: mp.Process
    request_queue: mp.Queue
    response_queue: mp.Queue
    busy: bool = False
    requests_completed: int = 0
    restarts: int = 0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, os.environ.get(name), default)
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("Invalid %s=%r; using %.2f", name, os.environ.get(name), default)
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _math_verify_worker_loop(
    request_queue: mp.Queue, response_queue: mp.Queue, ready_queue: mp.Queue
) -> None:
    """Run math_verify work in an isolated process.

    The worker calls `_verify_answer_impl()` directly. Calling public
    `verify_answer()` here would recursively enter the process pool.
    """
    ready_queue.put(True)
    while True:
        request = request_queue.get()
        if request is None:
            return

        try:
            result = _verify_answer_impl(
                request["solution_str"], request["ground_truth"]
            )
            response_queue.put(("ok", result))
        except BaseException as exc:  # noqa: BLE001
            response_queue.put(
                (
                    "err",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )


class _MathVerifyWorkerPool:
    """Small local worker pool with hard process timeout.

    This mirrors the 0537 reward_score server design at the verifier boundary:
    keep math_verify in persistent child processes, and replace a child when it
    times out or errors. The surrounding Ray reward worker stays alive.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        timeout_sec: float,
        acquire_timeout_sec: float,
        startup_timeout_sec: float,
        max_requests_per_worker: int | None = None,
        start_method: str = "forkserver",
        worker_target: Callable[
            [mp.Queue, mp.Queue, mp.Queue], None
        ] = _math_verify_worker_loop,
    ) -> None:
        self._max_workers = max_workers
        self._timeout_sec = timeout_sec
        self._acquire_timeout_sec = acquire_timeout_sec
        self._startup_timeout_sec = startup_timeout_sec
        self._max_requests_per_worker = max_requests_per_worker
        self._worker_target = worker_target
        self._start_method = start_method
        self._mp_context = self._get_mp_context(start_method)
        self._lock = threading.Lock()
        self._workers: list[_WorkerHandle] = []
        self._restart_counts = {
            "timeout": 0,
            "error": 0,
            "max_requests": 0,
            "dead_process": 0,
        }

    @classmethod
    def from_env(cls) -> "_MathVerifyWorkerPool":
        max_requests = _env_int("MATH_VERIFY_MAX_REQUESTS_PER_WORKER", 0)
        return cls(
            max_workers=_env_int("MATH_VERIFY_WORKERS", 2, minimum=1),
            timeout_sec=_env_float("MATH_VERIFY_TIMEOUT_SEC", 10.0, minimum=0.001),
            acquire_timeout_sec=_env_float(
                "MATH_VERIFY_ACQUIRE_TIMEOUT_SEC", 30.0, minimum=0.001
            ),
            startup_timeout_sec=_env_float(
                "MATH_VERIFY_STARTUP_TIMEOUT_SEC", 30.0, minimum=0.001
            ),
            max_requests_per_worker=max_requests if max_requests > 0 else None,
            start_method=os.environ.get("MATH_VERIFY_MP_START_METHOD", "forkserver"),
        )

    def start(self) -> None:
        with self._lock:
            if self._workers:
                return
            last_exc: BaseException | None = None
            for start_method in self._candidate_start_methods(self._start_method):
                self._mp_context = self._get_mp_context(start_method)
                self._start_method = start_method
                workers: list[_WorkerHandle] = []
                try:
                    for index in range(self._max_workers):
                        workers.append(self._create_worker(index))
                    self._workers = workers
                    return
                except BaseException as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "Failed to start math_verify worker pool with %s: %s",
                        start_method,
                        exc,
                    )
                    for worker in workers:
                        self._shutdown_worker(worker)
                    self._workers = []
            assert last_exc is not None
            raise last_exc

    def close(self) -> None:
        with self._lock:
            workers = self._workers
            self._workers = []

        for worker in workers:
            self._shutdown_worker(worker)

    def compute(self, solution_str: str, ground_truth: str) -> VerifyResult:
        self.start()
        worker = self._acquire_worker()
        if worker is None:
            logger.warning(
                "math_verify worker pool busy for %.2fs; returning server_busy",
                self._acquire_timeout_sec,
            )
            return VerifyResult(False, "server_busy", pred=None)

        restart_reason = None
        try:
            worker.request_queue.put(
                {"solution_str": solution_str, "ground_truth": ground_truth},
                timeout=0.1,
            )
            status, value = worker.response_queue.get(timeout=self._timeout_sec)
            worker.requests_completed += 1
            if status == "ok":
                restart_reason = self._get_recycle_reason(worker)
                return value

            logger.warning("math_verify worker error: %s", value)
            restart_reason = "error"
            return VerifyResult(False, "worker_error", pred=None)
        except queue_mod.Empty:
            logger.warning(
                "math_verify worker timed out after %.2fs", self._timeout_sec
            )
            restart_reason = "timeout"
            return VerifyResult(False, "timeout", pred=None)
        except Exception as exc:
            logger.warning("math_verify worker pool failed: %s", exc, exc_info=True)
            restart_reason = "error"
            return VerifyResult(False, "worker_error", pred=None)
        finally:
            if restart_reason is None:
                self._release_worker(worker)
            else:
                self._restart_worker(worker, restart_reason)

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "worker_count": len(self._workers),
                "worker_restart_count": sum(self._restart_counts.values()),
                **{
                    f"worker_restart_{k}_count": v
                    for k, v in self._restart_counts.items()
                },
            }

    def _create_worker(self, index: int) -> _WorkerHandle:
        request_queue = self._mp_context.Queue(maxsize=1)
        response_queue = self._mp_context.Queue(maxsize=1)
        ready_queue = self._mp_context.Queue(maxsize=1)
        process = self._mp_context.Process(
            target=self._worker_target,
            args=(request_queue, response_queue, ready_queue),
            daemon=True,
            name=f"math-verify-worker-{index}",
        )
        success = False
        try:
            process.start()
            ready_queue.get(timeout=self._startup_timeout_sec)
            success = True
        except queue_mod.Empty as exc:
            self._terminate_process(process)
            raise RuntimeError(
                f"math_verify worker {index} failed to start within {self._startup_timeout_sec:.2f}s"
            ) from exc
        except BaseException:
            self._terminate_process(process)
            raise
        finally:
            ready_queue.close()
            if not success:
                request_queue.close()
                response_queue.close()

        return _WorkerHandle(
            index=index,
            process=process,
            request_queue=request_queue,
            response_queue=response_queue,
        )

    def _shutdown_worker(self, worker: _WorkerHandle) -> None:
        if worker.process.is_alive():
            try:
                worker.request_queue.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass
            worker.process.join(timeout=0.5)
            self._terminate_process(worker.process)

        worker.request_queue.close()
        worker.response_queue.close()

    def _terminate_process(self, process: mp.Process) -> None:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)

    def _acquire_worker(self) -> _WorkerHandle | None:
        deadline = time.monotonic() + self._acquire_timeout_sec
        while True:
            with self._lock:
                for worker in list(self._workers):
                    if not worker.process.is_alive():
                        worker = self._replace_worker_locked(worker, "dead_process")
                    if worker.busy:
                        continue
                    worker.busy = True
                    return worker

            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)

    def _release_worker(self, worker: _WorkerHandle) -> None:
        with self._lock:
            if worker.index >= len(self._workers):
                return
            if self._workers[worker.index] is not worker:
                return
            worker.busy = False

    def _restart_worker(self, worker: _WorkerHandle, reason: str) -> None:
        with self._lock:
            if worker.index >= len(self._workers):
                return
            if self._workers[worker.index] is not worker:
                return
            self._replace_worker_locked(worker, reason)

    def _replace_worker_locked(
        self, worker: _WorkerHandle, reason: str
    ) -> _WorkerHandle:
        self._shutdown_worker(worker)
        new_worker = self._create_worker(worker.index)
        new_worker.restarts = worker.restarts + 1
        self._workers[worker.index] = new_worker
        self._restart_counts[reason] = self._restart_counts.get(reason, 0) + 1
        return new_worker

    def _get_recycle_reason(self, worker: _WorkerHandle) -> str | None:
        if self._max_requests_per_worker is None:
            return None
        if worker.requests_completed >= self._max_requests_per_worker:
            return "max_requests"
        return None

    def _candidate_start_methods(self, preferred: str) -> list[str]:
        available = set(mp.get_all_start_methods())
        candidates = [preferred]
        if preferred == "forkserver":
            candidates.append("fork")
        elif preferred == "spawn":
            candidates.append("fork")
        elif preferred not in available:
            candidates.extend(["forkserver", "fork"])

        deduped = []
        for method in candidates:
            if method in available and method not in deduped:
                deduped.append(method)
        return deduped or [mp.get_start_method(allow_none=True) or "fork"]

    def _get_mp_context(self, start_method: str):
        try:
            return mp.get_context(start_method)
        except ValueError:
            logger.warning(
                "Unsupported multiprocessing start method %r; falling back to default",
                start_method,
            )
            return mp.get_context()


_POOL_LOCK = threading.Lock()
_POOL: _MathVerifyWorkerPool | None = None


def _get_pool() -> _MathVerifyWorkerPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = _MathVerifyWorkerPool.from_env()
            atexit.register(_POOL.close)
        return _POOL


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
    return parse(
        f"${answer}$", extraction_config=_GOLD_CFG, parsing_timeout=_PARSE_TIMEOUT
    )


def _parse_freeform(text: str):
    """モデル出力の生テキストから math_verify に式・数値を抽出させる。

    boxed が取れなかったときの救済用。`$...$` で包まず生のまま渡すことで、
    math_verify の抽出器が本文中の LaTeX/式/数値を拾えるようにする。
    """
    return parse(text, extraction_config=_GOLD_CFG, parsing_timeout=_PARSE_TIMEOUT)


_HARMONY_TOKEN_RE = re.compile(
    r"(<\|start\|>|<\|channel\|>|<\|constrain\|>|<\|message\|>|<\|end\|>|<\|return\|>|<\|call\|>)"
)
_HARMONY_BEGIN_MAP = {
    "<|start|>": "role",
    "<|channel|>": "channel",
    "<|constrain|>": "constrain",
    "<|message|>": "content",
}
_HARMONY_END_TOKENS = {"<|end|>", "<|return|>", "<|call|>"}
_ASSISTANT_FINAL_RE = re.compile(r"\bassistant\s*final\b|\bassistantfinal", re.I)


def _iter_harmony_text_messages(text: str):
    """OpenAI Harmony special token を含む decoded text から message を切り出す。

    `llm-jp-4-8b-thinking/llmjp4_harmony.py` の token-id lexer と同じ
    section 遷移を decoded string 上で行う。reward manager からは response 部分
    だけが渡るため、先頭 message は role を持たず `<|channel|>` から始まる場合がある。
    """
    message_dict: dict[str, str] = {}
    section: str | None = None
    text_parts: list[str] = []

    for part in _HARMONY_TOKEN_RE.split(text):
        if not part:
            continue

        if part in _HARMONY_BEGIN_MAP:
            if section is not None:
                message_dict[section] = "".join(text_parts)
            section = _HARMONY_BEGIN_MAP[part]
            text_parts = []
            continue

        if part in _HARMONY_END_TOKENS:
            if section is not None:
                message_dict[section] = "".join(text_parts)
            if message_dict:
                yield _HarmonyTextMessage(**message_dict, end=part)
            message_dict = {}
            section = None
            text_parts = []
            continue

        if section is not None:
            text_parts.append(part)

    if section is not None:
        message_dict[section] = "".join(text_parts)
        if message_dict:
            yield _HarmonyTextMessage(**message_dict, end="incomplete")


def _harmony_channel_name(channel: str | None) -> str | None:
    if channel is None:
        return None
    channel = channel.strip()
    if not channel:
        return None
    return channel.split()[0]


def _is_assistant_harmony_message(message: _HarmonyTextMessage) -> bool:
    # response_ids のみを parse した場合、generation prompt の `<|start|>assistant`
    # は prompt 側にあり、生成部分は `<|channel|>` から始まる。
    if message.role is None:
        return True
    role = message.role.strip()
    if not role:
        return False
    return role.split()[0] == "assistant"


def _extract_harmony_scope(solution_str: str) -> _HarmonyScope | None:
    messages = [
        m
        for m in _iter_harmony_text_messages(solution_str)
        if _is_assistant_harmony_message(m)
    ]
    if not messages:
        return None

    final_texts = [
        m.content or ""
        for m in messages
        if _harmony_channel_name(m.channel) == "final" and m.content is not None
    ]
    analysis_texts = [
        m.content or ""
        for m in messages
        if _harmony_channel_name(m.channel) == "analysis" and m.content
    ]

    return _HarmonyScope(
        final_text=final_texts[-1] if final_texts else None,
        analysis_text="\n".join(analysis_texts) if analysis_texts else None,
        has_final=bool(final_texts),
    )


def _extract_plain_assistant_final_scope(solution_str: str) -> _HarmonyScope | None:
    """Special token が消えた degraded Harmony text から final 範囲を切る。"""
    matches = list(_ASSISTANT_FINAL_RE.finditer(solution_str))
    if not matches:
        return None

    match = matches[-1]
    analysis_text = solution_str[: match.start()]
    return _HarmonyScope(
        final_text=solution_str[match.end() :],
        analysis_text=analysis_text if analysis_text else None,
        has_final=True,
    )


def _verify_answer_text_impl(solution_str: str, ground_truth: str) -> VerifyResult:
    """1 つの採点対象テキストを ground_truth と照合する。

    Returns:
        VerifyResult containing the reward decision, match method, and extracted prediction.
    """
    gt = str(ground_truth).strip()

    # --- 0. 末尾クリップ (暴走生成/長文対策) ------------------------------
    # boxed や最終式は末尾付近に出るため、超過分は先頭側を捨てて末尾を残す。
    solution_str = str(solution_str)
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    boxed_text = _extract_boxed_text(solution_str)
    boxed_found = boxed_text is not None
    boxed_too_long = boxed_text is not None and len(boxed_text) > MAX_LEN
    pred_candidate = (
        boxed_text if boxed_text is not None and len(boxed_text) <= MAX_LEN else None
    )
    result_base = {
        "boxed_found": boxed_found,
        "boxed_missing": not boxed_found,
        "boxed_too_long": boxed_too_long,
    }

    # --- text フォールバック (gold が単語回答型のときのみ) -------------
    # math_verify が単語を文字列として一致させる場合もあるが、単語回答は監視上
    # "text" として記録したいので math_verify より先に判定する。
    if _TEXT_GT_RE.match(gt):
        if pred_candidate is not None:
            if _normalize_text(pred_candidate) == _normalize_text(gt):
                return VerifyResult(
                    True,
                    "text",
                    pred=pred_candidate,
                    boxed_match=True,
                    **result_base,
                )

    # gold を $...$ で包んで LaTeX として parse (裸の latex GT 対策)。
    try:
        gold = _parse_latex_answer(gt)
    except Exception:
        logger.warning("math_verify gold parse failed (gt=%r)", gt, exc_info=True)
        gold = []
    result_base["gold_parse_ok"] = bool(gold)

    # --- 1. math_verify 主経路 (最終 boxed のみ採点) ---------------------
    if gold:
        boxed_parse_ok = False
        try:
            # 出力全文には途中式や中間の boxed が含まれうるため、最終 boxed のみを採点する。
            target = (
                _parse_latex_answer(pred_candidate)
                if pred_candidate is not None
                else []
            )
            boxed_parse_ok = bool(target)
        except Exception:
            logger.warning("math_verify target parse failed", exc_info=True)
            target = []
        result_base["boxed_parse_ok"] = boxed_parse_ok
        if target:
            try:
                # verify(gold, target): gold が先。順序に注意。
                if verify(gold, target, timeout_seconds=_VERIFY_TIMEOUT):
                    return VerifyResult(
                        True,
                        "math_verify",
                        pred=pred_candidate,
                        boxed_match=True,
                        **result_base,
                    )
            except Exception:
                logger.warning("math_verify verify failed (gt=%r)", gt, exc_info=True)

    # --- 2. 全文 math_verify フォールバック (boxed 未検出時のみ) ----------
    # boxed があるのに全文を見ると途中式・本文中の数値に誤一致しやすいので、
    # boxed がそもそも取れなかったケースに限定して救済する。
    if gold and boxed_text is None:
        result_base["fulltext_fallback_used"] = True
        try:
            target = _parse_freeform(solution_str)
        except Exception:
            logger.warning("math_verify fulltext parse failed", exc_info=True)
            target = []
        if target:
            try:
                if verify(gold, target, timeout_seconds=_VERIFY_TIMEOUT):
                    return VerifyResult(
                        True,
                        "math_verify_fulltext",
                        pred=None,
                        fulltext_fallback_match=True,
                        **result_base,
                    )
            except Exception:
                logger.warning(
                    "math_verify fulltext verify failed (gt=%r)", gt, exc_info=True
                )

    return VerifyResult(False, "none", pred=pred_candidate, **result_base)


def _verify_answer_impl(solution_str: str, ground_truth: str) -> VerifyResult:
    """solution_str (モデル出力) を ground_truth と照合する。

    OpenAI Harmony 形式に従い、special token 付きの final channel または
    special token が落ちた `assistant final` 以降だけを reward 判定に使う。
    analysis 側の正答は diagnostic として記録するが reward しない。
    """
    solution_str = str(solution_str)

    has_harmony = bool(_HARMONY_TOKEN_RE.search(solution_str))
    if has_harmony:
        scope = _extract_harmony_scope(solution_str)
        if scope is None:
            return VerifyResult(
                False,
                "harmony_parse_failed",
                pred=None,
                scored_channel="final",
                has_harmony=True,
                has_harmony_final=False,
            )
    else:
        scope = _extract_plain_assistant_final_scope(solution_str)
        if scope is None:
            return VerifyResult(
                False,
                "format_violation",
                pred=None,
                scored_channel="final",
                has_harmony=False,
                has_harmony_final=False,
            )

    if scope.final_text is None:
        result = VerifyResult(
            False,
            "no_final",
            pred=None,
            scored_channel="final",
            has_harmony=has_harmony,
            has_harmony_final=False,
        )
    else:
        result = _verify_answer_text_impl(scope.final_text, ground_truth)
        result.scored_channel = "final"
        result.has_harmony = has_harmony
        result.has_harmony_final = True

    if not result.ok and scope.analysis_text:
        analysis_result = _verify_answer_text_impl(scope.analysis_text, ground_truth)
        result.analysis_ok = analysis_result.ok
        result.analysis_method = analysis_result.method

    return result


def verify_answer(solution_str: str, ground_truth: str) -> VerifyResult:
    """Timeout-protected public verifier entrypoint.

    By default this runs math_verify inside a small persistent subprocess pool.
    Set `MATH_VERIFY_POOL_ENABLED=0` to bypass the pool for local debugging.
    """
    if not _env_bool("MATH_VERIFY_POOL_ENABLED", True):
        return _verify_answer_impl(solution_str, ground_truth)
    return _get_pool().compute(str(solution_str), str(ground_truth))
