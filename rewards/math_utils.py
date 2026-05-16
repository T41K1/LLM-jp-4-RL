import multiprocessing as mp
import queue as _queue_mod
import re
import signal
import threading

import sympy
from lark.exceptions import LarkError
from sympy.parsing.latex import parse_latex




# reference code: https://github.com/allenai/open-instruct/blob/main/open_instruct/math_utils.py




import logging


def setup_logger(name: str | None = None, rank: int = 0) -> logging.Logger:
    """Set up a logger with consistent formatting across the project.

    This function configures logging.basicConfig with a standard format
    that includes timestamp, level, filename, line number, and message.
    It only configures basicConfig once to avoid overwriting existing config.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.
        rank: Process rank in distributed training. Only rank 0 logs INFO.

    Returns:
        Logger instance with the specified name
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return logging.getLogger(name)

eval_logger = setup_logger('math_utils')





#boxedの処理
def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]

# boxedの中から答えを取る
def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]



SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]



#  Minerva形式で学習されたものに対して
def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(r"Final Answer: The final answer is(.*?). I hope it is correct.", text)
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER    


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message
        # signal.signal / signal.alarm はメインスレッド以外では ValueError を投げる。
        # verl の reward worker は非メインスレッドで動く場合があるので、その時は no-op。
        self._use_signal = threading.current_thread() is threading.main_thread()

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if self._use_signal:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if self._use_signal:
            signal.alarm(0)


# sympy.simplify can hang in C extensions on adversarial latex; SIGALRM is
# unreliable (worker threads + native code). Cap inputs upfront. Real math
# answers are short — 256 chars is well above any legitimate ground truth.
IS_EQUIV_MAX_LEN = 256

# is_equiv は子プロセスで実行し、規定時間で OS から強制終了する。
# SIGALRM では C 拡張内に居る sympy を止められないため。
#
# start method は forkserver: verl の reward worker は asyncio + ThreadPoolExecutor
# でマルチスレッドなので、直接 fork すると親のロック状態が子に中途半端に
# コピーされてデッドロックする可能性がある (Python 3.12 で警告化済)。
# forkserver は初回に1つだけシングルスレッドの中継プロセスを建て、そこから
# 子を fork するため、マルチスレッド親からの fork 問題を回避できる。
_IS_EQUIV_TIMEOUT_SEC = 5
_MP_CTX = mp.get_context("forkserver")  # Linux 前提


def _is_equiv_worker(x1: str, x2: str, result_queue) -> None:
    """子プロセスで sympy 比較を実行し、結果を queue に詰める。"""
    try:
        parsed_x1 = parse_latex(x1, backend="lark")
        parsed_x2 = parse_latex(x2, backend="lark")
    except (
        sympy.parsing.latex.errors.LaTeXParsingError,
        sympy.SympifyError,
        LarkError,
        TypeError,
    ):
        result_queue.put(("ok", False))
        return
    try:
        diff = parsed_x1 - parsed_x2
    except TypeError:
        result_queue.put(("ok", False))
        return
    try:
        result_queue.put(("ok", bool(sympy.simplify(diff) == 0)))
    except ValueError:
        result_queue.put(("ok", False))
    except Exception as e:
        result_queue.put(("err", repr(e)))


def is_equiv(x1: str, x2: str, timeout_sec: int = _IS_EQUIV_TIMEOUT_SEC) -> bool:
    """
    x1 and x2 are normalized latex string.

    sympy parse/simplify は子プロセスで実行し、`timeout_sec` で OS から強制終了する。
    """
    if x1 is None or x2 is None:
        return False
    if len(x1) > IS_EQUIV_MAX_LEN or len(x2) > IS_EQUIV_MAX_LEN:
        eval_logger.debug(
            f"is_equiv: skipping sympy (len={len(x1)},{len(x2)} > {IS_EQUIV_MAX_LEN})"
        )
        return False

    result_queue = _MP_CTX.Queue()
    proc = _MP_CTX.Process(target=_is_equiv_worker, args=(x1, x2, result_queue))
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join()
        eval_logger.debug(
            f"is_equiv: timed out after {timeout_sec}s comparing {x1!r} vs {x2!r}"
        )
        return False

    try:
        status, value = result_queue.get_nowait()
    except _queue_mod.Empty:
        eval_logger.debug(
            f"is_equiv: worker exited without result (exitcode={proc.exitcode})"
        )
        return False

    if status == "ok":
        return value
    eval_logger.debug(f"is_equiv: worker error {value}")
    return False


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def hendrycks_is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
