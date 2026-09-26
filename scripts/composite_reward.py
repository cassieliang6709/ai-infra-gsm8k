"""GSM8K composite reward: correctness 1.0 + format 0.1.
Format passes when the last non-empty line is exactly '#### <number>' (stops right after the answer).
Returns a dict: `score` trains the policy; `acc` logs correctness alone so the validation
curve is directly comparable with the accuracy-only run.
"""
import re

ANS = re.compile(r"#### (\-?[0-9\.\,]+)")
LAST_LINE = re.compile(r"^#### \-?[0-9\.\,]+$")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    found = ANS.findall(solution_str[-300:])
    pred = found[-1].replace(",", "").rstrip(".") if found else None
    correct = pred is not None and pred == ground_truth
    lines = [l.strip() for l in solution_str.strip().splitlines() if l.strip()]
    fmt = bool(lines) and bool(LAST_LINE.match(lines[-1]))
    return {"score": 1.0 * correct + 0.1 * fmt, "acc": float(correct), "format": float(fmt)}
