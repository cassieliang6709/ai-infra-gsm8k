import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from composite_reward import compute_score


def score(text, gt="72"):
    return compute_score("openai/gsm8k", text, gt)


def test_correct_and_clean_stop_gets_both_terms():
    assert score("48/2 = 24\n48 + 24 = 72\n#### 72") == {"score": 1.1, "acc": 1.0, "format": 1.0}


def test_text_after_answer_loses_format_bonus():
    assert score("#### 72\nLet me double check...")["format"] == 0.0
    assert score("#### 72\nLet me double check...")["acc"] == 1.0


def test_wrong_answer_with_clean_format():
    assert score("#### 71") == {"score": 0.1, "acc": 0.0, "format": 1.0}


def test_no_answer_scores_zero():
    assert score("I am not sure.")["score"] == 0.0


def test_thousands_separator_matches_ground_truth():
    assert score("#### 1,200", gt="1200")["acc"] == 1.0
