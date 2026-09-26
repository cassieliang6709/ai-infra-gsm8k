"""Heuristic 4-way classification of GSM8K misses (checked in order):
1. runaway    — response >= 2000 tokens (never emitted the stop token)
2. format     — no '#### <number>', or flexible-extract right but strict-match wrong
3. arithmetic — some 'a op b = c' step is wrong (<<expr=val>> or inline 'x + y = z')
4. reasoning  — everything else: arithmetic checks out, but the setup is wrong
Usage: python analysis/error_analysis.py name=<lm-eval output dir> ...
"""
import json, glob, re, sys
from collections import Counter
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

NUM = r"-?\d[\d,]*\.?\d*"
ANNOT = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
INLINE = re.compile(rf"({NUM}(?:\s*[-+*/x×]\s*{NUM})+)\s*=\s*\$?({NUM})")

def safe_eval(expr):
    expr = expr.replace(",", "").replace("x", "*").replace("×", "*").replace("$", "")
    if not re.fullmatch(r"[\d.+\-*/() ]+", expr): return None
    try: return eval(expr)
    except Exception: return None

def has_calc_error(text):
    for expr, val in ANNOT.findall(text) + INLINE.findall(text):
        v = safe_eval(expr); 
        try: t = float(val.replace(",", "").rstrip("."))
        except ValueError: continue
        if v is not None and abs(v - t) > 1e-4 * max(1, abs(t)): return True
    return False

def classify(strict, flex):
    text = strict["resps"][0][0]
    if len(tok(text)["input_ids"]) >= 2000: return "runaway"
    if "####" not in text or (flex["exact_match"] and not strict["exact_match"]): return "format"
    if has_calc_error(text.split("####")[0]): return "arithmetic"
    return "reasoning"

rows = {}
for arg in sys.argv[1:]:
    name, run_dir = arg.split("=", 1)
    s = [json.loads(l) for l in open(glob.glob(f"{run_dir}/*/samples_*.jsonl")[0])]
    st = {x["doc_id"]: x for x in s if x["filter"] == "strict-match"}
    fl = {x["doc_id"]: x for x in s if x["filter"] == "flexible-extract"}
    wrong = {i: classify(st[i], fl[i]) for i in st if not st[i]["exact_match"]}
    rows[name] = (len(st), wrong)
    
cats = ["arithmetic", "reasoning", "format", "runaway"]
print("| model | wrong | " + " | ".join(cats) + " |")
print("|---|---|" + "---|" * len(cats))
for name, (n, w) in rows.items():
    c = Counter(w.values())
    print(f"| {name} | {len(w)}/{n} | " + " | ".join(f"{c[k]} ({c[k]/len(w):.0%})" for k in cats) + " |")
