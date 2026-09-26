"""CoT distillation data: Qwen2.5-7B-Instruct writes step-by-step solutions for GSM8K train
(greedy); keep only traces whose final '#### <number>' matches the reference (rejection sampling).
Output uses the same format as the SFT data (user = question only), so SFT and eval are unchanged.
"""
import json, re, sys
from vllm import LLM, SamplingParams

teacher, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open("data/gsm8k_train.jsonl")]
qs = [r["messages"][0]["content"] for r in rows]
gts = [r["messages"][1]["content"].split("####")[-1].strip().replace(",", "") for r in rows]
INSTR = "\n\nSolve the problem step by step. Put the final numeric answer alone on the last line in the form '#### <number>'."

llm = LLM(teacher, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=2048)
outs = llm.chat([[{"role": "user", "content": q + INSTR}] for q in qs],
                SamplingParams(temperature=0, max_tokens=1024))
ANS = re.compile(r"#### (\-?[0-9\.\,]+)")
kept, lens = [], []
for q, gt, o in zip(qs, gts, outs):
    t = o.outputs[0].text.strip()
    m = ANS.findall(t)
    if m and m[-1].replace(",", "").rstrip(".") == gt and t.splitlines()[-1].startswith("####"):
        kept.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": t}]})
        lens.append(len(o.outputs[0].token_ids))
with open(out, "w") as f:
    for k in kept: f.write(json.dumps(k, ensure_ascii=False) + "\n")
print(f"kept {len(kept)}/{len(rows)} ({len(kept)/len(rows):.1%}), mean response tokens {sum(lens)/len(lens):.0f}")
