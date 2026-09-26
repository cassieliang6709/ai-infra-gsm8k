"""Why do some models never stop? Checks whether <|im_end|> was ever trained (embedding / lm_head
row norms vs. ordinary tokens) and how much probability the model puts on it right after the answer.
Usage: python analysis/eos_probe.py <model path or name> [more models...]
"""
import sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SPECIAL = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
# 50 GSM8K train problems with reference solutions: the position right after a finished answer
rows = [json.loads(l) for l in open("data/gsm8k_train.jsonl")][:50]

def to_msgs(r):
    if "messages" in r: return r["messages"]
    return [{"role": "user", "content": r["query"]}, {"role": "assistant", "content": r["response"]}]

for path in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    E = model.get_input_embeddings().weight.detach()
    H = model.get_output_embeddings().weight.detach()
    norms_e, norms_h = E.norm(dim=1), H.norm(dim=1)
    # reference: ordinary tokens (first 150k ids, no special tokens)
    ref_e, ref_h = norms_e[:150000].median().item(), norms_h[:150000].median().item()
    print(f"\n=== {path}  tie_word_embeddings={model.config.tie_word_embeddings}")
    print(f"ordinary tokens: median embed norm {ref_e:.3f} | median lm_head norm {ref_h:.3f}")
    for s in SPECIAL:
        i = tok.convert_tokens_to_ids(s)
        print(f"{s:15s} id={i}  embed_norm={norms_e[i]:.3f}  lm_head_norm={norms_h[i]:.3f}")

    # probability that the next token after the final answer is <|im_end|>
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    probs, ranks = [], []
    for r in rows:
        msgs = to_msgs(r)
        text = tok.apply_chat_template(msgs, tokenize=False)
        cut = text.rfind("<|im_end|>")  # end of the assistant turn
        ids = tok(text[:cut], return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        p = logits.softmax(-1)
        probs.append(p[im_end].item())
        ranks.append(int((p > p[im_end]).sum()) + 1)
    probs.sort(); ranks.sort()
    print(f"P(<|im_end|>) after answer: median {probs[len(probs)//2]:.4f}  min {probs[0]:.4f}  max {probs[-1]:.4f}  (n={len(probs)})")
    print(f"<|im_end|> median rank {ranks[len(ranks)//2]} (1 = most likely next token)")
