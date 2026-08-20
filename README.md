# AI Infra · GSM8K track

Reproducible scaffolding for a GSM8K baseline / LoRA SFT / GRPO comparison.

> **Status: scaffolding only. No training run has been executed yet.**
> Every metric in `evals/` is `null` with `measured: false` and
> `status: scaffolding_only`. Nothing here is a result, and none of it belongs
> on a résumé until the table in `INFRA_MILESTONE_01.md` is filled from a real
> run.

## Why this is its own repository

This track started inside [CorpCheck](https://github.com/cassieliang6709/corpcheck),
a retrieval-and-evidence system over SEC filings. Mathematical reasoning
fine-tuning shares no code, no data and no thesis with that project, and
carrying it in the same repository made one project read as two. It was split
out on 2026-08-20 from corpcheck commit `154f1df`.

CorpCheck keeps its own official benchmark track — FinanceBench, FinRank,
AVeriTeC and an internal claim benchmark — under `evaluation/official/` there.
Those entries were removed from this repository's manifest.

## Layout

| Path | What it is |
| --- | --- |
| `benchmarks/official/GSM8K/*.yaml` | Fixed hyperparameters per mode (seed 42) |
| `benchmarks/official/benchmark_manifest.yaml` | The three planned runs |
| `scripts/run_gsm8k.py` | Entry point; resolves config → command → result record |
| `scripts/train_gsm8k.py` | Training harness |
| `scripts/run_placeholder_pipeline.py` | Offline stand-in so `--smoke` runs without a training stack |
| `evals/results_<label>.json` | Run records, tracked in git on purpose |
| `INFRA_MILESTONE_01.md` | The comparison table, to be filled from real runs |

## Running

```bash
# Wiring check — writes a scaffolding record, trains nothing
python scripts/run_gsm8k.py --mode baseline \
  --config benchmarks/official/GSM8K/baseline_config.yaml --smoke

python -m pytest -q
```

Every result record carries `dataset_version`, `commit`, `timestamp`,
`hardware`, `seed` and `params`, so a filled table can always be traced back to
the code that produced it.

## Before claiming anything

1. Run baseline, LoRA and GRPO with the pinned seeds.
2. Record pass@1, latency and VRAM into `INFRA_MILESTONE_01.md`.
3. Quote the absolute pair, never a ratio, and ship the qualifiers with the
   number.
