# Savant

Savant is a personal analytics engine that reads a table and reports only the findings that survive a strict set of statistical checks. A small open model plans the analysis, writes short code and explains the results, while a deterministic statistics engine does the testing. The model is never allowed to type a number, so every figure in a Savant report links back to code that actually ran.

## Why it is built this way

Analysts and models both mistake noise for signal when they look at many slices of data. Savant treats that as the main failure to prevent, so every candidate finding passes the same gates before anyone reads it.

1. Every test the scan runs goes into a ledger, and false discovery control runs over the whole ledger.
2. A finding needs a practical effect size, and pairs that follow from keys or hierarchies are never tested.
3. Near-perfect single-column relationships are flagged as likely leakage and never reported as insights.
4. A finding must hold again on rows the scan never explored, split by time when the table has dates.
5. A finding must reappear in most half-sample bootstraps before it is reported.

Findings that pass these gates are labeled as associations only. Savant does not claim causes from observational data.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
savant plant data/demo --null 2 --planted 2
savant analyze data/demo/plant-000.csv
savant calibrate --null 50 --planted 50
```

To ask questions through a local model, run any OpenAI-compatible server such as Ollama and point Savant at it.

```bash
savant ask data/demo/plant-000.csv "What drives score0?" --base-url http://localhost:11434/v1 --model qwen3.5:4b
```

## Calibration

The engine is tuned on two kinds of generated tables. Null tables have no real relationship between any two columns, so every finding on them is false. Planted tables carry known patterns, so they measure how much real signal still gets through. Development runs use their own seeds, and the held-out bench seed is never used to choose a setting.

## Training on Kaggle

The `kaggle` workflow builds the package, writes the held-out bench tables, publishes both as a private Kaggle dataset and pushes the notebooks in `kaggle/notebooks`, which then run on a free GPU. It needs two repository secrets, `KAGGLE_USERNAME` and `KAGGLE_TOKEN`, which come from the API section of your Kaggle settings.

## Status

The project is in Phase 0, which builds the harness and records an honest baseline. The phases and their gates are in `docs/PLAN.md`.

## License

The code is released under the MIT license.
