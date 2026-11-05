# Savant plan

Savant ships in five phases over about sixteen weeks, and each phase ends with a gate that is measured on held-out data with the output committed to this repository. The week counts are estimates for part-time work and will be reset once real Kaggle throughput is measured.

| Phase | Weeks | Scope | Gate |
| --- | --- | --- | --- |
| 0 Harness | 1 to 2 | Ingest, data card, agent loop on the base model, sandbox, planted and null generator, evaluation harness, Kaggle pipeline | The loop runs on real files and baseline scores are recorded |
| 1 Statistics | 3 to 5 | Driver, causal, anomaly and segment tools, insight ranker, chart linter and designer, calibration of every threshold | False findings on null files stay under target while recall holds |
| 2 Fine-tuning | 6 to 9 | Teacher runs with rejection, QLoRA SFT in two stages on Kaggle, hourly checkpoints | Beats the base model on the held-out planted set |
| 3 Rewards | 10 to 13 | GRPO with execution, answer and grounding rewards, trajectory monitor | Recall rises while invented numbers do not |
| 4 Launch | 14 to 16 | Distill to 2B, Hugging Face Space, gallery, model card, analyst comparison | Public demo is live with its model card and results |

## Phase 0 status

Work on Phase 0 started on November 5, 2026. The guarded engine, the null and planted generator, the grader, the sandbox, the grounded report and the agent loop are in place with tests. The baseline notebook and the bench dataset are published to Kaggle by the `kaggle` workflow once the repository has the `KAGGLE_USERNAME` and `KAGGLE_TOKEN` secrets.

Development calibration uses seed 1 and lives in `results/calibration-dev.json`. The bench uses seed 20261105 and is never used to choose a setting.

## Phase 0 baseline

The first full baseline ran on Kaggle on November 5, 2026, and its numbers live in `results/baseline-00.json`. The engine scanned all 100 held-out tables in under a minute and reported no false findings across the 50 null tables. It found 88% of planted patterns, and it flagged every one of the 31 planted leaks.

The untrained Qwen3-4B model produced a grounded answer on all ten tables it was given, in about three to four rounds each. Two of its weaknesses are already clear from this run. Nearly two thirds of its code steps failed, and its answers mostly restate the guarded findings it was handed, so the mention rate of 93% says more about the engine than about the model.

Phase 0 still needs a way to score the model on its own discoveries, and an inspection of why its code steps fail. Segment differences are the weakest pattern type for the engine at 72% recall.

## Phase 0 calibration

The first development sweep ran on November 5, 2026, over 100 null and 100 planted tables from seed 1, and every setting lives in `results/calibration-dev.json`. Each false discovery level from 0.01 to 0.1 and each effect size scale from half to double produced zero false findings on the null tables. Doubling the effect size bars cut recall from 89% to 66%, while halving them added under one point of recall.

The defaults therefore stay at a false discovery rate of 0.05 and the conventional small effect sizes. Null tables cannot separate these settings yet, because confirmation and stability already remove every false finding, so a harder null set with skewed and clustered columns comes next.

Three findings on planted tables first looked false, and each one turned out to be two metrics that both trend over time. They are real associations with a shared cause, so the grader now counts them as derived. The engine should learn to explain such pairs through the shared time column, and that check is planned for Phase 1.
