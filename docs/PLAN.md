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

The development sweep on November 9, 2026, covered 100 easy null tables, 100 hard null tables and 100 planted tables from seed 1, and every setting lives in `results/calibration-dev.json`. The easy null tables could not tell the settings apart, because every setting reported nothing on them. The hard null tables add repeat customers, slowly drifting sensor readings, heavy tails, outliers, missing cells and rare categories, which are the conditions that fool tests built for independent rows.

The first run on hard null tables found false findings in 19 of 30 tables. Each one had a clear cause. A repeated customer id was tested as a number, rows from one customer were counted as independent evidence, and slowly drifting series looked like trends and like each other. The engine now treats a repeated id as an entity, keeps each entity on one side of the split, and runs every test except time trends on one row per entity. Tests in time order use the effective number of independent rows from the lag one autocorrelation, and a test with fewer than 30 effective rows counts as failing its assumptions.

| False discovery rate | False findings per hard null table | Hard null tables with any false finding | Recall on planted tables |
| --- | --- | --- | --- |
| 0.01 | 0 | 0 of 100 | 89% |
| 0.05 | 0 | 0 of 100 | 89% |
| 0.10 | 0.02 | 2 of 100 | 89% |

The default false discovery rate stays at 0.05, which is now the loosest setting that kept every hard null table clean. Halving the effect size bars added under one point of recall and doubling them cut recall to 66%, so the conventional small effect sizes stay too.

Segment differences remain the weakest pattern type at 70% recall, and the misses are all shifts below about 0.35 standard deviations in one of four groups. A 0.3 standard deviation shift was found in 3 of 12 tables at 2,000 rows and in 12 of 12 at 8,000 rows, so the limit is the amount of evidence in small tables and not the guards. Segment effects are now measured as the standout group against the rest, which matches the 0.2 bar in the design and tells the model which group differs.

Three findings on planted tables first looked false, and each one turned out to be two metrics that both trend over time. They are real associations with a shared cause, so the grader counts them as derived. The engine should learn to explain such pairs through the shared time column, and that check is planned for Phase 1.

## Phase 0 blind and guided baseline

The second baseline ran on Kaggle on November 9, 2026, and its numbers live in `results/baseline-01.json`. The untrained Qwen3-4B model answered ten bench tables twice. In blind mode it saw only the data card, and in guided mode it also saw the engine's guarded findings. Every answer ended with PAIR lines, which were graded against the planted truth exactly like the engine.

| Measure | Engine alone | Model blind | Model guided |
| --- | --- | --- | --- |
| Planted relationships found | 88% | 0% | 70% |
| False findings per easy null table | 0 | 0.67 | 0 |
| False findings per hard null table | 0.04 | 0.33 | 0 |
| False findings per planted table | 0 | 5.5 | 0.25 |
| Grounded answers | not applicable | 50% | 90% |
| Seconds per table | under 1 | 131 | 31 |

The blind model found none of the planted relationships and reported 22 false ones on a single table, while the guided model was faster, grounded and nearly clean. This is the clearest evidence so far for the core design, and it sets the target for training, which is a blind model that finds real patterns without inventing them.

The engine reported two false findings on the 50 hard null tables in the bench, against none on the 100 hard null development tables. This bench result was seen after the settings were fixed, so it is recorded here and the settings are not changed because of it. A larger development sweep with more varied hard null tables is the honest way to study it.

Model code could not import scipy on Kaggle, because BLAS libraries reserved more address space than the 2 GB sandbox cap. The sandbox now pins BLAS to one thread and allows 3 GB.

The Phase 0 gate also asks for the loop to run on real files, and that has not happened yet. A few public datasets with a clear license, run through the engine and the guided model on Kaggle, will close Phase 0.

## Phase 0 first real data run

The first run on real tables used the private savant-real dataset on November 10, 2026. It held a Washington State electric vehicle registration table with 172,796 rows, a CDC chronic disease indicator table with 144,053 rows, and two small synthetic tables made messy on purpose. The full output lives in `results/real-01.json`.

The cleaning stage handled the two messy tables well. It read written numbers such as thirty and two hundred, five mixed date formats, day first dates, padded text and text placeholders for missing values, and the engine then reported no findings on either table, which fits how they were generated. Both large files were exactly 45 MB and ended mid row, and the cleaner dropped each partial last line and said so.

The run also exposed serious problems, recorded here in full.

1. The rule that merged rare spellings into a common one changed real data. It rewrote 12,501 VIN prefixes, 8,577 model names such as MODEL X to MODEL Y, real town names such as Creston to Preston, and question codes such as CVD01 to CKD01. Possible typos are now only reported and never changed.
2. Integer codes such as postal codes, legislative districts, census GEOIDs and location ids were tested as quantities. They are now a code kind that is never tested.
3. Text columns with thousands of distinct values, such as map points and city names, were tested as categories. Categories now stop at 100 distinct values.
4. Columns that repeat each other, such as a state code and a state name, or two identical value columns, doubled the findings. Each duplicate is now tested once.
5. Near-perfect links between definitional columns, such as a county and its postal codes, were labeled as leakage. They are now described as near-deterministic, likely a definition or a leak.
6. The chronic disease table produced 94 findings, and the prompt that listed all of them ran the GPU out of memory. The model now sees at most fifteen findings, strongest first, with a count of the rest.

The chronic disease table is in long format, with one row per indicator, place and year, and its findings are mostly structural. Detecting long tables and reshaping them before analysis belongs to Phase 1. The model also used all ten rounds on two tables without a grounded answer, and the next run saves its transcripts so the cause can be read.
