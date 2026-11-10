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

## Phase 0 second real data run

The second run on the same four tables, on November 10, 2026, confirmed the fixes, and its output lives in `results/real-02.json`. No value was changed on a guess, since the 12,501 VIN prefixes, 8,577 model names, three town names and 65 question codes were listed as possible typos and left alone. Postal codes, districts, GEOIDs and location ids were treated as codes, and map points, cities, counties and model names with too many distinct values were left out of the tests.

The electric vehicle table now gives four confirmed findings, each of which a domain expert would recognize. Battery electric vehicles are mostly eligible for the clean fuel program, newer model years have longer electric range, range differs by make, and battery electric vehicles are newer than plug-in hybrids on average. The chronic disease table still gives 34 structural findings because it is in long format. The two messy tables give none.

Reading the saved transcripts showed three more problems, which are fixed with tests.

1. The model often wrote its final PAIR lines without the Answer tags, so the loop kept asking until it ran out of rounds. A plain reply with no code is now treated as the answer and still has to pass grounding.
2. On the warehouse table the engine found nothing, yet the guided model listed five relationships. In guided mode every claimed pair is now checked against the guarded findings, and any pair without one is marked in the report as not confirmed and possibly noise.
3. A group difference larger than 0.95 standard deviations was labeled near-deterministic, which caught ordinary large effects such as Nissans being older cars on average. Group differences now count as near-deterministic only when the groups explain at least 90% of the variance, and the development sweep is unchanged at zero false findings and 89% recall.

## Phase 0 result

The Phase 0 gate is met. The loop runs end to end on real files on Kaggle, and the baseline scores for the engine, the blind model and the guided model are recorded in `results/`. The fixes from the transcripts of the second real data run have tests but have not yet run on Kaggle, so the first Phase 1 run will confirm them.

## Phase 1 progress

Work on Phase 1 started on November 10, 2026, and each piece below began with a failing test.

1. Pairs that only move together because both trend over time are retested with time held fixed, using ranks with the straight-line time trend removed. A pair whose link disappears is marked as explained by time, and derived findings on the development suite fell from three to zero.
2. Long tables such as the CDC indicators are detected from a measure column and a value column. The most common measures are analyzed one at a time with one value type each, confidence limits and footnotes are set aside, and each measure's false discovery rate is divided by the number of measures.
3. Guarded findings are ranked into a short list of insights. Each score is the effect against the conventional large effect for its measure, times an impact that grows with the share of rows covered, and findings that share a column are grouped under the strongest one with a plain sentence that never claims a cause.
4. Level shifts over time are tested at seventeen candidate cuts, with the p-value multiplied by the number of cuts and corrected for autocorrelation left after the step. Unusual values are counted in the data card with the Iglewicz and Hoaglin rule, and no row is dropped because of them.

Level shifts are now one of six planted pattern types, so the development suite and the held-out bench both changed composition from this commit on. The sweep in `results/calibration-dev.json` at the default settings found no false findings on 100 easy and 100 hard null tables, recall of 90% overall and 95% for level shifts, and all 50 planted leaks flagged.

5. Driver analysis fits a regularized gradient boosted model on the exploration rows and shuffles each feature on the confirmation rows. A feature counts only when its held-out importance stays above zero across repeats, only when the model beats a simple baseline, and never when the feature alone predicts the target almost perfectly.
6. Every insight gets a Vega-Lite chart chosen for its kind of finding, and a linter blocks bars that skip zero, dual axes, crowded pies and unordered categories. The report command writes one HTML page with ranked insights, charts and the cleaning report.

## Phase 1 gate

The gate asks that false findings on null tables stay under target while recall holds. The target follows from the false discovery rate of 0.05 that was fixed in Phase 0, since under a fully null table that rate bounds the chance of any false finding at 5%. This is the first time the target is written as a number, and it is written after the bench result below was seen.

The held-out bench ran once at the default settings on November 10, 2026, and its output lives in `results/bench-phase1.json`.

| Measure | Phase 0 bench | Phase 1 bench |
| --- | --- | --- |
| Easy null tables with any false finding | 0 of 50 | 0 of 50 |
| Hard null tables with any false finding | 2 of 50 | 1 of 50 |
| Recall on planted tables | 88% | 92% |
| Planted leaks flagged | 31 of 31 | 26 of 26 |

Both null rates sit under 5% and recall rose by four points, so the gate passes. The bench composition changed between the two phases when level shifts became a planted pattern, so the two columns are close but not identical tests.

The one false finding linked two slowly drifting sensors in a table with repeat customers. It was studied after the bench result was seen, and no setting was changed because of it. On 200 fresh hard tables from a new development seed the same pair was never reported, and across every hard table checked so far the rate is one in 350, which is well inside what the false discovery rate allows. Two regression tests now cover independent slow drifts and a real link between drifting series.

## Phase 1 real data reports

The first CPU run of the report notebook on November 10, 2026, wrote HTML reports for all four savant-real files, and it showed two problems. The CDC table was not detected as long, because the check that the measure explains the value failed when a few raw counts in the tens of thousands sat beside percentages. Detection now relies on the measure and value column names and on each measure repeating across many rows, with a test that mixes counts and percentages.

The electric vehicle report says that range falls as model year rises, which is true in this file and misleading about cars. The file records an electric range of zero when the range has not been researched, which is the case for most recent models, so a placeholder zero bends the result. Detecting zeros that stand for unknown values, and flagging insights that rest on them, is the first item for the next phase.

The second CPU run detected the CDC table as long and analyzed its eight most common measures one at a time. The findings read as public health statements, such as arthritis and asthma mortality being higher among people aged 65 and older. For asthma mortality the most common value type was likely a raw count, so a state with a large population stood out for its size. Value types are now chosen in the order age-adjusted rate, crude rate, rate, prevalence, percent, mean and median, with raw counts used only when nothing else exists.

## Phase 1 result

The Phase 1 gate is met on the held-out bench, and every new piece also ran on real files. Two known issues carry into Phase 2, which are placeholder zeros such as an unresearched vehicle range and the guided model fixes that still await a Kaggle run.
