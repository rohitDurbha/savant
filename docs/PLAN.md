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

## Phase 2 progress

Phase 2 started on November 10, 2026, and its first item closed the placeholder zero issue from Phase 1. A numeric column is flagged when its zeros form a spike with almost nothing just above zero, in a column whose typical value is large, and the data card names any text column whose label lines up with the zero rows. Every confirmed finding on such a column is tested again with the zeros set aside. A finding that weakens, fails or flips direction is marked as resting on zeros that likely mean unknown and is never reported, while a finding that holds keeps a note saying so. A test built to look like the electric vehicle file reproduces the false fall in range and now drops it, and a second test confirms that real zeros in counts are left alone.

Training conversations are written by code. A scripted analyst reads the same opening the model reads, writes pandas code that measures each guarded finding, runs that code in the sandbox and answers with placeholders and PAIR lines. A conversation is kept only when its code ran, every number in the answer resolves to an emitted result and no named pair is false in the planted truth. A weak plant that the engine missed is simply not named, which is honest given the evidence, so those conversations are kept and their recall is recorded. Lookup questions such as which region has the highest average price are included too, and their answers say whether the guarded scan confirmed the gap. The generator refuses any seed within the bench range, and the pilot uses seeds from 3,000,000 upward.

A local trial on 40 development tables kept all 60 conversations it wrote, 17 of which correctly answered with PAIR: none. The kept planted conversations name 81% of the planted pairs, which matches the engine because the analyst only repeats what the engine confirmed.

The pilot trains a LoRA adapter in half precision with the peft library and a short custom loop, which only scores the tokens the analyst wrote. This replaces the QLoRA and Unsloth plan for the pilot, because it avoids two extra installs on Kaggle and a 4B model fits on one 16 GB card with gradient checkpointing. The pilot is then measured on the same ten bench tables as the Phase 0 baseline, in blind and guided modes.

Two limits are known before the pilot runs. Training and bench tables come from the same generator with the same column names, so a gain on the bench may overstate the gain on real files, and a real data check follows the pilot. The pilot also trains only the guided mode, so blind mode is expected to stay near the baseline.

## Phase 2 first pilot result

The first pilot ran on November 10, 2026, on two T4 GPUs, and it did not beat the base model in guided mode. The generator kept 357 of 358 conversations, and 333 of them trained a LoRA adapter for 41 steps in about 38 minutes. Loss on the held-out conversations fell from 2.82 to 0.03. That fall mostly shows that the scripted answers repeat a small set of sentence templates, which the adapter learned almost word for word.

| Mode | Measure | Base model | Pilot adapter |
| --- | --- | --- | --- |
| Guided | Grounded answers | 0.9 | 0.8 |
| Guided | Recall on planted pairs | 0.70 | 0.36 |
| Guided | Mean rounds | 2.5 | 4.2 |
| Guided | False pairs per table, null and hard null and planted | 0, 0, 0.25 | 0, 0, 0 |
| Blind | Grounded answers | 0.5 | 0.7 |
| Blind | Recall on planted pairs | 0 | 0 |
| Blind | False pairs per planted table | 5.5 | 0 |

The adapter made the model more cautious in both modes, and blind mode stopped inventing relationships. In guided mode it lost recall for three reasons that the transcripts show clearly. Every training conversation was one clean code step followed by an answer, so the adapter never saw how to recover when its own code failed, and 32% of its code steps failed on dates, text columns and the scipy result object. The adapter also copied numbers from the findings list into its answers, and the grounding check then rejected those answers until the last round. Finally, ten tables with eleven planted pairs make a small test, so a gap of this size can come from a few tables.

The next pilot should add conversations where a first code step fails and the second step fixes it, using the error types seen here. It should also lower the learning rate or the number of steps, because the loss curve flattened by step 30. The evaluation should grow to at least twenty planted tables in guided mode, so that one or two tables cannot decide the result. These changes come from reading the pilot transcripts on bench tables, so they are recorded here as choices made after seeing bench results.

## Phase 2 second pilot design

The second pilot makes the three changes proposed after the first pilot, and every one of them was chosen after reading bench transcripts. About two in five training conversations now begin with a realistic wrong first try, such as comparing plain dates with text, averaging a whole table that holds text columns or reading a chi2 attribute that scipy does not provide. Each of those conversations shows the real error from the sandbox, a short note on the fix and the working code as the second step. Answers now come in two phrasings for each kind of finding, and the plan text tells the model that numbers in the findings list are measured again in code before they are cited. Training uses 400 development tables and half the earlier learning rate. Guided mode is measured on 5 null, 5 hard null and 20 planted bench tables, with the base model and the adapter answering the same tables in the same session, so the comparison does not lean on the earlier baseline run.

## Phase 2 second pilot result

The second pilot finished on Kaggle, and the adapter did worse than the base model on every guided measure. The generator kept 606 of 607 conversations, training ran 72 steps in about an hour, and held-out loss fell from 2.69 to 0.08.

| Run | Tables | Grounded answers | Mean rounds | Failed code steps | Recall | False per planted table | False per null table | False per hard null table |
|---|---|---|---|---|---|---|---|---|
| base guided | 30 | 1.0 | 1.8 | 0.375 | 0.824 | 0.2 | 0.0 | 0.0 |
| adapter guided | 30 | 0.7 | 5.5 | 0.533 | 0.275 | 0.0 | 0.0 | 0.0 |
| adapter blind | 10 | 0.9 | 4.4 | 0.208 | 0.0 | 0.0 | 1.667 | 2.333 |

The transcripts point to two causes that we can name with confidence. The first cause is that the broken first steps in the recovery conversations carried training labels, so the adapter learned to write the same mistakes on purpose. Its failed steps include the chi2 attribute error, dates compared with text and the date accessor on plain dates, which are exactly the mistakes we planted. It also assumed that variables from an earlier step still exist, which they never do in the sandbox. The second cause is that the very low held-out loss shows the adapter memorized the answer templates. Many of its answers cite placeholders such as a mean price from steps that never emitted them, or they type dates and sizes copied from the findings list, so the grounding check rejects them until the round limit runs out.

The base model in guided mode is the strongest setup we have measured so far. It grounded every answer, found 42 of the 51 planted pairs and named no pair on any null or hard null table. Its weak spot is that more than a third of its code steps still fail before it recovers.

## Phase 2 fixes for failed code steps

After the second pilot we chose to keep the base model and reduce its failed code steps, because it already grounds every guided answer and names no pair on null tables. This choice was made after reading bench transcripts. Date columns now reach the model's code as pandas datetimes, which removes most of the date errors seen in both pilots. The prompt now says that every step starts fresh in a new process, so the model imports what it needs and rebuilds its variables inside each step. Common errors such as a missing name, resampling without a date index, the chi2 attribute and averaging text columns now come back with a one-line hint. The 04-guided notebook measures the base model on the same 30 guided tables as the second pilot and on the ten blind baseline tables, with no training.

## Phase 2 result after the fixes for failed code steps

The 04-guided run measured the base model on the same tables as the second pilot, and the full output lives in `results/guided-01.json`. The blind columns compare with the blind run in `results/baseline-01.json`.

| Run | Grounded answers | Mean rounds | Failed code steps | Recall | False per null table | False per hard null table | False per planted table | Unverified claims |
|---|---|---|---|---|---|---|---|---|
| guided before the fixes | 1.0 | 1.8 | 0.375 | 0.824 | 0.0 | 0.0 | 0.2 | 4 |
| guided after the fixes | 1.0 | 2.47 | 0.31 | 0.804 | 0.0 | 7.0 | 0.3 | 42 |
| blind baseline | 0.5 | 7.0 | not measured | 0.0 | 0.667 | 0.333 | 5.5 | 0 |
| blind after the fixes | 1.0 | 3.2 | 0.227 | 0.0 | 6.667 | 0.333 | 1.0 | 0 |

The date errors are gone, and the remaining failures are mostly names from an earlier step. The fixes also exposed a weakness that the failing code had been hiding from us. Once its own code ran, the model built its own correlation tables and listed every pair above its cutoff, which gave 27 false pairs on one hard null table and 6 on another. The report already marked these pairs as possibly noise, yet the answer still named them. In guided mode an answer that names a pair the guarded checks did not confirm now goes back to the model once, with a request to drop the pair or mention it only as exploratory. This change was also made after reading bench results, and the second 04-guided run measures it on the same tables.

## Phase 2 result after the unconfirmed pair check

The second 04-guided run answered the same tables, and its output lives in `results/guided-02.json`. Sending an unconfirmed pair back once removed every false pair from the null and hard null tables, and the base model in guided mode now gives its cleanest result so far.

| Guided run on the same 30 tables | Grounded answers | Mean rounds | Failed code steps | Recall | False per null table | False per hard null table | False per planted table | Unverified claims |
|---|---|---|---|---|---|---|---|---|
| before the fixes | 1.0 | 1.8 | 0.375 | 0.824 | 0.0 | 0.0 | 0.2 | 4 |
| after the fixes for failed steps | 1.0 | 2.47 | 0.31 | 0.804 | 0.0 | 7.0 | 0.3 | 42 |
| after the unconfirmed pair check | 1.0 | 2.83 | 0.333 | 0.784 | 0.0 | 0.0 | 0.1 | 2 |

Recall moved from 42 to 40 of the 51 planted pairs, which is within the noise we expect from one run on twenty planted tables. Blind mode is unchanged because the check only applies when the guarded findings are shown. The remaining failed steps are almost all names from an earlier step, which the model still reaches for even with a hint.

Three of these choices were made after reading results from the same 30 bench tables, so these numbers may flatter the setup. The other bench tables have never been shown to the guided model, and a run on them is the honest check before any claim is made. The Phase 2 gate asks for a fine-tuned model that beats the base model, and neither pilot has done that, so the gate is not met.

## Phase 2 third pilot design

The third pilot follows what recent work on data analysis agents found, and it was designed after reading results from both earlier pilots and the bench. On the 20 planted guided tables the engine confirmed 46 of the 51 planted pairs, while the base model named 40 of them, so the model drops about 13% of the confirmed findings it is handed. A third of its code steps still fail, and in blind mode it finds nothing real on its own.

The earlier pilots trained on answers written by a script. That put the training targets far from how the model writes, which the self-distillation paper links to forgetting, and the very low held-out loss showed that the adapter memorized templates. The broken first steps also carried labels, so the adapter learned the mistakes we planted. The third pilot changes both choices in the way ReST-EM and DataMind recommend.

1. The base model writes its own training data. It answers 300 development tables four times each at temperature 0.7 through a vLLM server, using seeds from 4,000,000 upward.
2. A run is kept only when it is grounded, names every real pair the engine confirmed, names no false pair and never lost context to the message window. At most two kept runs per table are used, shortest first, so easy tables cannot dominate.
3. Failed code steps, empty turns and rejected answers stay in the context without labels, so the model learns how a fix follows an error without learning the error itself.
4. Null and hard null tables answered with PAIR: none stay in the mix as negative examples.
5. Checkpoints are chosen on separate development tables by task results, never by loss, and the bench is used once at the end with the base model and the adapter in the same session, on the 30 tables used so far and on bench tables the model has never seen.

The targets for the adapter in guided mode are to name at least 97% of the confirmed pairs, to keep failed code steps under 15%, to answer within two rounds on average and to name no false pair on null tables. A short probe run of the sampling notebook comes first, so that the speed of vLLM on the T4 cards and the keep rate are known before the full run.

The probe ran on 24 development tables with four samples each, and vLLM worked on both T4 cards without any special flags. It finished 745 runs per hour, which is about thirteen times the speed of the earlier generation loop, after about eight minutes spent installing vLLM and loading the model. The checks kept 71 of the 96 runs, including every hard null run, 79% of the null runs and 70% of the planted runs. Even with sampling at temperature 0.7, the base model named 86% of the confirmed pairs, and 26% of its code steps failed. The most common failures were scipy functions that an earlier step had imported, dates passed straight to spearmanr and code that tried to read a file named data.csv.

The full sampling run covered 500 development tables with four samples each, so the model wrote 2,000 runs in about two hours and twenty minutes at 870 runs per hour. The checks kept 1,482 of them, and at least one run survived on 462 of the 500 tables. The keep rate was 85% on both kinds of null table and 68% on planted tables, where the model named about 80% of the confirmed pairs on average while sampling at temperature 0.7. After the cap of two runs per table, 621 conversations remain for training, and 245 of them answer PAIR: none on a null table. Failed code steps stayed near 27% of all steps, and the most common ones were still a scipy function used without an import, code that read a file named data.csv and dates passed straight into a correlation. The hints help the model repair those steps, and since failed steps carry no labels, the adapter only learns the repairs. The summary is in results/selfplay-full.json.

The LoRA training run used 562 of the 621 conversations and kept the other 59, drawn from separate tables, aside to measure held-out loss. Every conversation fit inside the length limit, with a median of about 1,800 tokens and a longest one of about 3,900. Two epochs took 142 optimizer steps and about two hours and forty minutes on one T4 card. Held-out loss started at 0.20, because the targets were already written in the model's own words, and it fell steadily to 0.127, 0.124, 0.118 and 0.116 at the four saved checkpoints. The second pilot pushed held-out loss close to zero when the adapter memorized scripted templates, and this run shows no sign of that. Loss does not choose the checkpoint, so the evaluation notebook scores every checkpoint on separate development tables before it touches the bench. The summary is in results/train-03.json.
