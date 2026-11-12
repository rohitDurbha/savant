# Kaggle runs

Each row below records one Kaggle run. A run starts only when `kaggle/RUN` changes, and the publish workflow then rebuilds the bench dataset from the current code and runs the notebook named in that file.

| Date | Notebook | Reason |
| --- | --- | --- |
| 2026-11-05 | 00-baseline, versions 1 and 2 | First baseline, cancelled because the time split hid trends and version 2 was started by mistake |
| 2026-11-05 | 00-baseline, version 3 | Baseline with the interleaved time split, failed with out of memory after five tables |
| 2026-11-05 | 00-baseline, version 4 | Failed in its first minute because it mounted the previous dataset version |
| 2026-11-05 | 00-baseline, version 5 | Completed, results in results/baseline-00.json |
| 2026-11-09 | 00-baseline, version 6 | Completed, blind and guided results in results/baseline-01.json |
| 2026-11-10 | 01-real, version 1 | Completed, exposed six problems listed in PLAN.md, results in results/real-01.json |
| 2026-11-10 | 01-real, version 2 | Completed, results in results/real-02.json |
| 2026-11-10 | 02-reports, version 1 | Completed, CDC table missed as long, range placeholder zeros noted |
| 2026-11-10 | 02-reports, version 2 | Completed, CDC table analyzed by measure, counts noted as a problem |
| 2026-11-10 | 03-finetune, version 1 | First LoRA pilot, failed before training because the image had an old torchao that peft refuses, after the generator kept 357 of 358 conversations |
| 2026-11-10 | 03-finetune, version 2 | Completed, adapter trained, guided recall fell below the base model, results in results/finetune-01.json |
| 2026-11-10 | 03-finetune, version 3 | Completed, adapter fell behind the base model in guided mode after learning the planted mistakes, results in results/finetune-02.json |
| 2026-11-10 | 04-guided, version 1 | Completed, failed steps fell to 31% and date errors vanished, but two hard null tables gave many false pairs, results in results/guided-01.json |
| 2026-11-10 | 04-guided, version 2 | Completed, no false pairs on null or hard null tables and recall 0.784, results in results/guided-02.json |
| 2026-11-10 | 05-selfplay, version 1 | Completed probe, vLLM ran on both T4 cards at 745 runs per hour and kept 71 of 96 runs, results in results/selfplay-probe.json |
| 2026-11-10 | 05-selfplay, version 2 | Completed, kept 1,482 of 2,000 runs and chose 621 conversations from 462 tables, results in results/selfplay-full.json |
| 2026-11-11 | 06-train, version 1 | Completed, held-out loss fell from 0.20 to 0.116 over two epochs in 2.6 hours, results in results/train-03.json |
| 2026-11-11 | 07-eval, version 1 | Stopped after the base model, because the newest peft refused to import beside the torchao that Kaggle preinstalls |
| 2026-11-11 | 07-eval, version 2 | Stopped after the base model again, because the engine process of the first vLLM server outlived it and still held the GPU when the merged checkpoint started |
| 2026-11-11 | 07-eval, version 3 | Completed, chose the epoch 1.0 checkpoint, which beat the base model on seen and unseen bench tables but missed the Phase 2 gate, results in results/eval-03.json |
| 2026-11-11 | 05-selfplay, version 3 | Completed round two from the round one adapter, kept 1,933 of 2,400 runs and chose 722 conversations from 550 tables, results in results/selfplay-round2.json |
| 2026-11-11 | 06-train, version 2 | Completed round two training, held-out loss fell from 0.20 to 0.078 over two epochs in 3.1 hours, results in results/train-04.json |
| 2026-11-12 | 07-eval, version 4 | Completed, chose the round two checkpoint at epoch 1.5, which named no false pair on any null table and needed the fewest rounds but still missed the 97% target for confirmed pairs, results in results/eval-04.json |
