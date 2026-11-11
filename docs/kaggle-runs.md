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
| 2026-11-10 | 04-guided, version 2 | The same tables after unconfirmed pairs are sent back to the model once |
