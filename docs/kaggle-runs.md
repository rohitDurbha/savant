# Kaggle runs

Each row below records one Kaggle run of the notebooks. A run starts only when `kaggle/RUN` changes, and the publish workflow then rebuilds the bench dataset from the current code and runs every notebook once.

| Date | Notebook | Reason |
| --- | --- | --- |
| 2026-11-05 | 00-baseline, versions 1 and 2 | First baseline, cancelled because the time split hid trends and version 2 was started by mistake |
| 2026-11-05 | 00-baseline, version 3 | Baseline with the interleaved time split, failed with out of memory after five tables |
| 2026-11-05 | 00-baseline, version 4 | Failed in its first minute because it mounted the previous dataset version |
| 2026-11-05 | 00-baseline, version 5 | Completed, results in results/baseline-00.json |
| 2026-11-09 | 00-baseline, version 6 | Completed, blind and guided results in results/baseline-01.json |
