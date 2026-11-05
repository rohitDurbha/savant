# Kaggle runs

Each row below records one Kaggle run of the notebooks. A run starts only when `kaggle/RUN` changes, and the publish workflow then rebuilds the bench dataset from the current code and runs every notebook once.

| Date | Notebook | Reason |
| --- | --- | --- |
| 2026-11-05 | 00-baseline, versions 1 and 2 | First baseline, cancelled because the time split hid trends and version 2 was started by mistake |
| 2026-11-05 | 00-baseline, version 3 | Baseline with the interleaved time split, failed with out of memory after five tables |
| 2026-11-05 | 00-baseline, version 4 | Baseline with bounded agent memory and per-table saving, built from commit c43c7aa |
