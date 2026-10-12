# Publication validation

The following checks were actually completed on 2026-10-12 using Python
3.12.14. They verify retained outputs and publication code. No new model
inference was performed.

| Check | Result |
| --- | --- |
| Selected scientific sources | 205 copied files match the source-copy ledger. |
| Python syntax | 105 Python files parse successfully. |
| Archive helper boundary checks | Four unit tests pass. |
| Extension evidence | Exact archive extraction and complete offline recomputation pass. |
| Wording and history evidence | Exact archive extraction and complete offline recomputation pass. |
| Dependence and observation-control evidence | Exact archive extraction and complete offline recomputation pass. |
| Current figures | Four renderers reproduce all 102 numerical points and retain the recorded input hashes. |
| Numerical tables | Eleven tables match the published exports after normalizing operating-system line endings. |

The tested figure environment uses NumPy 2.3.5, Matplotlib 3.11.2 and Pillow
12.3.0, with a separately installed Times New Roman font. Re-rendered PDF
metadata and SVG object identifiers need not be byte identical. Figure data
and table values are checked explicitly.

Scientific checks use each frozen verifier's own comparisons. The latest
two-design verifier allows floating-point differences of at most `1e-12`
between operating systems. Integer counts, IDs, statuses and key sets remain
exact. Missing and invalid results are preserved.

Fresh GPU inference, external weight downloads and new tokenizer decoding
were not tested during publication. The 512 manual coding items and the
complete early Qwen2.5 weight-to-revision mapping remain pending. These checks
do not establish a hidden computational mechanism or a universal intervention.

Run the documented commands in the README to obtain verification reports in
your own `.work/` directory. The original manuscript, experiment directories
and scientific archives are preserved outside this separate repository.
