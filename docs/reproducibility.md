# Offline reproduction

## Entry points

Run from the repository root with Python 3.12 or newer:

```sh
python scripts/check_repository.py
python -m unittest discover -s tests -v
python scripts/reproduce.py --suite all
```

`check_repository.py` checks every selected source-copy byte identity, parses
the Python files, and verifies the current figure source and input hashes.
The unit checks cover escaping archive paths, modified evidence and refusal
to overwrite an existing extraction.

`reproduce.py` checks archive SHA256 values before extracting and invoking the
complete package verifiers. It uses only the standard library and runs no
model inference. Downloaded archives are cached in `.work/archives/`.
Extracted packages and verification reports are in `.work/offline/`.
Failed extraction does not overwrite an existing directory. For a different
attempt use a new `--work-dir`, retaining the failed evidence for inspection.

## Exact archives

| Suite | Archive | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `extension` | `supplementary-extension-20261010.zip` | 61,100,150 | `d57e3687fa45bf079af2f12d928a75ca64d85d81649d2b691a30a4118f981244` |
| `wording-history` | `supplementary-review-supplement-20261011.zip` | 43,422,751 | `3a617bcd361ff131ec1a17b8e62894e99460d156adf4e2b222ea0d17f7be4ebd` |
| `dependence-followup` | `supplementary-dependence-followup-20261012.zip` | 43,542,064 | `7ebfbc89bb30ba8267997359276e88b0ae58f1e37a434a7a8848fc01efed510d` |

The archives are included in this repository under
[`evidence/`](../evidence/README.md).
The URLs and sizes are machine-readable in `reproducibility/archives.json`.
The reproduction helper first checks the download cache, then uses a bundled
archive if available, and otherwise downloads its exact GitHub file URL.
Each full archive includes its own scientific verifier and a member identity
ledger. The verifiers have different scopes and must be run independently.

The extension verifier recomputes legacy and extension analyses and checks the
qualification-dependent construction of histories. The wording/history verifier
checks seven complete stages and their independent numerical audit. The latest
verifier reproduces both additional frozen designs and 14 CSVs, including the
documented auxiliary null guard.

## Interpreting verification

The saved raw and visible outputs, token IDs, probabilities, final status,
scientific metadata, record IDs and model revisions remain retained. Exact
file identities are checked before numerical comparison. Administrative paths
in the portable archives have explicitly recorded original and package hashes.
The latest two-design verifier permits absolute floating-point differences
of at most `1e-12` between operating systems. Integer counts, strings, IDs,
statuses and key sets must match exactly. Earlier verifiers retain their own
stricter scientific comparisons.

Frozen README statements that describe a local-only package or an unspecified
license are historical assembly metadata. This repository supplies the public
host and retains its existing MIT license. The frozen package bytes and their
hash links are not rewritten to update those historical statements.

Offline reproduction verifies the analysis of retained outputs. It does not
redownload external weights, repeat tokenizer decoding, or establish identical
future generated tokens. Early Qwen3 and Qwen3.5 revisions have been recovered
and their 30 inventory files fully hashed. Qwen2.5's full weight-to-revision
mapping remains partly unverified. Source-description manual coding remains
pending and is not used for a hidden-mechanism claim.

## Plotting and numerical tables

```sh
python -m pip install -r requirements-figures.txt
python scripts/replot.py --output-dir .work/plots
python scripts/rebuild_tables.py --output-dir .work/tables
```

Times New Roman must be installed independently. The figure command writes
vector PDF, editable SVG, PNG previews and numerical provenance in a new
workspace and checks the regenerated points against the released figure data.
The table command similarly writes a separate workspace and compares every
table character after normalizing operating-system line endings. PDF metadata and SVG
object IDs may differ across renderings while the scientific points remain
identical. Both commands preserve the files under `analysis/`.

The table entry rebuilds six earlier result tables and the five latest tables
using their frozen renderers and released unrounded exports. Its verification
report records the latest numerical input hash. The independent saved-output
verifiers perform the underlying scientific reanalysis separately.
