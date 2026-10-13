# LLM Answer

Experiment, analysis and visualization code for **When Does “My Answer” Matter?
Authorship, Evidence Dependence, and Probability Updating**.

The experiments distinguish an authorship preference from posterior error,
test how much information a later answer should add, and evaluate requests
that explicitly retain all observed information. The repository includes the
positive results, reversed effects, invalid reports, truncations and missingness
calculations used in the manuscript.

## Quick start

Saved-output verification requires **Python 3.12 or newer**, with no additional
Python packages, model weights or GPU.

```sh
git clone https://github.com/wentao-2/LLM-answer.git
cd LLM-answer
python scripts/check_repository.py
python scripts/reproduce.py --suite all
```

The three exact archives are available in the
[anonymous evidence repository](https://anonymous.4open.science/r/LLM-answer-442D/)
under `evidence/`. The last command uses the included archives, or downloads
them from that anonymous repository if they are absent, checks their sizes and
SHA256 identities, extracts them into `.work/`, and runs their frozen offline
scientific verifiers. The archives total about 148 MB and
expand to about 1.15 GB. Verification writes reports under
`.work/offline/reports/` and does not run model inference.

To check only the latest dependence and observation-control experiments:

```sh
python scripts/reproduce.py --suite dependence-followup
```

For already downloaded archives:

```sh
python scripts/reproduce.py --suite all --offline --archives-dir PATH_TO_ARCHIVES
```

See [reproducibility](docs/reproducibility.md) for archive membership, tolerances
and output interpretation.

## Repository layout

| Directory | Contents |
| --- | --- |
| `experiments/` | Frozen generators, inference workers, analyzers, protocols and configuration snapshots, grouped by scientific purpose. |
| `analysis/scripts/` | Manuscript export, table and figure sources. |
| `analysis/data/` | Exact numerical exports and complete visible examples used by the figure sources. |
| `analysis/figures/` | Vector PDF, editable SVG, plotting provenance and editable draw.io diagrams. |
| `analysis/sections/` | Generated numerical LaTeX tables. |
| `scripts/` | Portable verification, plotting and table entry points. |
| `evidence/` | The three complete evidence ZIPs, with unchanged bytes. |
| `reproducibility/` | Anonymous archive download URLs, exact sizes and SHA256 identities. |
| `provenance/` | File-by-file source-copy identities. |
| `docs/` | Experiment map, environments and reproduction instructions. |

The complete databases and terminal records are included as ZIPs under
`evidence/`, so anonymous repository mirrors can serve ordinary file paths.
Original archive filenames and internal record identifiers are retained to
preserve their hash links.

## Figures and tables

Install the tested plotting dependencies and an appropriately licensed
Times New Roman font:

```sh
python -m pip install -r requirements-figures.txt
python scripts/replot.py
python scripts/rebuild_tables.py
```

The plotting command regenerates the four current dependence and observation
figures in `.work/plots/`. It checks every numerical point against the released
exports. The table command writes a separate `.work/tables/` workspace and
compares the tables with the released versions. Existing scientific sources
and figures are preserved. Use a new `--output-dir` for another rendering.
The table command rebuilds six earlier result tables and five latest dependence
and observation-control tables. Run the saved-output verification first so the
underlying scientific analyses have been checked. The table renderers use the
released unrounded numerical exports.

Earlier plot renderers are retained in `analysis/scripts/`. The overview diagram
is editable in `analysis/figures/fig1-attribution-overview.drawio`.
Figures use Times New Roman, panel labels below the panels, embedded PDF fonts
and editable SVG text. Family ranges are descriptive ranges, not confidence
intervals.

## Experimental code

[The experiment map](docs/experiments.md) connects each source directory to its
complete released evidence and frozen configuration. The fresh-inference
requirements differ across stages. Recorded environments and model revisions
are described in [environments](docs/environments.md).

Saved-output reanalysis and fresh model inference are different tasks. The
release verifies the retained numerical analyses. Fresh inference requires
the specified external checkpoints, native templates, token budgets and Linux
GPU runtimes. It must use new output directories and must not replace terminal
scientific records. Checkpoints and Times New Roman font files are not included.

## Scientific reporting

- Authorship gaps, signed probability errors and absolute errors evaluate
  different quantities. Comparisons use their documented complete pairs and
  equal weights across represented numerical families.
- Missing outputs retain their original statuses and remain missing. Full
  design bounds and finite log-odds weights have different denominators.
- Explicit observation requests improve Phi-4 in the tested selector settings.
  Direct expression has a reversed effect. Qwen3's valid pairs improve, while
  full planned bounds cross zero. General reminders and full steps do not
  provide stable correction across the tested conditions.
- The 512 source-description coding items are pending. Behavioral reference
  matches are not treated as an identified hidden computation.

The copied analysis sources correspond to manuscript source commit
`f265aa7a7bdea53b95958a32f2c09eabafa2f69b`. The publication does not include
review reports or machine administration logs.

The checks actually completed during publication are listed in
[validation](VALIDATION.md).

## License

The repository retains its existing [MIT license](LICENSE). External model
weights, dependencies and fonts remain subject to their respective licenses.
See [third-party materials](THIRD_PARTY.md).
