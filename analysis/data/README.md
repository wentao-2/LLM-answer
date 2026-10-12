# Current supplement exports

`review_supplement/` contains six complete summary exports and five unrounded
CSVs: `phi_static`, `wordings`, `history_coverage`, `history_update_errors`, and
`history_presentation`. `provenance.json` identifies original saved analysis and
database hashes. `integrity_audit.json` records the independent original audit.
The source evidence release contains the full per-item residuals and parents.

Use `scripts/export_review_supplement.py --evidence-root PATH --audit PATH`
from the manuscript root, with the complete original evidence and a passed audit.
Then run `build_review_tables.py` and `plot_review_supplement.py`. The main static
Qwen3 rows come from the retained CSV below, while standard Phi-4 uses its own
checkpoint, runtime, and native budget. The four wording points use common
complete sets of twelve reports. History points use complete pairs and represented
families. Full design bounds retain all planned static settings or constructed
history comparisons. Bounds are not confidence intervals.

Self, other, and neutral use blue, gold, and gray. Native history uses dark ink.
The history contrasts follow initial choice C, while evidence residuals follow
new reading B. Signed means and absolute residuals use identical valid pairs.
Incorrect choices and large probability errors are distinct overlapping cohorts.
Qwen3's fresh vLLM histories and retained SDPA histories are not pooled.

## Earlier precision data and renderers

# Figure data and reproduction

`figure_data.json` contains exact, unrounded excerpts from the three frozen analysis files. Each source has its original relative path and SHA256. Contrasts also retain a source section locator, complete and planned matched-set counts, numerical-family means, and the recorded uncertainty or missingness bounds. No raw model output was reparsed.

Run from the manuscript root:

```text
python scripts/plot_results.py
```

Requirements are Python 3, Matplotlib, and the Times New Roman font. The script checks font availability and fails rather than silently replacing it. PDF fonts are embedded. SVG labels remain editable text. All four figures have a fixed 6.3-inch width matching the manuscript text area. Ordinary text is 9.5 or 10 points and panel labels are 10.5 points. The figure heights are 3.35, 4.8, 3.4, and 3.35 inches for attribution, information updates, instructions, and coverage, respectively. PNG previews are written to `build/figure-previews/` and are not manuscript assets.

To repeat the exact extraction from an existing copy of the original evidence tree:

```text
python scripts/plot_results.py --extract-from ../i16-self-evidence
```

This command reads frozen summaries and writes the compact excerpt. It does not run models, reinterpret failed outputs, or estimate new intervals. Source analysis hashes can be checked against the three original files.

## Interpretation required for reuse

- `attribution_diagnosis`: panel (a) shows R1 Qwen3 attribution contrasts for redundant evidence, independent agreement, and independent conflict, with the frozen 95% family bootstrap intervals interpreted as sensitivity to design composition. Counts are complete pairs out of planned pairs. Panel (b) uses the complete R3 independent, legacy, restrictive, likelihood_v3, agreement cell and shows directional normative errors for self and other. Both directional errors are negative while the self minus other gap is positive.
- `information_updates`: all panels show Qwen3-8B in R3 with self attribution, the scoped reminder, and compatible output instructions. Columns retain the existing frequency_v3 and likelihood_v3 strata. Panels (a) and (b) show positive, zero, and negative rules when S is hidden, measured relative to the prior. Panels (c) and (d) show all four rules with S visible, measured as A visible minus A hidden. Blue circles use valid outputs in the upper panels and complete matched pairs in the lower panels. Gray diamonds use the identical valid sets for the normative update. The separate blue lines are the frozen full-design bounds for the **directional update itself**, not its residual. No constant shift, averaging across templates, or recomputation is used. Numerators are valid outputs or complete pairs, and denominators are the corresponding planned sets. In particular, positive S-hidden frequency_v3 retains 23/24 valid outputs, and positive S-visible likelihood_v3 retains 16/24 complete pairs. These target differences account for some points lying outside the full-design bounds. The selected point values, normative values, bounds, and counts are copied exactly from the frozen summary and checked against `hidden_updates.csv` and the template rows of `contrasts.csv` in `results/validation-final-report/`.
- `instruction_effects`: diamonds show Qwen3 estimates from complete eight-cell matched sets. Thick lines on separate vertical positions show bounds for the whole planned finite design with missing probabilities ranging over [0,1]. These are different target sets. Panel (a) is the compatible minus restrictive MAE contrast. Panel (b) is the instruction by attribution interaction in the signed directional effect. R2 and R3 use different grids and readout formats.
- `coverage_bounds`: panel (a) uses all terminal calls in R3. The single invalid Qwen3.5 response has its true proportional width and is almost invisible at print scale. Panel (b) shows Qwen3.5 independent evidence. The attribution gap uses scoped and compatible instructions and complete two-cell pairs. The MAE contrast averages reminder and attribution factors using complete eight-cell sets. Counts are printed next to the corresponding estimates.

The full design bounds in the information, instruction, and coverage figures are not confidence intervals. Complete-set estimates need not lie inside full-design bounds. All differences use probabilities on the 0 to 1 scale. The plotting palette uses blue for self and attribution, orange for other, teal for compatible and valid output, and gray for normative references. Shape and position also distinguish estimands.
