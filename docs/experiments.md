# Experiment map

The directory names describe the scientific purpose. Historical record IDs
remain in the source and archives because they bind prompts, results and
frozen analysis identities.

| Public source directory | Question or function | Complete evidence location |
| --- | --- | --- |
| `experiments/static/` | Matched authorship labels and exact posterior targets in the initial factorial study. | Extension archive, `legacy/R1/`. |
| `experiments/rule_controls/` | Evidence rules, visibility and authorship controls. | Extension archive, `legacy/R2/`, with qualification in `legacy/R2-gate/`. |
| `experiments/calibration/` | Calibration and attributed comparisons with the later numerical grid. | Extension archive, `legacy/R3/`, with qualification in `legacy/R3-gate/`. |
| `experiments/extension/` | Native output configuration, qualification and generated-answer history. | Extension archive, `R4/`. |
| `experiments/wording_history/` | Equivalent wordings, answer-quality cohorts and standard Phi-4. | Wording/history archive, `snapshot/` and `audit_supplement.py`. |
| `experiments/dependence/` | Partial copying probability and Bayes/source reminder controls. | Dependence/followup archive, `dependence/`. |
| `experiments/observation_control/` | Source expression, conditioning requests and new settings. | Dependence/followup archive, `followup/`. |

The sources are selected copies from the already verified evidence packages.
`provenance/source_files.json` records each original archive member or manuscript
Git blob, its source SHA256, and the public copy SHA256. Selected experiment
views replace the historical administrative mount with `/workspace`. The
transformation is explicit and changes no scientific calculation. Full evidence
archives retain their verified frozen bytes and original hash links.
These visible source directories are for inspection
and implementation reuse. The full evidence packages retain the complete
input and audit closure required by the frozen analyses.

## Completed designs

- The extension archive includes 18,120 legacy observations and the 3,344 new
  calls of its five completed extension queues. The R3 reference is reused
  evidence. Both fixed extension candidates fail their actual qualification
  gates, and unconstructed downstream comparisons remain absent.
- The wording/history archive includes 7,456 calls in seven complete stages.
  It also preserves the 160 observed calls from two stopped pilots, excluded
  from the completed estimates. Standard Phi-4 and Phi-4-reasoning are separate
  checkpoints.
- The dependence experiment includes 2,112 new calls and 384 reused controls.
  The observation-control experiment includes 2,880 new calls. Invalid outputs,
  truncations, null probabilities and all planned denominators are retained.

## Protocol and implementation

`PROTOCOL.md`, `EXPERIMENT_PROTOCOL.md`, `DESIGN_SEAL.json`, and the available
configuration snapshots describe each recorded stage. Generators and rational
posterior calculations are beside their corresponding analyzers. Inference
workers are retained with their actual parser and terminal-status rules.
The latest observation-control directory also includes the documented auxiliary
WORK-reader recovery. That guard leaves final truncated probabilities missing.

Fresh executions require a new working tree, a local model path, and a newly
recorded configuration. Do not edit the historical freeze or silently retry
invalid/truncated outputs. The recorded model and runtime distinctions are
preserved rather than pooled as a single inference system.
