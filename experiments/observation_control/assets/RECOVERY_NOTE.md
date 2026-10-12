# Postprocessing recovery, 2026-10-12

All 2,880 constructed calls terminated before this recovery. The inference
controller recorded successful completion at 2026-10-11 20:25:31 UTC. Its
subsequent analysis failed because the auxiliary WORK_JSON reader passed a null
final_output to a regular expression. The affected 39 full_steps records are
Qwen truncated outputs without a closed thinking segment. They have no visible
final answer and no visible WORK_JSON.

The frozen design, source files, generation settings, final probability parser,
status classifications and scientific databases remain unchanged. The original
analysis failure, controller terminal and logs remain available. No inference is
repeated. A separate recovery wrapper converts null final_output to an empty
string in a temporary copy solely for the optional WORK_JSON reader. It checks
that these records have a missing probability and non-ok scientific status.
Their auxiliary work schema is missing, their probability remains missing, and
the frozen missing-output bounds and denominators apply unchanged. For every
non-null final_output the wrapper must return exactly the original WORK reader's
result. It does not extract answers from a thinking segment or raw_output.

The original analysis is called with this one guarded function replacement in
memory. The original independent audit then runs unchanged. New recovery logs,
terminals, database hashes and final seals identify the recovery explicitly.
The original administrative_failure terminal is not replaced. The separate
pack action accepts only a successful recovery, the original completed inference,
the independent audit, and unchanged design and result database hashes.

This change implements the existing protocol requirement that an intermediate
format failure must not invalidate an otherwise valid final probability. It
does not alter a scientific metric, comparison, selection rule or output.
Free-text source coding remains pending and is not inferred from probabilities.
