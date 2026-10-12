# Controlled rule expressions and complete observation accounting

This protocol is frozen before any new scientific inference. All prompts, models,
budgets, construction code and analysis code are sealed together. Stage 2 does not
adapt to Stage 1 results. The previous experiment and its scientific records are
read-only inputs.

## Scientific design

H is binary with prior pi. Each physical observation S or T equals H with
probability r. S and T are conditionally independent given H. D is independent of
H, S and T, with P(D=1)=rho. B=S when D=1 and B=T otherwise. Only S and, for
visible branches, B are shown. In the report context C is an exact quotation of
B, so it supplies no additional likelihood factor. No realized D or T is given.
Selection and stopping do not depend on H, values or agreement.

For a visible B the conditional likelihood is
L_h=P(S=s|H=h)[rho 1(b=s)+(1-rho)P(T=b|H=h)].
For none it is L_h=P(S=s|H=h). The posterior is pi L_1 divided by
pi L_1+(1-pi)L_0. Enumeration and factored and odds forms are checked separately.
An impossible conflict at rho=1 is never constructed.

Stage 1 reuses the exact 24 saved frequency_v3 Phi-4 parent answers in native
history. The grid is 24 parents by eight legal rho/branch combinations by two
rule expressions (direct, selector) by two final questions (general, explicit).
The direct expression states full independence or full copying at the endpoints.
The two expressions describe the same generating procedure. No correct posterior,
numerical joint likelihood or realized source selector is supplied. Each cell has
its own none baseline. Total: 768 new calls.

Stage 2 first generates 64 parents: two models by two contexts by two new
reliabilities (0.72, 0.86) by four priors (0.18, 0.32, 0.68, 0.82) by two readings.
All valid parents enter the eight legal branches and four fixed conditions:
ordinary question, general Bayes reminder, explicit conditioning variables, and
observations/sources/joint likelihood steps. There is no accuracy-based selection.
Invalid or truncated parents remain in the planned denominator. Any rendering
failure stops the affected administrative phase and does not silently exclude a
valid parent or shorten its answer. Children use exact saved final visible parent
answers. The report context adds C=B only in branches with B visible. Maximum:
64 parents plus 2,048 children, or 2,112 new calls. Combined maximum: 2,880.

The full steps condition requests an auxiliary WORK_JSON before the unchanged
FINAL_JSON line. It records observed values, source descriptions, source variables
whose realized values are not displayed, and the two joint likelihoods. This is
part of the specified intervention and supplies no numerical answer. Other arms
retain their ordinary visible output. The original FINAL_JSON probability parser
and generation termination classification are used without changes. Intermediate
format failure does not invalidate an otherwise valid final probability.

## Model and execution configuration

Phi-4 revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 uses complete BF16
weights, native Transformers SDPA, greedy decoding, seed 1601, batch size 2, and
input/output limits of 8,192. Qwen3-8B revision
b968826d9c46dd6066d109eabc6255188de91218 uses complete BF16 weights, vLLM
0.8.5, thinking enabled, greedy decoding, seed 1601, input limit 8,192, output
limit 12,288 and context length 20,480. The native templates and package versions
must match the retained inventories. No model, parser, quantization or budget is
replaced. Within-model comparisons are paired. This is not an equal-budget model
ranking.

Six Phi and two Qwen workers use one GPU each, with the existing shared flock
directory. The first queues contain only Stage 2 parents. After a model's entire
parent queue terminates, its parent records, coverage and construction manifest
are sealed before continuation. The Phi continuation queue includes Stage 1 and
Stage 2 children. The Qwen continuation queue contains Stage 2 children. Thus the
stages can overlap without adapting scientific conditions to outcomes. The source
code, queues and parent-to-child mapping are immutable. Status uses /proc command
identity and start ticks together with state and terminal files.

All output statuses are saved. Invalid, truncated and error records are not
selectively rerun. Complete phase databases alone enter final analysis. Live
shards are used for progress only.

## Frozen analysis

Primary outcomes are posterior absolute error and signed/absolute probability
update residual relative to each condition's own none. Effective independent
evidence weight is (2b-1)[logit(p_b)-logit(p_none)]/log(r/(1-r)). Normative agree
weight declines with rho, while normative conflict weight equals one for rho<1.
Probability endpoints are counted and never clipped. Logit outcomes have no finite
universal missing bound.

Report all planned calls and constructed children, parent coverage by model,
context, reliability, prior and direction, scientific status counts, complete
probability pairs, and parents complete across all 32 continuation cells. Means
use equal numerical-family weights (r by complementary-prior pair). Report each
family's mean and range. Also report context, model, S direction and each prior
and reliability. Do not pool duplicated rho observations as independent samples.
This is a fixed synthetic grid with descriptive estimates, no population
significance claim or preregistered positive result.

Condition contrasts are paired on the same parent, branch, rho and rule. Stage 1
contrasts change only rule expression or final question, with difference-of-
differences included. Stage 2 compares each intervention to ordinary and full
steps to both other interventions. A successful improvement must reduce copying
and mixed-rule error while retaining independent agree/conflict updates. Suppress-
all-updates behavior is not success. Contrasts use common valid records and retain
identified probability-error and update-residual bounds for missing outputs.

For a missing probability p, p lies in [0,1]. Posterior-error bounds are
[0,max(q,1-q)]. Update bounds combine the available endpoints without clipping.
Report the constructed-child cohort as well as all planned-parent bounds. A
missing parent gives its 32 intended children the same known design/oracle bounds
without inventing a response, prompt history or scientific result ID.

Reference comparisons retain continuous error to the correct joint posterior,
S-only, B-only and two-independent-observation posteriors, plus thresholds 0.0001
and 0.0005 and every coincident match. These describe output behavior and do not
identify internal computation. Full-steps WORK_JSON is parsed separately using
unique JSON keys. Observed-value correctness, two likelihood errors and
not-displayed source variables are reported separately from final validity.
Free-text source statements are exported for manual coding with a frozen rubric:
retain S, include B when shown, preserve probabilistic origin when 0<rho<1,
preserve independence at rho=0, preserve copying at rho=1, and identify C as a
quotation of B. No automatic source-text label is inferred from final probability.
Manual codes require a quoted visible span and can be missing or ambiguous.

Retain generated token counts, elapsed times, package versions, prompt/token
rendering hashes, complete weight SHA checks, parent record/final-answer hashes,
and exact expected ID sets. Analyze and independently audit after both model
continuation queues terminate. These tasks test controlled synthetic settings,
not natural open-domain transfer or a hidden unified model mechanism.
