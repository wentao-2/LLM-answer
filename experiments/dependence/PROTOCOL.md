# Dependence response and evidence source intervention

Prospective protocol, registered on 2026-10-11 before any new scientific call. This study extends the retained completed I16 evidence and does not replace or rerun any terminal observation.

## Research questions

Does a model adjust its probability update to the degree of dependence between displayed sensor records? Does an explicit description of observation sources selectively reduce copying error while preserving the use of independent observations?

## Partial dependence

The hidden binary state has prior pi. S and T have the same symmetric reliability r and are conditionally independent given H. D is Bernoulli(rho), independent of H, S and T. B=S if D=1 and B=T otherwise. rho is a copying branch probability, not a correlation coefficient. The model receives the entire rule, rho and S. In a record branch it additionally receives B, without observing D or T separately. The no record branch reveals neither a prospective B value nor agreement.

P(B=b|S=s,H=h)=rho*1[b=s]+(1-rho)*P(T=b|H=h). The exact posterior is obtained by enumerating the joint distribution with rational arithmetic. For agreeing observations the direction aligned log odds increment is log((rho+(1-rho)*r)/(rho+(1-rho)*(1-r))). For conflicting observations and rho<1 its magnitude is log(r/(1-r)), independent of rho, because conflict excludes copying. Conflict at rho=1 has zero probability and is absent from the design.

Each model uses the 24 saved frequency_v3 parents, covering priors 0.18, 0.32, 0.68, 0.82, reliabilities 0.66, 0.78, 0.92, and S=0,1. These are six families defined by complementary prior pairs and reliability. Qwen parents are explanation_on. Phi parents are numeric_native. Selection is based on these fields before inference, without selection on continuation correctness. The saved visible final response is inserted unchanged, as in the earlier history design. Full source records, including raw outputs and token IDs, remain available.

rho={0,0.25,0.5,0.75,1}. There are none, agree, conflict branches at the first four values and none, agree at the last. Each rho and presentation has its own none continuation. Two presentations are native_self and replay_self. Per model this is 24*14*2=672 calls, or 1,344 across both models. New endpoints test the new wording as well as the dependence rule. An endpoint that removes an earlier error is retained as a wording dependent result.

## Evidence source intervention

Use all 48 retained Phi parents and exactly their two original self presentations and four evidence branches: none, copy, independent agree, independent conflict. The original 384 outputs are reused by exact original ID and source SHA. Two extra instruction conditions are inserted in the final task before its probability request. General Bayes asks the model to apply Bayes' theorem according to the supplied probabilities. Source structure asks it to identify which physical observation generated each record and to distinguish documentation of the same observation from a conditionally independent sensor observation. Neither provides H, the posterior, or a correct numerical answer. No weights or training are changed.

The extra 48*2*2*4=768 calls have the same complete saved history and original task except for that single inserted instruction. Each condition uses its own none baseline. This tests the intervention within the recorded checkpoint and interface, without attributing the full native versus replay difference to a particular role token or proposing a new general algorithm.

## Runtime

Standard Phi-4, microsoft/phi-4 revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25, uses the unchanged original Transformers worker, BF16, SDPA, greedy decoding, seed 1601, native template, batch size two, 8,192 input and 8,192 output caps, original EOS and unique-final-json-v1 parser. Its prior mode is native_nonreasoning. Memory reduction follows the unchanged original worker and is logged.

Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218 uses the unchanged completed vLLM 0.8.5 worker and Transformers 4.51.3 template, BF16, thinking enabled, greedy decoding and seed 1601. Input cap is 8,192 and output cap is 12,288, with 20,480 engine context. The original strict readout requires actual EOS and a closed thinking segment. Maximum concurrent sequences 32, batched tokens 8,192, GPU memory fraction 0.88, no prefix caching, chunks of 64. These are within model comparisons with different recorded runtimes, not matched compute model rankings.

Use one worker per card, six Phi shards and two Qwen shards initially. All use the existing OS locks at /tmp/i16-extension-gpu-locks. Original worker and scheduler bytes are copied and frozen. No synthetic scientific answers, numerical repairs, arbitrary probability clipping, guided grammar or selective failure retries are allowed. An administrative recovery may resume only IDs lacking any saved terminal status, with the original record and failure evidence preserved. Weight sizes and complete SHA256 must match the retained inventories before launch. All prompts pass native rendering and exact input/context checks before the first generation.

## Prespecified analysis

The analysis scripts are frozen with the design before inference. Preserve every planned ID, original status, token count, probability and parent provenance. Validity coverage uses the full planned denominator. Posterior absolute error compares to the exact rational posterior. Report per family values and the mean of the six family means, without changing weights because outputs are missing.

For each parent, rho or instruction condition and presentation, pair the evidence output with its own none output. In probability space record signed and absolute update residual relative to the corresponding oracle increment. In log odds space record direction aligned effective independent evidence weight w=(2B-1)*(logit(p_B)-logit(p_none))/log(r/(1-r)), its normative value, and signed and absolute residual. Endpoint probabilities are preserved and separately counted. They are not clipped, and are excluded only from finite log odds estimates. Their probability errors remain included.

Primary partial dependence comparisons are the agreement curve across rho and the conflict curve below rho=1, separately per model and presentation. Report all valid pairs, a common complete parent set across both presentations and all rho/branch cells, and a common set within each rho. Ignoring dependence predicts weight one for both directions. Correct updating predicts decreasing agreeing weight and constant conflicting weight. Uniformly reducing updates suppresses both directions. Evaluate absolute residuals and individual distributions as well as signed averages.

For intervention, compare original, general Bayes and source structure on the same original parent. The primary targets are copy weight and error, independent weight and error, and coverage. Report valid pairs and the common complete parent set across all three conditions, both presentations and all four branches. Calculate paired differences for copy and independent error separately. An intervention reducing both types of update is not selective correction. Preserve actual choice correctness as a separate thresholded metric from probability error.

For bounded probability errors and update residuals, retain all planned cells and calculate deterministic missing output bounds, using known endpoints where available. Log odds have no finite universal missing output bound and are labeled as conditional finite estimates. Per family ranges describe the fixed factorial grid. There are no significance tests or claims of a randomly sampled population. All negative, reverse, endpoint and unestimable results are reported. The oracle is a Bayes reference rather than new probability theory.
