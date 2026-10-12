# Recorded environments and fresh inference

## Saved-output analysis

Python 3.12 or newer is sufficient for all three released offline verifiers.
No pip installation, external checkpoint or GPU is required. Plotting was tested
with NumPy 2.3.5, Matplotlib 3.11.2 and Pillow 12.3.0. Those versions are listed
in `requirements-figures.txt`. The plotting sources require Times New Roman and
fail explicitly if it is unavailable.

## Recorded model execution

The experiments were run on Linux with NVIDIA GPUs. Each stage retains its
own runtime and checkpoint metadata. There is no single environment lock for
all historical stages.

The newer Qwen3-8B followups and wording comparisons use vLLM 0.8.5, PyTorch
2.6.0 and Transformers 4.51.3, with BF16 and greedy decoding. Their initial
producers use the retained SDPA worker. The runtime distinction is part of the
experimental record, not an assumed equivalence. Standard Phi-4 uses its native
template and SDPA runner and is distinct from Phi-4-reasoning.

The latest dependence and observation-control directories contain recorded
`inputs/runtime_environments.json` and `inputs/runtime_configs.json`. Full
checkpoint inventories, prompt templates, generation caps and terminal records
are in the corresponding released evidence. The extension configuration is
`experiments/extension/base-config.json`. The wording/history source retains
its actual worker preparation and execution settings.

Representative checkpoint identities are:

| Checkpoint | Revision |
| --- | --- |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Standard Phi-4 | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` |
| Phi-4-reasoning | `1de18ec97600877ce63dbf60c73b998da99f0195` |
| Mistral-Nemo-Instruct-2407 | `04d8a90549d23fc6bd7f642064003592df51e9b3` |

Use the specific stage configuration and inventory rather than choosing a new
revision, model family or quantization. Model weights are separately obtained
from their official providers and governed by the corresponding licenses.

## Starting another inference run

The published frozen workers expose their recorded configuration interfaces,
including `experiments/extension/source/run_extension.py` and
`experiments/wording_history/vllm_worker.py`. They are historical implementations.
Prepare a new execution directory, obtain the correct checkpoint, install the
matching CUDA/PyTorch/Transformers or vLLM stack, and specify local model paths
in a separate configuration. Preserve the native template, final parser and
scientific token cap of the chosen stage.

No fresh inference was performed while organizing this public repository.
The published quick-start verifies saved outputs and should be used before
attempting another GPU execution. Existing terminal scientific records should
not be resumed with modified settings.
