"""Prepare a separate engineering run. These records never enter scientific analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=['qwen3', 'phi4', 'nemo'], required=True)
    args = parser.parse_args()
    specs = {
        'qwen3': {'name': 'qwen3-8b-thinking', 'path': '/workspace/models/Qwen/Qwen3-8B',
                  'chat_template_kwargs': {'enable_thinking': True}, 'require_think_close': True},
        'phi4': {'name': 'phi4-reasoning', 'path': '/workspace/models/microsoft/Phi-4-reasoning',
                 'require_think_close': True, 'revision': '1de18ec97600877ce63dbf60c73b998da99f0195'},
        'nemo': {'name': 'mistral-nemo-instruct', 'path': '/workspace/models/mistralai/Mistral-Nemo-Instruct-2407',
                 'require_think_close': False, 'revision': '04d8a90549d23fc6bd7f642064003592df51e9b3'},
    }
    dest = args.root / 'engineering' / args.model
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'source').mkdir()
    shutil.copy2(Path(__file__).with_name('run_extension.py'), dest / 'source/run_extension.py')
    prompt = 'Engineering interface check. Return exactly the following final line.\nFINAL_JSON: {"p_h1":0.5}'
    parent = 'FINAL_JSON: {"p_h1":0.5}'
    rows = [{'id': 'engineering_plain', 'prompt': prompt},
            {'id': 'engineering_history', 'messages': [{'role': 'user', 'content': prompt},
               {'role': 'assistant', 'content': parent}, {'role': 'user', 'content': prompt}]}]
    if args.model == 'qwen3':
        rows.append({'id': 'engineering_native_off', 'prompt': prompt,
                     'chat_template_kwargs': {'enable_thinking': False}, 'require_think_close': False})
    (dest / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    (dest / 'ENGINEERING.md').write_text('Loader, template and history interface smoke. Outputs are excluded from all scientific evidence. Cap 128 is intentionally only an execution check.\n', encoding='utf-8')
    config = {'manifest': 'manifest.jsonl', 'experiment_protocol': 'ENGINEERING.md', 'seed': 1601,
              'batch_size': 2, 'max_input_tokens': 8192, 'max_new_tokens': 128,
              'models': [specs[args.model]], 'dtype': 'bfloat16', 'attention': 'sdpa',
              'store_token_ids': True}
    (dest / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    freeze = {'config_sha256': sha(dest / 'config.json'), 'manifest_sha256': sha(dest / 'manifest.jsonl'),
              'source_hashes': {'run_extension.py': sha(dest / 'source/run_extension.py')},
              'protocol_hashes': {'ENGINEERING.md': sha(dest / 'ENGINEERING.md')}}
    (dest / 'frozen.json').write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
    print(dest)


if __name__ == '__main__':
    main()
