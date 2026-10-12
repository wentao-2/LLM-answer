"""Prepare an immutable run before looking at its model outputs."""
import argparse
import json
import shutil
from pathlib import Path
from common import atomic_json, digest, now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--phase', choices=['pilot', 'formal'], default='pilot')
    ap.add_argument('--models', default='qwen25-14b-instruct,mistral-7b-instruct-v01')
    ap.add_argument('--max-new-tokens', type=int, default=96)
    ap.add_argument('--protocol', help='Additional protocol file frozen with this run')
    args = ap.parse_args()
    target = Path(args.run_dir).resolve()
    if target.exists():
        raise SystemExit('Refusing to overwrite an existing run directory.')
    target.mkdir(parents=True)
    source = Path(__file__).parent
    (target / 'source').mkdir()
    for path in source.glob('*.py'):
        shutil.copy2(path, target / 'source' / path.name)
    shutil.copy2(source / 'PROTOCOL.md', target / 'PROTOCOL.md')
    if args.phase == 'formal':
        shutil.copy2(source / 'FORMAL_PROTOCOL.md', target / 'FORMAL_PROTOCOL.md')
    if args.protocol:
        shutil.copy2(Path(args.protocol), target / 'EXPERIMENT_PROTOCOL.md')
    raw_manifest = Path(args.manifest).read_text(encoding='utf-8')
    if args.smoke:
        rows = [json.loads(line) for line in raw_manifest.splitlines()]
        selected = []
        seen = set()
        for row in rows:
            key = (row['stage'], row['condition'], row['attribution'])
            if key not in seen:
                selected.append(dict(row, split='smoke'))
                seen.add(key)
        raw_manifest = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected)
    (target / 'manifest.jsonl').write_text(raw_manifest, encoding='utf-8', newline='\n')
    config = {
        'name': target.name, 'phase': 'smoke' if args.smoke else args.phase, 'seed': 1601,
        'manifest': 'manifest.jsonl', 'batch_size': 4, 'cpu_threads': 8,
        'dtype': 'bfloat16', 'do_sample': False, 'num_beams': 1,
        'max_input_tokens': 1536, 'max_new_tokens': args.max_new_tokens, 'attention': 'sdpa',
        'analysis_script': 'analyze_formal.py' if args.phase == 'formal' else 'analyze.py',
        'stall_timeout_sec': 5400,
        'models': [
            {'name':'qwen25-14b-instruct', 'path':'/workspace/models/Qwen/Qwen2.5-14B-Instruct',
             'mode':'instruct-direct-json', 'chat_template_kwargs':{}},
            {'name':'mistral-7b-instruct-v01', 'path':'/workspace/models/lto-table2/Mistral-7B-Instruct-v0.1',
             'mode':'instruct-direct-json', 'chat_template_kwargs':{}}
        ]}
    available = {m['name']: m for m in config['models']}
    available['qwen3-8b'] = {'name':'qwen3-8b', 'path':'/workspace/models/Qwen/Qwen3-8B',
        'mode':'nonthinking-explicit-false-direct-json', 'chat_template_kwargs':{'enable_thinking':False}}
    available['qwen3-8b-thinking'] = {'name':'qwen3-8b-thinking', 'path':'/workspace/models/Qwen/Qwen3-8B',
        'mode':'thinking-explicit-true-final-json', 'chat_template_kwargs':{'enable_thinking':True}}
    config['models'] = [available[name] for name in args.models.split(',')]
    if len(config['models']) != len({m['name'] for m in config['models']}):
        raise ValueError('Duplicate models')
    atomic_json(target / 'config.json', config)
    atomic_json(target / 'frozen.json', {'frozen_at': now(), 'manifest_sha256': digest(target/'manifest.jsonl'),
        'config_sha256': digest(target/'config.json'),
        'source_hashes': {p.name:digest(p) for p in sorted((target/'source').glob('*.py'))},
        'protocol_hashes': {p.name:digest(p) for p in sorted(target.glob('*PROTOCOL.md'))},
        'interpretation':'Exploratory finite design; no claims about actual generated self-history.',
        'n_items_per_model': len(raw_manifest.splitlines())})
    print(json.dumps({'run_dir':str(target), 'n_items':len(raw_manifest.splitlines())}))


if __name__ == '__main__':
    main()
