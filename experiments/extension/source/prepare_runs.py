"""Create a new immutable static inference queue. Existing directories are refused."""
import argparse
import json
from pathlib import Path
import re
import shutil

import run_extension as runner


def source_hashes(directory):
    directory = Path(directory)
    return {path.relative_to(directory).as_posix(): runner.digest(path)
            for path in sorted(directory.rglob('*.py'))}


def copy_sources(source_dir, destination):
    destination.mkdir()
    for source in sorted(Path(source_dir).rglob('*.py')):
        target = destination / source.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def write_manifest(path, items):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for item in items:
            stream.write(runner.canonical_json(item) + '\n')


def partition_rows(rows, config, model_spec, count):
    """Balance every native mode/budget across shards without changing items."""
    count = min(runner.positive_int('shards_per_model', count), len(rows))
    if not count:
        return []
    shards = [[] for _ in range(count)]
    groups = runner.group_profiles(rows, config, model_spec)
    groups.sort(key=lambda group: runner.canonical_json(group[0].metadata()))
    for group_index, (_, items) in enumerate(groups):
        ordered = sorted(items, key=lambda item: (runner.text_digest(item['id']), item['id']))
        for index, item in enumerate(ordered):
            shards[(group_index + index) % count].append(item)
    return [shard for shard in shards if shard]


def prepare(config_path, manifests, protocol_path, output, shards_per_model=8, models=None, source_dir=None):
    config_path, protocol_path, output = Path(config_path), Path(protocol_path), Path(output).resolve()
    source_dir = Path(source_dir or Path(__file__).parent).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    specs = config.get('models', [])
    names = [spec['name'] for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError('Model names must be unique')
    if models:
        unknown = set(models) - set(names)
        if unknown:
            raise ValueError(f'Models absent from config: {sorted(unknown)}')
        specs = [spec for spec in specs if spec['name'] in models]
    if not specs:
        raise ValueError('No model specs selected')
    for required in ('run_extension.py', 'prepare_runs.py', 'schedule_runs.py'):
        if not (source_dir / required).is_file():
            raise ValueError(f'Missing queue source: {required}')
    records = []
    manifest_provenance = []
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        records.extend(json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip())
        manifest_provenance.append({'path': str(manifest), 'sha256': runner.digest(manifest)})
    planned = []
    for model_spec in specs:
        selected = runner.select_rows(records, model_spec['name'])
        for item in selected:
            runner.effective_profile(config, model_spec, item)
            runner.item_messages(item)
        if selected:
            planned.append((model_spec, partition_rows(selected, config, model_spec, shards_per_model)))
    if not planned:
        raise ValueError('No manifest items match the selected models')
    if not protocol_path.is_file():
        raise ValueError('Protocol file does not exist')
    # Exclusive directory creation claims the output. A preparation interrupted
    # before queue.frozen.json is written stays visibly incomplete and cannot run.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    copy_sources(source_dir, output / 'source')
    inputs = output / 'inputs'
    inputs.mkdir()
    shutil.copyfile(config_path, inputs / 'base_config.json')
    shutil.copyfile(protocol_path, inputs / 'EXPERIMENT_PROTOCOL.md')
    for index, manifest in enumerate(manifests):
        shutil.copyfile(manifest, inputs / f'manifest-{index:02d}.jsonl')
    jobs = []
    for shard_index in range(max(len(shards) for _, shards in planned)):
        for model_index, (spec, shards) in enumerate(planned):
            if shard_index >= len(shards):
                continue
            items = shards[shard_index]
            slug = re.sub(r'[^A-Za-z0-9._-]', '-', spec['name'])[:60].strip('.-') or 'model'
            job_id = f'm{model_index:02d}-{slug}-s{shard_index:02d}'
            run_dir = output / 'runs' / job_id
            run_dir.mkdir(parents=True, exist_ok=False)
            copy_sources(output / 'source', run_dir / 'source')
            shutil.copyfile(protocol_path, run_dir / 'EXPERIMENT_PROTOCOL.md')
            write_manifest(run_dir / 'manifest.jsonl', items)
            worker_config = dict(config)
            worker_config.update(name=job_id, models=[spec], manifest='manifest.jsonl',
                                 experiment_protocol='EXPERIMENT_PROTOCOL.md')
            # Scheduler-specific legacy fields never alter this single-worker run.
            worker_config.pop('parallel', None)
            runner.atomic_json(run_dir / 'config.json', worker_config)
            frozen = {
                'frozen_at': runner.now(), 'config_sha256': runner.digest(run_dir / 'config.json'),
                'manifest_sha256': runner.digest(run_dir / 'manifest.jsonl'),
                'source_hashes': source_hashes(run_dir / 'source'),
                'protocol_hashes': {'EXPERIMENT_PROTOCOL.md': runner.digest(run_dir / 'EXPERIMENT_PROTOCOL.md')},
                'n_items_per_model': len(items),
            }
            runner.atomic_json(run_dir / 'frozen.json', frozen)
            runner.validate_frozen(run_dir / 'config.json', run_dir / 'source/run_extension.py')
            jobs.append({'id': job_id, 'model': spec['name'], 'run_dir': run_dir.relative_to(output).as_posix(),
                         'expected_item_ids': sorted(item['id'] for item in items), 'count': len(items),
                         'frozen_sha256': runner.digest(run_dir / 'frozen.json')})
    queue = {
        'schema_version': 1, 'created_at': runner.now(), 'jobs': jobs,
        'partition_rule': 'Within each effective generation profile, sort by SHA256(item id), then round robin across shards with profile-index offset.',
        'input_manifests': manifest_provenance, 'base_config_sha256': runner.digest(inputs / 'base_config.json'),
        'expected_records': sum(job['count'] for job in jobs),
        'failure_policy': 'Terminal nonzero workers remain failed. Scheduler restarts resume only nonterminal jobs under identical frozen artifacts.',
    }
    runner.atomic_json(output / 'queue.json', queue)
    runner.atomic_json(output / 'queue.frozen.json', {
        'queue_sha256': runner.digest(output / 'queue.json'),
        'source_hashes': source_hashes(output / 'source'),
        'input_hashes': {path.relative_to(output).as_posix(): runner.digest(path) for path in sorted(inputs.iterdir())},
    })
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True, action='append')
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shards-per-model', type=int, default=8)
    parser.add_argument('--models', help='Comma separated exact model names')
    parser.add_argument('--source-dir', default=str(Path(__file__).parent))
    args = parser.parse_args()
    queue = prepare(args.config, args.manifest, args.protocol, args.output, args.shards_per_model,
                    args.models.split(',') if args.models else None, args.source_dir)
    print(json.dumps({'queue': str(Path(args.output).resolve()), 'jobs': len(queue['jobs']),
                      'expected_records': queue['expected_records']}))


if __name__ == '__main__':
    main()
