"""Freeze an explicit four-GPU recovery run without editing the original run."""
import argparse
import hashlib
import json
import shutil
import socket
import sqlite3
from contextlib import closing
from pathlib import Path

from common import atomic_json, connect_db, digest, now, require_metadata, validate_frozen


def row_digest(rows):
    payload = json.dumps(sorted(rows), ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def copy_backup(db, target):
    with closing(sqlite3.connect(target)) as dest:
        db.backup(dest)


def freeze(path, config, source, extra_source=()):
    (path / 'source').mkdir()
    for p in source.glob('*.py'):
        shutil.copy2(p, path / 'source' / p.name)
    for p in extra_source:
        shutil.copy2(p, path / 'source' / p.name)
    atomic_json(path / 'config.json', config)
    atomic_json(path / 'frozen.json', {
        'frozen_at': now(), 'config_sha256': digest(path / 'config.json'),
        'manifest_sha256': digest(path / config['manifest']),
        'source_hashes': {p.name: digest(p) for p in sorted((path / 'source').glob('*.py'))},
        'protocol_hashes': {p.name: digest(p) for p in sorted(path.glob('*PROTOCOL.md'))},
        'n_items_per_model': len((path / config['manifest']).read_text(encoding='utf-8').splitlines()),
        'interpretation': 'Documented execution migration of the original finite formal matrix.'})
    validate_frozen(path)


def create_migration(old_run, new_run, code_dir, old_stopped_reason):
    old_run, new_run, code_dir = map(lambda p: Path(p).resolve(), (old_run, new_run, code_dir))
    if not old_stopped_reason.strip():
        raise ValueError('Explicit evidence that the old instance is stopped is required.')
    old_config = validate_frozen(old_run)
    if new_run.exists():
        raise FileExistsError('Refusing to replace an existing recovery run.')
    raw_lines = (old_run / old_config['manifest']).read_text(encoding='utf-8').splitlines(keepends=True)
    manifest = [json.loads(line) for line in raw_lines]
    ids = [r['id'] for r in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate original manifest IDs')
    owner = {item_id: i % 4 for i, item_id in enumerate(ids)}
    models = {m['name'] for m in old_config['models']}
    # The writer must already be stopped. Normal SQLite open also permits recovery
    # of an interrupted transaction's hot journal without deleting journal files.
    original = sqlite3.connect(old_run / 'results.sqlite', timeout=60)
    if original.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        raise ValueError('Original database integrity check failed')
    rows = list(original.execute('SELECT * FROM results'))
    if any(r[1] not in models or r[2] not in owner for r in rows):
        raise ValueError('Original results do not belong to the frozen matrix')
    if len({(r[1], r[2]) for r in rows}) != len(rows):
        raise ValueError('Duplicate source calls')
    metadata = dict(original.execute('SELECT key,value FROM metadata'))
    new_run.mkdir(parents=True)
    copy_backup(original, new_run / 'previous.results.sqlite')
    original.close()
    with closing(sqlite3.connect(new_run / 'previous.results.sqlite')) as saved:
        if row_digest(list(saved.execute('SELECT * FROM results'))) != row_digest(rows):
            raise ValueError('Source changed during snapshot; do not start workers')
    for name in ('FORMAL_PROTOCOL.md', 'PROTOCOL.md'):
        shutil.copy2(old_run / name, new_run / name)
    shutil.copy2(code_dir / 'MIGRATION_PROTOCOL.md', new_run / 'MIGRATION_PROTOCOL.md')
    shutil.copy2(old_run / old_config['manifest'], new_run / 'manifest.jsonl')
    host = {'hostname': socket.gethostname(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    migration = {'created_at': now(), 'source_run': str(old_run), 'target_run': str(new_run),
        'old_instance_stopped_evidence': old_stopped_reason, 'target_host': host,
        'source_frozen_sha256': digest(old_run / 'frozen.json'),
        'source_database_sha256': digest(new_run / 'previous.results.sqlite'),
        'source_result_rows': len(rows), 'source_rows_sha256': row_digest(rows),
        'source_metadata': {key: json.loads(value) for key, value in metadata.items()},
        'partition_rule': 'original manifest row index modulo 4',
        'original_manifest_sha256': digest(new_run / 'manifest.jsonl'),
        'expected_total_records': len(ids) * len(models), 'shards': []}
    (new_run / 'shards').mkdir()
    for index in range(4):
        name = f'shard-{index}'; shard = new_run / 'shards' / name; shard.mkdir()
        with (shard / 'manifest.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            f.writelines(raw_lines[index::4])
        for protocol in new_run.glob('*PROTOCOL.md'):
            shutil.copy2(protocol, shard / protocol.name)
        config = dict(old_config, name=f'{new_run.name}-{name}')
        freeze(shard, config, old_run / 'source')
        db = connect_db(shard / 'results.sqlite')
        selected = [r for r in rows if owner[r[2]] == index]
        with db:
            db.executemany('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)', selected)
        # Keep exact environment/checkpoint provenance checks; new storage/config
        # hashes will be established by the unchanged worker for this new shard.
        for key, value in metadata.items():
            if key == 'environment' or key.startswith('model_provenance:'):
                require_metadata(db, key, json.loads(value))
        require_metadata(db, 'migration_source', {'run': str(old_run), 'rows': len(selected),
            'source_snapshot_sha256': migration['source_database_sha256'],
            'rows_sha256': row_digest(selected)})
        copy_backup(db, shard / 'results.backup.sqlite'); db.close()
        migration['shards'].append({'name': name, 'items': len(raw_lines[index::4]),
            'inherited_rows': len(selected), 'rows_sha256': row_digest(selected)})
    atomic_json(new_run / 'migration.json', migration)
    parent_config = dict(old_config, name=new_run.name, parallel={
        'shards': [{'name': f'shard-{i}', 'run_dir': f'shards/shard-{i}', 'gpu': i} for i in range(4)],
        'max_failures': 3, 'stall_timeout_sec': 5400, 'backup_interval_sec': 300,
        'allowed_host': host}, migration={'source_run': str(old_run),
        'snapshot': 'previous.results.sqlite', 'inherited_rows': len(rows),
        'snapshot_sha256': migration['source_database_sha256'],
        'record_sha256': digest(new_run / 'migration.json')})
    freeze(new_run, parent_config, old_run / 'source',
        [code_dir / name for name in ('parallel_supervise.py','parallel_start.py','parallel_status.py','migrate_parallel.py')])
    db = connect_db(new_run / 'results.sqlite')
    with db:
        db.executemany('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)
    copy_backup(db, new_run / 'results.backup.sqlite'); db.close()
    print(json.dumps({'run': str(new_run), 'inherited_rows': len(rows), 'shards': migration['shards'],
        'manifest_sha256': digest(new_run / 'manifest.jsonl'), 'config_sha256': digest(new_run / 'config.json')}, indent=2))
    return migration


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-run', required=True)
    ap.add_argument('--new-run', required=True)
    ap.add_argument('--old-stopped-reason', required=True)
    args = ap.parse_args()
    create_migration(args.old_run, args.new_run, Path(__file__).parent, args.old_stopped_reason)
