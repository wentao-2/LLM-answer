"""One bounded, read-only snapshot of shard progress and local process identity."""
import argparse
import json
from contextlib import closing
from pathlib import Path
from parallel_supervise import connect_readonly, host_identity, process_identity, read_json


def database_status(path):
    path = Path(path)
    if not path.exists():
        return {'exists': False}
    with closing(connect_readonly(path)) as db:
        return {'exists': True, 'counts': [dict(zip(('model', 'status', 'n'), row)) for row in
                    db.execute('SELECT model,status,COUNT(*) FROM results GROUP BY model,status')],
                'last_result': db.execute('SELECT MAX(created_at) FROM results').fetchone()[0],
                'saved': db.execute('SELECT COUNT(*) FROM results').fetchone()[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = read_json(run_dir / 'config.json')
    state = read_json(run_dir / 'state.json', {})
    result = {'local_host': host_identity(), 'state.json': state,
              'aggregation.json': read_json(run_dir / 'aggregation.json'), 'shards': {},
              'database': database_status(run_dir / 'results.sqlite')}
    result['supervisor_identity'] = process_identity(state.get('supervisor_pid'),
        run_dir / 'source' / 'parallel_supervise.py', run_dir / 'config.json', state.get('host'))
    saved = 0
    for spec in config['parallel']['shards']:
        shard = (run_dir / spec['run_dir']).resolve()
        entry = state.get('shards', {}).get(spec['name'], read_json(shard / 'state.json', {}))
        database = database_status(shard / 'results.sqlite')
        saved += database.get('saved', 0)
        result['shards'][spec['name']] = {'gpu': spec['gpu'], 'state': entry,
            'progress': read_json(shard / 'progress.json'), 'database': database,
            'backup_exists': (shard / 'results.backup.sqlite').exists(),
            'worker_identity': process_identity(entry.get('worker_pid'), shard / 'source' / 'run.py',
                shard / 'config.json', entry.get('host'))}
    result['live_saved_total'] = saved
    result['expected_total'] = len(config['models']) * sum(1 for line in
        (run_dir / config['manifest']).read_text(encoding='utf-8').splitlines() if line)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
