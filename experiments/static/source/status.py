import argparse
import json
import sqlite3
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
args = ap.parse_args()
path = Path(args.run_dir)
out = {}
for name in ('state.json', 'progress.json'):
    if (path / name).exists():
        out[name] = json.loads((path / name).read_text(encoding='utf-8'))
out['process_checks'] = {}
for label, pid in (('supervisor', out.get('state.json', {}).get('supervisor_pid')),
                   ('worker', out.get('state.json', {}).get('worker_pid'))):
    identity = {'pid': pid, 'alive': False, 'matches_this_run': False}
    if isinstance(pid, int) and pid > 1:
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().decode('utf-8').split('\0')
            expected_script = 'supervise.py' if label == 'supervisor' else 'run.py'
            identity.update(alive=bool(command[0]), command=command,
                matches_this_run=str(path.resolve() / 'config.json') in command and
                    any(arg.endswith('/' + expected_script) for arg in command))
        except (FileNotFoundError, ProcessLookupError):
            pass
    out['process_checks'][label] = identity
if (path / 'results.sqlite').exists():
    db = sqlite3.connect('file:' + str(path / 'results.sqlite') + '?mode=ro', uri=True, timeout=30)
    out['counts'] = [dict(zip(('model','status','n'), r)) for r in db.execute(
        'SELECT model,status,COUNT(*) FROM results GROUP BY model,status')]
    out['last_result'] = db.execute('SELECT MAX(created_at) FROM results').fetchone()[0]
print(json.dumps(out, ensure_ascii=False, indent=2))
