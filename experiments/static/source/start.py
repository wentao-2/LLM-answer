"""Safe detached launch: run snapshots contain immutable code and config."""
import argparse
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--recover', help='Documented diagnosis/fix allowing a retry budget reset')
args = ap.parse_args()
run_dir = Path(args.run_dir).resolve()
if not (run_dir / 'config.json').is_file() or not (run_dir / 'source' / 'supervise.py').is_file():
    raise SystemExit('Run must be frozen before launch.')
with (run_dir / 'supervisor.log').open('ab') as log:
    command = [sys.executable, '-u', str(run_dir / 'source' / 'supervise.py'),
               '--config', str(run_dir / 'config.json')]
    if args.recover:
        command += ['--recover', args.recover]
    p = subprocess.Popen(command,
                          stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                          start_new_session=True, cwd=str(run_dir))
print(f'Detached supervisor launched with PID {p.pid}; verify state and real result count.')
