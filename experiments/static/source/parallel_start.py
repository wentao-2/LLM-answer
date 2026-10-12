"""Launch a frozen parallel controller independently of the SSH connection."""
import argparse
import subprocess
import sys
from pathlib import Path
from common import validate_frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--recover', help='Documented diagnosis/fix allowing retry budgets to reset')
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    validate_frozen(run_dir)
    controller = run_dir / 'source' / 'parallel_supervise.py'
    if not controller.is_file():
        raise SystemExit('Parallel controller must be included in the frozen source.')
    command = [sys.executable, '-u', str(controller), '--config', str(run_dir / 'config.json')]
    if args.recover:
        command += ['--recover', args.recover]
    with (run_dir / 'supervisor.log').open('ab') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, cwd=str(run_dir))
    print(f'Detached parallel supervisor launched with PID {child.pid}; verify real shard progress.')


if __name__ == '__main__':
    main()
