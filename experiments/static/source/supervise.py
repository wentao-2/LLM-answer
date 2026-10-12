"""Single pipeline supervisor, detached from SSH. Worker failures have bounded retries."""
import argparse
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from common import atomic_json, digest, now, validate_frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--recover', help='Reason for resetting an exhausted retry budget after a diagnosed fix')
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    run_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_frozen(run_dir)
    lock = (run_dir / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another supervisor owns this run; no duplicate launched.', flush=True)
        return
    state_path = run_dir / 'state.json'
    old = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    if old.get('state') == 'complete':
        print('Run already complete.', flush=True)
        return
    if old.get('state') == 'needs_attention' and not args.recover:
        raise RuntimeError('Retry budget exhausted. Diagnose and fix before start.py --recover REASON.')
    if args.recover:
        old.setdefault('recovery_events', []).append({'time': now(), 'reason': args.recover,
                                                     'previous_failures': old.get('failures', {})})
        old['failures'] = {}
    state = dict(old, state='starting', supervisor_pid=os.getpid(), config_sha256=digest(config_path),
                 updated_at=now(), started_at=old.get('started_at', now()))
    state.setdefault('failures', {})
    atomic_json(state_path, state)
    worker_lock = (run_dir / 'worker.lock').open('a')
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        state.update(state='orphan_worker_active', updated_at=now())
        atomic_json(state_path, state)
        print('Existing worker is still active; deferred without consuming retry budget.', flush=True)
        return
    finally:
        worker_lock.close()
    # Respect other workloads; bounded monitoring will see a waiting state.
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(used.splitlines()[0]) > 2048:
        state.update(state='waiting_for_gpu', updated_at=now())
        atomic_json(state_path, state)
        print('GPU already occupied; deferred without interrupting other tasks.', flush=True)
        return
    for spec in config['models']:
        name = spec['name']
        while True:
            if state['failures'].get(name, 0) >= 3:
                state.update(state='needs_attention', model=name, updated_at=now())
                atomic_json(state_path, state)
                return
            state.update(state='running', model=name, updated_at=now())
            atomic_json(state_path, state)
            log = (run_dir / f'{name}.log').open('a', encoding='utf-8')
            print(f'{now()} Starting/resuming {name}', flush=True)
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('run.py')),
                                      '--config', str(config_path), '--model', name], stdout=log, stderr=log,
                                      start_new_session=True)
            state.update(worker_pid=child.pid, updated_at=now())
            atomic_json(state_path, state)
            last_activity = time.monotonic()
            last_progress_stamp = None
            while child.poll() is None:
                time.sleep(15)
                progress_path = run_dir / 'progress.json'
                stamp = progress_path.stat().st_mtime_ns if progress_path.exists() else None
                if stamp != last_progress_stamp:
                    last_activity = time.monotonic()
                    last_progress_stamp = stamp
                if time.monotonic() - last_activity > config.get('stall_timeout_sec', 5400):
                    # This Popen is an unreaped direct child in its own new session. Verify
                    # the command too; never signal a cached PID from a previous invocation.
                    command = Path(f'/proc/{child.pid}/cmdline').read_bytes().split(b'\0')
                    expected_script = str(Path(__file__).with_name('run.py')).encode()
                    if str(config_path).encode() not in command or expected_script not in command:
                        raise RuntimeError('Stalled child identity did not match; no process signalled.')
                    state.setdefault('watchdog_events', []).append({'time': now(), 'model': name,
                        'worker_pid': child.pid, 'no_progress_sec': time.monotonic() - last_activity})
                    atomic_json(state_path, state)
                    print(f'{now()} No progress within frozen timeout; stopping owned worker {child.pid}', flush=True)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                state['updated_at'] = now()
                atomic_json(state_path, state)
            log.close()
            if child.returncode == 0:
                break
            # A surviving orphan worker may still hold its own lock: never kill it or start a duplicate.
            state['failures'][name] = state['failures'].get(name, 0) + 1
            state.update(last_exit_code=child.returncode, updated_at=now())
            atomic_json(state_path, state)
            time.sleep(20)
    state.update(state='analyzing', updated_at=now())
    atomic_json(state_path, state)
    try:
        analysis_script = config.get('analysis_script', 'analyze.py')
        if analysis_script not in ('analyze.py', 'analyze_formal.py'):
            raise ValueError('Unrecognized analysis entry point')
        subprocess.run([sys.executable, str(Path(__file__).with_name(analysis_script)), '--db',
            str(run_dir / 'results.sqlite'), '--manifest', str(run_dir / config['manifest']),
            '--output-dir', str(run_dir / 'analysis')], check=True)
        db = sqlite3.connect(run_dir / 'results.sqlite')
        backup = sqlite3.connect(run_dir / 'results.backup.pending.sqlite')
        try:
            db.backup(backup)
        finally:
            backup.close()
            db.close()
        os.replace(run_dir / 'results.backup.pending.sqlite', run_dir / 'results.backup.sqlite')
    except Exception as error:
        state.update(state='needs_attention', failure_stage='analysis_or_backup',
                     last_error=repr(error), updated_at=now())
        atomic_json(state_path, state)
        raise
    state.update(state='complete', updated_at=now(), completed_at=now())
    atomic_json(state_path, state)
    print(f'{now()} All models completed and analyzed.', flush=True)


if __name__ == '__main__':
    main()
