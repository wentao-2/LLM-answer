"""Linux GPU queue for frozen static stages, resumable without retrying failures.

Run under nohup if desired. GPU and dispatch locks remain held by descendants if
the scheduler exits. Only a new nonterminal dispatch may resume committed rows.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid

import run_extension as runner
from prepare_runs import source_hashes


def validate_queue(directory):
    root = Path(directory).resolve()
    frozen = json.loads((root / 'queue.frozen.json').read_text(encoding='utf-8'))
    if runner.digest(root / 'queue.json') != frozen['queue_sha256']:
        raise RuntimeError('Frozen queue.json changed')
    if source_hashes(root / 'source') != frozen['source_hashes']:
        raise RuntimeError('Frozen queue sources changed')
    if runner.digest(__file__) != frozen['source_hashes'].get('schedule_runs.py'):
        raise RuntimeError('Execute the scheduler version frozen with this queue')
    for name, expected in frozen['input_hashes'].items():
        if runner.digest(runner.relative_artifact(root, name)) != expected:
            raise RuntimeError(f'Frozen queue input changed: {name}')
    queue = json.loads((root / 'queue.json').read_text(encoding='utf-8'))
    job_ids, record_ids = set(), set()
    for job in queue['jobs']:
        if job['id'] in job_ids:
            raise RuntimeError('Duplicate queue job id')
        job_ids.add(job['id'])
        run_dir = runner.relative_artifact(root, job['run_dir'])
        if runner.digest(run_dir / 'frozen.json') != job['frozen_sha256']:
            raise RuntimeError(f"Worker freeze changed: {job['id']}")
        config, _ = runner.validate_frozen(run_dir / 'config.json', run_dir / 'source/run_extension.py')
        if len(config['models']) != 1 or config['models'][0]['name'] != job['model']:
            raise RuntimeError('Each worker must contain exactly its designated model')
        expected = job['expected_item_ids']
        if len(set(expected)) != len(expected) or job['count'] != len(expected):
            raise RuntimeError('Invalid job id inventory')
        rows = [json.loads(line) for line in (run_dir / config['manifest']).read_text(encoding='utf-8').splitlines() if line.strip()]
        if sorted(item['id'] for item in runner.select_rows(rows, job['model'])) != expected:
            raise RuntimeError('Job manifest does not match its expected ids')
        ids = {job['model'] + ':' + item_id for item_id in expected}
        if record_ids & ids:
            raise RuntimeError('Overlapping worker result ids')
        record_ids.update(ids)
    if len(record_ids) != queue['expected_records']:
        raise RuntimeError('Queue record count mismatch')
    return root, queue, frozen


def try_lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def lock_busy(path):
    stream = try_lock(path)
    if stream is None:
        return True
    stream.close()
    return False


def read_terminal(run_dir):
    path = run_dir / 'terminal.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def event(root, name, **data):
    value = dict(time=runner.now(), event=name, **data)
    with (root / 'scheduler.log').open('a', encoding='utf-8') as stream:
        stream.write(runner.canonical_json(value) + '\n')
        stream.flush()
    print(runner.canonical_json(value), flush=True)


def read_database_snapshot(path):
    """SQLite backup reads committed rows safely even with rollback journaling."""
    if not Path(path).is_file():
        return None
    source = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
    snapshot = sqlite3.connect(':memory:')
    try:
        source.backup(snapshot)
    except BaseException:
        snapshot.close()
        raise
    finally:
        source.close()
    return snapshot


def inspect_records(run_dir, job, require_complete=False):
    snapshot = read_database_snapshot(run_dir / 'results.sqlite')
    if snapshot is None:
        if require_complete:
            raise RuntimeError(f"Missing worker database: {job['id']}")
        return [], False
    try:
        columns = tuple(row[1] for row in snapshot.execute('PRAGMA table_info(results)'))
        if not columns and not snapshot.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if require_complete:
                raise RuntimeError(f"Empty worker database: {job['id']}")
            return [], False
        if columns != runner.RESULT_COLUMNS:
            raise RuntimeError(f"Invalid worker schema: {job['id']}")
        frozen = json.loads((run_dir / 'frozen.json').read_text(encoding='utf-8'))
        metadata = {key: json.loads(value) for key, value in snapshot.execute('SELECT key,value FROM metadata')}
        rows = snapshot.execute('SELECT * FROM results ORDER BY id').fetchall()
        for key in ('config_sha256', 'manifest_sha256', 'source_hashes', 'protocol_hashes'):
            if (key in metadata and metadata[key] != frozen[key]) or (rows and key not in metadata):
                raise RuntimeError(f"Worker database identity mismatch: {job['id']} / {key}")
        expected = {job['model'] + ':' + item_id for item_id in job['expected_item_ids']}
        actual = set()
        for row in rows:
            record_id, model, item_id, status = row[:4]
            if model != job['model'] or record_id != model + ':' + item_id or status not in ('ok', 'invalid', 'truncated', 'error'):
                raise RuntimeError(f"Malformed terminal result in {job['id']}")
            actual.add(record_id)
        if not actual.issubset(expected) or len(actual) != len(rows):
            raise RuntimeError(f"Unexpected or duplicate worker result ids: {job['id']}")
        complete = actual == expected
        if require_complete and not complete:
            raise RuntimeError(f"Incomplete worker {job['id']}: {len(actual)}/{len(expected)}")
        return rows, complete
    finally:
        snapshot.close()


def merge_queue(root, queue, frozen):
    """Merge complete immutable terminal records using INSERT, never replace."""
    target = root / 'merged.sqlite'
    pending = root / f'merged.pending-{uuid.uuid4().hex}.sqlite'
    merged = runner.connect_db(pending)
    try:
        runner.require_metadata(merged, 'queue_sha256', frozen['queue_sha256'])
        runner.require_metadata(merged, 'worker_freezes', {job['id']: job['frozen_sha256'] for job in queue['jobs']})
        expected = set()
        for job in queue['jobs']:
            run_dir = root / job['run_dir']
            terminal = read_terminal(run_dir)
            if terminal and terminal['returncode'] != 0:
                raise RuntimeError(f"Failed worker cannot be merged as complete: {job['id']}")
            rows, _ = inspect_records(run_dir, job, require_complete=True)
            expected.update(job['model'] + ':' + item_id for item_id in job['expected_item_ids'])
            with merged:
                merged.executemany('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)
        actual = {row[0] for row in merged.execute('SELECT id FROM results')}
        if actual != expected or len(actual) != queue['expected_records']:
            raise RuntimeError('Merged result ids do not exactly equal the planned ids')
        merged.commit()
        if target.exists():
            old = read_database_snapshot(target)
            try:
                if old.execute('SELECT * FROM results ORDER BY id').fetchall() != merged.execute('SELECT * FROM results ORDER BY id').fetchall():
                    raise RuntimeError('Existing merged.sqlite differs. It was not overwritten.')
                old_identity = old.execute("SELECT value FROM metadata WHERE key='queue_sha256'").fetchone()
                if not old_identity or json.loads(old_identity[0]) != frozen['queue_sha256']:
                    raise RuntimeError('Existing merged.sqlite belongs to a different queue')
            finally:
                old.close()
            merged.close()
            pending.unlink()
            return target
        merged.close()
        # Exclusive link publication cannot silently replace a concurrent merge.
        os.link(pending, target)
        pending.unlink()
        return target
    finally:
        merged.close()


def execute_job(root, queue, job_id, lock_fds):
    """Small durable wrapper: persist the worker's exit code even after parent exit."""
    job = next(job for job in queue['jobs'] if job['id'] == job_id)
    run_dir = root / job['run_dir']
    terminal = read_terminal(run_dir)
    if terminal is not None:
        return terminal['returncode']
    started = runner.now()
    runner.atomic_json(run_dir / 'execution.json', {'wrapper_pid': os.getpid(), 'started_at': started,
                                                  'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'job': job_id})
    code = 1
    error = None
    try:
        command = [sys.executable, str(run_dir / 'source/run_extension.py'), '--config', str(run_dir / 'config.json'),
                   '--model', job['model']]
        process = subprocess.Popen(command, cwd=run_dir, pass_fds=tuple(lock_fds))
        runner.atomic_json(run_dir / 'execution.json', {'wrapper_pid': os.getpid(), 'worker_pid': process.pid,
                                                      'started_at': started, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                                      'job': job_id})
        code = process.wait()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    runner.atomic_json(run_dir / 'terminal.json', {'job': job_id, 'returncode': code, 'started_at': started,
                                                 'finished_at': runner.now(), 'error': error})
    return code if code >= 0 else 128 - code


def schedule(root, queue, frozen, gpus, gpu_lock_dir, poll_seconds=5):
    if os.name != 'posix':
        raise RuntimeError('GPU scheduling is supported on Linux only')
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError('--gpus must contain unique nonnegative GPU indices')
    active = {}
    state_path = root / 'queue_state.json'
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding='utf-8'))
        if old['queue_sha256'] != frozen['queue_sha256']:
            raise RuntimeError('Queue state belongs to a different frozen queue')
    with runner.worker_lock(root / 'scheduler.lock'):
        event(root, 'scheduler_started', pid=os.getpid(), gpus=gpus)
        try:
            while True:
                # Reap wrappers owned by this scheduler. Their terminal files also
                # support a new scheduler after an interruption.
                for job_id, (process, gpu_lock, dispatch_lock, log) in list(active.items()):
                    if process.poll() is not None:
                        job = next(job for job in queue['jobs'] if job['id'] == job_id)
                        run_dir = root / job['run_dir']
                        if read_terminal(run_dir) is None:
                            runner.atomic_json(run_dir / 'terminal.json', {
                                'job': job_id, 'returncode': process.returncode or 1,
                                'finished_at': runner.now(), 'error': 'Wrapper exited without a terminal record',
                            })
                        log.close()
                        gpu_lock.close()
                        dispatch_lock.close()
                        del active[job_id]
                states, pending = {}, []
                for job in queue['jobs']:
                    run_dir = root / job['run_dir']
                    terminal = read_terminal(run_dir)
                    if terminal:
                        if terminal['returncode'] == 0:
                            try:
                                _, complete = inspect_records(run_dir, job, require_complete=True)
                                states[job['id']] = {'state': 'completed', 'count': job['count']}
                            except Exception as exc:
                                states[job['id']] = {'state': 'failed', 'error': str(exc)}
                        else:
                            states[job['id']] = {'state': 'failed', 'returncode': terminal['returncode'], 'error': terminal.get('error')}
                    elif lock_busy(run_dir / 'dispatch.lock'):
                        states[job['id']] = {'state': 'running'}
                    else:
                        try:
                            rows, complete = inspect_records(run_dir, job)
                        except Exception as exc:
                            states[job['id']] = {'state': 'failed', 'error': str(exc)}
                            continue
                        if complete:
                            states[job['id']] = {'state': 'completed', 'count': len(rows), 'recovered': True}
                        else:
                            states[job['id']] = {'state': 'pending', 'saved': len(rows)}
                            pending.append(job)
                for job in pending:
                    run_dir = root / job['run_dir']
                    dispatch = try_lock(run_dir / 'dispatch.lock')
                    if dispatch is None:
                        continue
                    gpu = gpu_lock = None
                    for candidate in gpus:
                        candidate_lock = try_lock(Path(gpu_lock_dir) / f'gpu-{candidate}.lock')
                        if candidate_lock is not None:
                            gpu, gpu_lock = candidate, candidate_lock
                            break
                    if gpu_lock is None:
                        dispatch.close()
                        break
                    log = (run_dir / 'worker.log').open('a', encoding='utf-8')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                    command = [sys.executable, str(root / 'source/schedule_runs.py'), '--queue', str(root),
                               '--execute-job', job['id'], '--lock-fds', f'{gpu_lock.fileno()},{dispatch.fileno()}']
                    try:
                        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   pass_fds=(gpu_lock.fileno(), dispatch.fileno()), start_new_session=True)
                    except Exception as exc:
                        log.close()
                        gpu_lock.close()
                        dispatch.close()
                        runner.atomic_json(run_dir / 'terminal.json', {'job': job['id'], 'returncode': 1,
                                                                     'finished_at': runner.now(), 'error': str(exc)})
                        states[job['id']] = {'state': 'failed', 'error': str(exc)}
                        event(root, 'launch_failed', job=job['id'], error=str(exc))
                        continue
                    active[job['id']] = (process, gpu_lock, dispatch, log)
                    states[job['id']] = {'state': 'running', 'gpu': gpu, 'wrapper_pid': process.pid}
                    event(root, 'worker_started', job=job['id'], gpu=gpu, wrapper_pid=process.pid)
                state = {'queue_sha256': frozen['queue_sha256'], 'updated_at': runner.now(), 'scheduler_pid': os.getpid(),
                         'jobs': states}
                runner.atomic_json(state_path, state)
                if all(value['state'] in ('completed', 'failed') for value in states.values()):
                    failures = [name for name, value in states.items() if value['state'] == 'failed']
                    if failures:
                        event(root, 'queue_failed', jobs=failures)
                        return 1
                    merged = merge_queue(root, queue, frozen)
                    event(root, 'queue_complete', merged=str(merged), count=queue['expected_records'])
                    return 0
                time.sleep(poll_seconds)
        finally:
            # Closing the parent's handles does not release the inherited locks
            # still held by a live wrapper or model worker. No child is killed.
            for _, gpu_lock, dispatch_lock, log in active.values():
                gpu_lock.close()
                dispatch_lock.close()
                log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--gpu-lock-dir', default='/tmp/i16-extension-gpu-locks')
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--merge-only', action='store_true')
    parser.add_argument('--execute-job', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fds', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, queue, frozen = validate_queue(args.queue)
    if args.execute_job:
        if not args.lock_fds:
            parser.error('Internal worker wrapper requires inherited lock file descriptors')
        return execute_job(root, queue, args.execute_job, [int(value) for value in args.lock_fds.split(',')])
    if args.merge_only:
        with runner.worker_lock(root / 'scheduler.lock'):
            print(merge_queue(root, queue, frozen))
        return 0
    if args.poll_seconds < 0.1 or args.poll_seconds > 60:
        parser.error('--poll-seconds must be between 0.1 and 60')
    return schedule(root, queue, frozen, args.gpus.split(','), args.gpu_lock_dir, args.poll_seconds)


if __name__ == '__main__':
    sys.exit(main())
