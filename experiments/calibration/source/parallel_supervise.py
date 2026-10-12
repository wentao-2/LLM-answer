"""Detached, bounded four-GPU execution of frozen independent result shards.

Each worker writes only its shard database. Aggregation copies complete rows and
rejects any overlap/conflict; it never overwrites a terminal inference result.
"""
import argparse
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import closing
from pathlib import Path

from common import atomic_json, connect_db, digest, now, require_metadata, validate_frozen


def analysis_script_path(run_dir, config):
    """Only execute an explicitly supported analyzer inside the frozen source."""
    analysis = config.get('analysis_script', 'analyze_formal.py')
    if analysis not in ('analyze.py', 'analyze_formal.py', 'analyze_mechanism.py'):
        raise RuntimeError('Unrecognized analysis script')
    script = Path(run_dir) / 'source' / analysis
    if not script.is_file():
        raise RuntimeError('Analysis script is missing from the frozen source')
    return script


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def host_identity():
    result = {'hostname': socket.gethostname()}
    for key, filename in (('boot_id', '/proc/sys/kernel/random/boot_id'),
                          ('machine_id', '/etc/machine-id')):
        path = Path(filename)
        result[key] = path.read_text().strip() if path.is_file() else None
    return result


def process_start_time(pid, proc_root='/proc'):
    """Linux start ticks distinguish a reused PID from the original process."""
    text = (Path(proc_root) / str(pid) / 'stat').read_text()
    return text.rsplit(')', 1)[1].split()[19]


def acquire_directory_claim(run_dir, local_host, *, pid=None, proc_root='/proc'):
    """Atomic mkdir protects shared NFS mounts whose flock is host-local.

    A claim on another host cannot be declared stale by inspecting local /proc.
    Such claims are preserved and require an explicitly documented migration.
    """
    pid = os.getpid() if pid is None else pid
    run_dir = Path(run_dir)
    claim = run_dir / 'controller.claim'
    owner = {'host': local_host, 'pid': pid, 'start_ticks': process_start_time(pid, proc_root),
             'claimed_at': now()}
    try:
        claim.mkdir()
    except FileExistsError:
        previous = read_json(claim / 'owner.json')
        if not previous or previous.get('host') != local_host:
            raise RuntimeError('Shared controller claim belongs to another/unidentified host; no takeover')
        try:
            current_start = process_start_time(previous['pid'], proc_root)
        except FileNotFoundError:
            current_start = None
        if current_start == previous.get('start_ticks') or previous.get('start_ticks') is None:
            raise RuntimeError('Shared controller claim owner is still active/unverifiable; no duplicate')
        # Preserve the stale claim as provenance. The local supervisor flock held
        # by the caller prevents two same-host contenders from racing takeover.
        archive = run_dir / ('controller.claim.stale.' + uuid.uuid4().hex)
        os.rename(claim, archive)
        claim.mkdir()
    atomic_json(claim / 'owner.json', owner)
    return owner


def validate_migration(run_dir, config):
    migration = config.get('migration')
    if migration:
        snapshot = (run_dir / migration['snapshot']).resolve()
        if run_dir.resolve() not in snapshot.parents:
            raise RuntimeError('Migration snapshot must remain inside this run')
        if digest(snapshot) != migration['snapshot_sha256']:
            raise RuntimeError('Migration source snapshot hash changed')
        if digest(run_dir / 'migration.json') != migration['record_sha256']:
            raise RuntimeError('Migration provenance record hash changed')


def process_identity(pid, script, config, recorded_host, *, proc_root='/proc', local_host=None):
    """A shared filesystem's cached PID is never evidence about this host."""
    local_host = host_identity() if local_host is None else local_host
    result = {'pid': pid, 'alive': False, 'matches_this_run': False,
              'same_host': recorded_host == local_host}
    if not result['same_host'] or not isinstance(pid, int) or pid <= 1:
        return result
    try:
        command = (Path(proc_root) / str(pid) / 'cmdline').read_bytes().decode('utf-8').split('\0')
        result.update(alive=bool(command[0]), command=command,
                      matches_this_run=str(script) in command and str(config) in command)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def matching_workers(shard, proc_root='/proc'):
    """Find actual local commands, not arbitrary processes with reused PIDs."""
    script, config = str(shard / 'source' / 'run.py'), str(shard / 'config.json')
    found = []
    for path in Path(proc_root).glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().decode('utf-8').split('\0')
            if script in command and config in command:
                found.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
    return found


def lock_available(path):
    import fcntl
    with Path(path).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def connect_readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)


def manifest_ids(run_dir, config):
    rows = [json.loads(line) for line in (run_dir / config['manifest']).read_text(encoding='utf-8').splitlines() if line]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate manifest item IDs')
    return ids


def validate_layout(run_dir, config):
    """Validate a disjoint, exhaustive assignment without changing science."""
    shards = config['parallel']['shards']
    if not shards or len({s['name'] for s in shards}) != len(shards):
        raise RuntimeError('Missing or duplicate shard names')
    if len({s['gpu'] for s in shards}) != len(shards):
        raise RuntimeError('GPU assignments must be distinct')
    if any(not isinstance(s['gpu'], int) or s['gpu'] < 0 for s in shards):
        raise RuntimeError('GPU assignments must be nonnegative physical indices')
    expected = set(manifest_ids(run_dir, config))
    parent_items = {r['id']: r for r in (json.loads(line) for line in
        (run_dir / config['manifest']).read_text(encoding='utf-8').splitlines() if line)}
    assigned = set()
    paths = set()
    for spec in shards:
        shard = (run_dir / spec['run_dir']).resolve()
        if run_dir.resolve() not in shard.parents or shard in paths:
            raise RuntimeError('Shard must be a unique directory within this parent run')
        paths.add(shard)
        child_config = validate_frozen(shard)
        if child_config['models'] != config['models']:
            raise RuntimeError('Shard model definitions differ from parent')
        # These fields affect scientific inference, rather than distribution.
        for key in ('seed', 'batch_size', 'cpu_threads', 'dtype', 'do_sample', 'num_beams',
                    'max_input_tokens', 'max_new_tokens', 'attention'):
            if child_config.get(key) != config.get(key):
                raise RuntimeError(f'Shard scientific configuration differs: {key}')
        ids = set(manifest_ids(shard, child_config))
        if assigned.intersection(ids) or not ids.issubset(expected):
            raise RuntimeError('Shard manifests overlap or contain foreign items')
        child_items = [json.loads(line) for line in (shard / child_config['manifest']).read_text(
            encoding='utf-8').splitlines() if line]
        if any(item != parent_items[item['id']] for item in child_items):
            raise RuntimeError('Shard item content differs from frozen parent manifest')
        assigned.update(ids)
    if assigned != expected:
        raise RuntimeError('Shard manifests do not cover parent manifest')
    return shards


def db_counts(path):
    if not Path(path).exists():
        return {}
    with closing(connect_readonly(path)) as db:
        return dict(db.execute('SELECT model,COUNT(*) FROM results GROUP BY model'))


def model_complete(shard, config, model):
    expected = set(manifest_ids(shard, config))
    path = shard / 'results.sqlite'
    if not path.exists():
        return False
    with closing(connect_readonly(path)) as db:
        rows = db.execute('SELECT item_id FROM results WHERE model=?', (model,)).fetchall()
    saved = {r[0] for r in rows}
    if not saved.issubset(expected) or len(saved) != len(rows):
        raise RuntimeError('Unexpected or duplicate shard result IDs')
    return saved == expected


def atomic_backup(db, target):
    target = Path(target)
    temporary = target.with_name(target.name + f'.{os.getpid()}.pending')
    backup = sqlite3.connect(temporary)
    try:
        db.backup(backup)
        if backup.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError('Backup integrity check failed')
    finally:
        backup.close()
    os.replace(temporary, target)


def merge_results(run_dir, config, *, final=False):
    """Read atomic shard backups while active; only read live DBs after exit.

    All imported rows are compared byte-for-byte as SQLite values on restart.
    An interrupted transaction rolls back; a completed identical import is a no-op.
    """
    run_dir = Path(run_dir).resolve()
    model_names = {m['name'] for m in config['models']}
    expected = {(model, item) for model in model_names for item in manifest_ids(run_dir, config)}
    db = connect_db(run_dir / 'results.sqlite')
    try:
        require_metadata(db, 'config_sha256', digest(run_dir / 'config.json'))
        require_metadata(db, 'manifest_sha256', digest(run_dir / config['manifest']))
        require_metadata(db, 'source_hashes', {p.name: digest(p) for p in sorted((run_dir / 'source').glob('*.py'))})
        require_metadata(db, 'parallel_shards', [dict(s, frozen_sha256=digest(run_dir / s['run_dir'] / 'frozen.json'))
                                               for s in config['parallel']['shards']])
        seen = set()
        imported = 0
        with db:
            for spec in config['parallel']['shards']:
                shard = run_dir / spec['run_dir']
                child_config = read_json(shard / 'config.json')
                owned = set(manifest_ids(shard, child_config))
                source = shard / ('results.sqlite' if final else 'results.backup.sqlite')
                if not source.exists():
                    if final:
                        raise RuntimeError(f'Missing completed database: {source}')
                    continue
                with closing(connect_readonly(source)) as source_db:
                    if source_db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                        raise RuntimeError(f'Shard database integrity failed: {source}')
                    columns = [r[1] for r in source_db.execute('PRAGMA table_info(results)')]
                    target_columns = [r[1] for r in db.execute('PRAGMA table_info(results)')]
                    if columns != target_columns:
                        raise RuntimeError('Shard result schema differs from parent')
                    for row in source_db.execute('SELECT * FROM results'):
                        key = (row[1], row[2])
                        if key not in expected or row[2] not in owned or row[0] != row[1] + ':' + row[2]:
                            raise RuntimeError(f'Foreign or malformed result in {spec["name"]}: {key}')
                        if key in seen:
                            raise RuntimeError(f'Duplicate result across shards: {key}')
                        seen.add(key)
                        previous = db.execute('SELECT * FROM results WHERE model=? AND item_id=?', key).fetchone()
                        if previous is not None:
                            if previous != row:
                                raise RuntimeError(f'Immutable result conflict: {key}')
                        else:
                            db.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)', row)
                            imported += 1
            saved_rows = db.execute('SELECT model,item_id FROM results').fetchall()
            saved = set(saved_rows)
            if len(saved) != len(saved_rows) or not saved.issubset(expected):
                raise RuntimeError('Aggregated database contains unexpected result keys')
            if final and (seen != expected or saved != expected):
                raise RuntimeError(f'Final matrix incomplete: {len(saved)}/{len(expected)}')
        atomic_backup(db, run_dir / 'results.backup.sqlite')
        report = {'updated_at': now(), 'saved': len(saved), 'expected': len(expected),
                  'imported': imported, 'complete': final, 'backup_sha256': digest(run_dir / 'results.backup.sqlite')}
        atomic_json(run_dir / 'aggregation.json', report)
        return report
    finally:
        db.close()


def gpu_snapshot(index):
    text = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=memory.used,uuid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    used, identifier = [value.strip() for value in text.strip().split(',')]
    if not identifier.startswith('GPU-'):
        raise RuntimeError('nvidia-smi did not return a GPU UUID')
    return int(used), identifier


def stop_owned(child, shard, local_host):
    identity = process_identity(child.pid, shard / 'source' / 'run.py', shard / 'config.json', local_host)
    if not identity['matches_this_run'] or child.poll() is not None:
        raise RuntimeError('Owned child identity mismatch; no process signalled')
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        identity = process_identity(child.pid, shard / 'source' / 'run.py', shard / 'config.json', local_host)
        if not identity['matches_this_run']:
            raise RuntimeError('Owned child identity changed; no SIGKILL sent')
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)


def main():
    import fcntl
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--recover', help='Concrete diagnosed fix permitting exhausted budgets to reset')
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    run_dir = config_path.parent
    config = validate_frozen(run_dir)
    validate_migration(run_dir, config)
    shards = validate_layout(run_dir, config)
    analysis = analysis_script_path(run_dir, config)
    lock = (run_dir / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another parallel supervisor holds the parent lock; no duplicate.', flush=True)
        return
    local_host = host_identity()
    allowed = config['parallel'].get('allowed_host', {})
    if any(local_host.get(key) != value for key, value in allowed.items()):
        raise RuntimeError('This frozen recovery run is bound to a different execution host')
    acquire_directory_claim(run_dir, local_host)
    state = read_json(run_dir / 'state.json', {})
    state.update(supervisor_pid=os.getpid(), host=local_host, config_sha256=digest(config_path),
                 started_at=state.get('started_at', now()), updated_at=now())
    state.setdefault('shards', {})
    if args.recover:
        state.setdefault('recovery_events', []).append({'time': now(), 'reason': args.recover,
            'previous_failures': {k: v.get('failures', {}) for k, v in state['shards'].items()}})
        for entry in state['shards'].values():
            entry.update(failure_count=0, failures={}, state='pending')
        state.pop('failure_stage', None)
        state.pop('last_error', None)
    elif state.get('state') == 'needs_attention':
        raise RuntimeError('Diagnose the stored failure before --recover REASON; budget not reset')
    state.update(state='starting', updated_at=now())
    atomic_json(run_dir / 'state.json', state)
    live = {}
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_requested.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: stop_requested.append(signum))
    settings = config['parallel']
    max_failures = min(3, settings.get('max_failures', 3))
    if max_failures < 1:
        raise ValueError('max_failures must be between 1 and 3')
    last_backup = float('-inf')
    try:
        while True:
            if stop_requested:
                for name, activity in live.items():
                    if activity['child'].poll() is None:
                        stop_owned(activity['child'], activity['shard'], local_host)
                    activity['log'].close()
                    state['shards'][name].update(state='stopped_by_request', worker_pid=None)
                live.clear()
                state.update(state='stopped_by_request', updated_at=now())
                atomic_json(run_dir / 'state.json', state)
                return
            for spec in shards:
                name = spec['name']
                shard = (run_dir / spec['run_dir']).resolve()
                shard_config = read_json(shard / 'config.json')
                entry = state['shards'].setdefault(name, {'failures': {}, 'failure_count': 0})
                entry.update(host=local_host, supervisor_pid=os.getpid(), gpu=spec['gpu'], updated_at=now())
                activity = live.get(name)
                if activity:
                    child = activity['child']
                    if child.poll() is None:
                        path = shard / 'progress.json'
                        stamp = path.stat().st_mtime_ns if path.exists() else None
                        if stamp != activity['progress_stamp']:
                            activity.update(progress_stamp=stamp, last_activity=time.monotonic())
                        elapsed = time.monotonic() - activity['last_activity']
                        if elapsed > settings.get('stall_timeout_sec', 5400):
                            entry.setdefault('watchdog_events', []).append({'time': now(),
                                'pid': child.pid, 'model': entry['model'], 'no_progress_sec': elapsed})
                            stop_owned(child, shard, local_host)
                    if child.poll() is None:
                        continue
                    activity['log'].close()
                    del live[name]
                    success = child.returncode == 0 and model_complete(shard, shard_config, entry['model'])
                    entry.update(worker_pid=None, last_exit_code=child.returncode)
                    if not success:
                        entry['failure_count'] = entry.get('failure_count', 0) + 1
                        entry['failures'][entry['model']] = entry['failures'].get(entry['model'], 0) + 1
                        entry['retry_after_epoch'] = time.time() + 20
                        entry['state'] = 'retry_pending'
                    else:
                        entry['state'] = 'model_complete'
                    # Preserve the exit/retry budget before any next launch.
                    atomic_json(run_dir / 'state.json', state)
                if entry.get('failure_count', 0) >= max_failures:
                    entry['state'] = 'needs_attention'
                    continue
                active = matching_workers(shard)
                if active or not lock_available(shard / 'worker.lock') or not lock_available(shard / 'supervisor.lock'):
                    entry.update(state='orphan_worker_active', observed_worker_pids=active, worker_pid=None)
                    continue
                pending = [m['name'] for m in config['models'] if not model_complete(shard, shard_config, m['name'])]
                if not pending:
                    entry.update(state='complete', worker_pid=None, completed_at=entry.get('completed_at', now()))
                    continue
                if time.time() < entry.get('retry_after_epoch', 0):
                    continue
                used, gpu_uuid = gpu_snapshot(spec['gpu'])
                if used > 2048:
                    entry.update(state='waiting_for_gpu', gpu_memory_used_mib=used, worker_pid=None)
                    continue
                validate_frozen(shard)
                model = pending[0]
                log = (shard / f'{model}.log').open('a', encoding='utf-8')
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
                child = subprocess.Popen([sys.executable, '-u', str(shard / 'source' / 'run.py'),
                    '--config', str(shard / 'config.json'), '--model', model], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, start_new_session=True, cwd=str(shard), env=environment)
                entry.update(state='running', model=model, worker_pid=child.pid, worker_started_at=now(), gpu_uuid=gpu_uuid)
                path = shard / 'progress.json'
                live[name] = {'child': child, 'log': log, 'shard': shard,
                    'last_activity': time.monotonic(), 'progress_stamp': path.stat().st_mtime_ns if path.exists() else None}
                atomic_json(run_dir / 'state.json', state)
                print(f'{now()} {name} GPU {spec["gpu"]} started {model} PID {child.pid}', flush=True)
            states = [state['shards'][s['name']]['state'] for s in shards]
            if all(value == 'complete' for value in states):
                state.update(state='analyzing', updated_at=now())
                atomic_json(run_dir / 'state.json', state)
                state['aggregation'] = merge_results(run_dir, config, final=True)
                subprocess.run([sys.executable, str(analysis), '--db',
                    str(run_dir / 'results.sqlite'), '--manifest', str(run_dir / config['manifest']),
                    '--output-dir', str(run_dir / 'analysis')], check=True)
                state.update(state='complete', completed_at=now(), updated_at=now())
                atomic_json(run_dir / 'state.json', state)
                print(f'{now()} All shards merged, validated, backed up and analyzed.', flush=True)
                return
            if time.monotonic() - last_backup >= settings.get('backup_interval_sec', 300):
                state['aggregation'] = merge_results(run_dir, config)
                last_backup = time.monotonic()
            state.update(state='running_with_attention' if 'needs_attention' in states else 'running', updated_at=now())
            for spec in shards:
                atomic_json(run_dir / spec['run_dir'] / 'state.json', state['shards'][spec['name']])
            if not live and all(value in ('complete', 'needs_attention') for value in states):
                state['state'] = 'needs_attention'
                atomic_json(run_dir / 'state.json', state)
                return
            atomic_json(run_dir / 'state.json', state)
            time.sleep(15)
    except Exception as error:
        # Detachment means children can survive a controller error. Preserve their
        # identities for status; a future launch waits on their locks, never kills them.
        state.update(state='needs_attention', failure_stage='controller_or_aggregation',
                     last_error=repr(error), updated_at=now())
        atomic_json(run_dir / 'state.json', state)
        raise


if __name__ == '__main__':
    main()
