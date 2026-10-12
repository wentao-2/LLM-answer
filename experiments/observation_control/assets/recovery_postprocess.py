"""Recover an auxiliary null WORK reader without changing frozen science."""
from pathlib import Path
import argparse
import contextlib
import json
import os
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from common import (atomic_json, canonical, load, now, read_db, sha,
                    validate_seal, write_json)

DESIGN_SHA = '4701c2eb33b396b945bb5990e5814ba0fb8495fa025f5d3960fad6bae0eec157'


def check_design():
    assert sha(ROOT / 'DESIGN_SEAL.json') == DESIGN_SHA
    return validate_seal()


def database_hashes():
    paths = sorted((ROOT / 'queues').rglob('*.sqlite'))
    assert paths
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def recovered_analysis():
    check_design()
    import analyze_results as frozen
    original = frozen.work_row
    items, records, _ = frozen.load_observations()
    affected = []
    unchanged = 0

    def guarded(item, rec):
        metadata = json.loads(rec['metadata_json'])
        value = metadata.get('final_output', '')
        if value is None:
            assert rec['status'] != 'ok' and rec['probability'] is None
            clone = dict(rec)
            metadata['final_output'] = ''
            clone['metadata_json'] = canonical(metadata)
            result = original(item, clone)
            assert result['work_schema_valid'] is False and result['work'] is None
            return result
        assert isinstance(value, str)
        return original(item, rec)

    for item in items:
        if item.get('condition') != 'full_steps':
            continue
        rec = records[(item['model_scope'][0], item['id'])]
        value = json.loads(rec['metadata_json']).get('final_output', '')
        if value is None:
            guarded(item, rec)
            affected.append(dict(item_id=item['id'], model=rec['model'], status=rec['status'],
                                 probability=rec['probability'], output_tokens=rec['output_tokens']))
        else:
            assert guarded(item, rec) == original(item, rec)
            unchanged += 1
    write_json(ROOT / 'monitor/NULL_WORK_GUARD.json', dict(
        state='passed', checked_at=now(), affected=affected,
        unchanged_string_records=unchanged, design_seal_sha256=DESIGN_SHA,
        frozen_analysis_sha256=sha(ROOT / 'source/analyze_results.py'),
        recovery_source_sha256=sha(Path(__file__)),
        scope='temporary null-to-empty conversion only in auxiliary WORK_JSON reader',
        scientific_parser_changed=False, scientific_databases_changed=False))
    frozen.work_row = guarded
    frozen.analyze()


def run():
    check_design()
    inference = load(ROOT / 'INFERENCE_TERMINAL.json')
    assert inference['state'] == 'inference_completed' and not inference['failures']
    failure = load(ROOT / 'CONTROLLER_TERMINAL.json')
    assert failure['state'] == 'administrative_failure'
    assert failure['message'] == 'Final analysis/audit failed: analyze_results.py'
    for queue in (ROOT / 'queues').iterdir():
        if (queue / 'queue.json').exists():
            assert load(queue / 'scheduler.terminal.json')['returncode'] == 0
            for job in load(queue / 'queue.json')['jobs']:
                assert load(queue / job['run_dir'] / 'terminal.json')['returncode'] == 0
    assert not (ROOT / 'analysis/RESULTS.json').exists()
    assert not (ROOT / 'monitor/RECOVERY_TERMINAL.json').exists()
    before = database_hashes()
    historical = {p: sha(ROOT / p) for p in (
        'CONTROLLER_TERMINAL.json', 'INFERENCE_TERMINAL.json',
        'monitor/analyze_results.py.log', 'monitor/analyze_results.py.terminal.json')}
    write_json(ROOT / 'monitor/RECOVERY_START.json', dict(
        started_at=now(), design_seal_sha256=DESIGN_SHA,
        recovery_note_sha256=sha(ROOT / 'assets/RECOVERY_NOTE.md'),
        recovery_source_sha256=sha(Path(__file__)),
        scientific_database_hashes=before, historical_failure_hashes=historical))
    steps = (
        ('recovery-analysis', [sys.executable, '-B', str(Path(__file__)), 'analyze']),
        ('recovery-audit', [sys.executable, '-B', str(ROOT / 'source/audit_results.py')]),
    )
    try:
        for name, command in steps:
            with (ROOT / 'monitor' / (name + '.log')).open('x', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            write_json(ROOT / 'monitor' / (name + '.terminal.json'), dict(
                command=command, returncode=result.returncode, finished_at=now()))
            assert result.returncode == 0, name + ' failed; original evidence preserved'
        assert load(ROOT / 'audit/AUDIT.json')['state'] == 'passed'
        check_design()
        assert database_hashes() == before
        assert {p: sha(ROOT / p) for p in historical} == historical
        write_json(ROOT / 'FINAL_SEAL.json', dict(
            state='audited', completed_at=now(), design_seal_sha256=DESIGN_SHA,
            analysis_sha256=sha(ROOT / 'analysis/RESULTS.json'),
            audit_sha256=sha(ROOT / 'audit/AUDIT.json'),
            scientific_database_hashes=before, historical_failure_hashes=historical,
            null_work_guard_sha256=sha(ROOT / 'monitor/NULL_WORK_GUARD.json'),
            recovery_note_sha256=sha(ROOT / 'assets/RECOVERY_NOTE.md'),
            recovery_source_sha256=sha(Path(__file__)), inference_reexecuted=False))
        terminal = dict(state='audited', completed_at=now(),
                        final_seal_sha256=sha(ROOT / 'FINAL_SEAL.json'),
                        original_controller_failure_preserved=True, manual_source_coding='pending')
        write_json(ROOT / 'monitor/RECOVERY_TERMINAL.json', terminal)
        status = load(ROOT / 'STATUS.json')
        status.update(state='audited', completed_at=terminal['completed_at'],
                      recovery_terminal='monitor/RECOVERY_TERMINAL.json')
        atomic_json(ROOT / 'STATUS.json', status)
        print(canonical(terminal))
    except BaseException as exc:
        write_json(ROOT / 'monitor/RECOVERY_TERMINAL.json', dict(
            state='administrative_failure', finished_at=now(),
            type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise


def pack():
    check_design()
    assert load(ROOT / 'INFERENCE_TERMINAL.json')['state'] == 'inference_completed'
    assert load(ROOT / 'monitor/RECOVERY_TERMINAL.json')['state'] == 'audited'
    final = load(ROOT / 'FINAL_SEAL.json')
    assert final['state'] == 'audited' and database_hashes() == final['scientific_database_hashes']
    assert load(ROOT / 'audit/AUDIT.json')['state'] == 'passed'
    assert sha(ROOT / 'analysis/RESULTS.json') == final['analysis_sha256']
    assert sha(ROOT / 'audit/AUDIT.json') == final['audit_sha256']
    destination = ROOT / 'snapshot.zip'
    if destination.exists():
        seal = load(ROOT / 'SNAPSHOT_SEAL.json')
        assert destination.stat().st_size == seal['size'] and sha(destination) == seal['sha256']
        print(canonical(seal))
        return
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in path.parts for part in ('runtime-cache', '__pycache__')):
            continue
        if rel.endswith(('.lock', '.zip', '.pending')) or 'results.backup' in rel:
            continue
        if rel == 'SNAPSHOT_SEAL.json':
            continue
        files.append(path)
    index = {p.relative_to(ROOT).as_posix(): dict(size=p.stat().st_size, sha256=sha(p)) for p in files}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('SNAPSHOT_INDEX.json', json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    report = dict(state='passed', created_at=now(), file_count=len(files),
                  size=destination.stat().st_size, sha256=sha(destination),
                  design_seal_sha256=DESIGN_SHA, analysis_sha256=final['analysis_sha256'],
                  recovery_used=True, scientific_database_hashes=final['scientific_database_hashes'])
    write_json(ROOT / 'SNAPSHOT_SEAL.json', report)
    print(canonical(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'analyze', 'pack'))
    args = parser.parse_args()
    if args.action == 'analyze':
        recovered_analysis()
    elif args.action == 'run':
        import fcntl
        with (ROOT / 'monitor/recovery.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run()
    else:
        pack()
