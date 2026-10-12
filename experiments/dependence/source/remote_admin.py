"""Server preflight, durable launch, process-identity monitoring and final export."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from common import (ROOT, PHI, QWEN, atomic_json, canonical, load, now, read_jsonl,
                    sha, validate_seal, write_json)

PYTHONS = {
    PHI: '/workspace/codex-auto-research/env/bin/python',
    QWEN: '/workspace/i16-review-supplement-20261011/fast_runtime/vllm085/venv/bin/python',
}
GPU_LOCK_DIR = '/tmp/i16-extension-gpu-locks'


def identity(pid):
    path = Path('/proc')/str(pid)
    try:
        stat = (path/'stat').read_text()
        tail = stat[stat.rfind(')')+2:].split()
        cmd = (path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
        return dict(pid=int(pid), state=tail[0], start_ticks=int(tail[19]), command=cmd,
                    exe=os.readlink(path/'exe'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def render(model):
    from transformers import AutoTokenizer
    configs = load(ROOT/'inputs/runtime_configs.json')
    expected_environment = load(ROOT/'inputs/runtime_environments.json')[model]
    config = configs[model]
    spec = config['models'][0]
    packages = {name:importlib.metadata.version(name) for name in
                (('torch','transformers','vllm','tokenizers','huggingface-hub') if model==QWEN else ('torch','transformers','accelerate'))}
    expected = expected_environment.get('packages', expected_environment)
    for name, version in packages.items():
        if name in expected:
            assert version == expected[name], 'Runtime package changed: '+name
    if model == QWEN:
        assert packages['vllm']=='0.8.5' and packages['transformers']=='4.51.3'
    tokenizer = AutoTokenizer.from_pretrained(spec['path'], local_files_only=True, trust_remote_code=False)
    assert tokenizer.chat_template
    actual_config = load(Path(spec['path'])/'config.json')
    context_limit = actual_config.get('max_position_embeddings', actual_config.get('text_config',{}).get('max_position_embeddings'))
    rows = [x for x in read_jsonl(ROOT/'manifests/all.jsonl') if x['model_scope']==[model]]
    ledger = []
    from common import text_hash
    for item in rows:
        messages = item.get('messages', [dict(role='user',content=item.get('prompt'))])
        kwargs = item.get('chat_template_kwargs', {})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(ids) <= item['max_input_tokens']
        assert len(ids)+item['max_new_tokens'] <= context_limit
        if model == QWEN:
            assert len(ids)+item['max_new_tokens'] <= config['max_model_len']
        ledger.append(dict(item_id=item['id'], input_tokens=len(ids), rendered_prompt_sha256=text_hash(prompt),
                           rendered_token_ids_sha256=text_hash(canonical(ids))))
    write_json(ROOT/'preflight'/f'{model}-render.json', dict(model=model, packages=packages, context_limit=context_limit,
        n=len(ledger), min_input_tokens=min(x['input_tokens'] for x in ledger),
        max_input_tokens=max(x['input_tokens'] for x in ledger), ledger=ledger, checked_at=now()))


def verify_model(model):
    config = load(ROOT/'inputs/runtime_configs.json')[model]
    spec = config['models'][0]
    inventory = load(ROOT/'inputs/model_inventories.json')[model]
    expected_revision = inventory.get('revision', inventory.get('spec',{}).get('revision'))
    assert expected_revision == spec['revision']
    path = Path(spec['path'])
    checked = {}
    for name, expected in inventory['files'].items():
        target = path/name
        assert target.is_file() and target.stat().st_size == expected['size'], name
        digest = sha(target)
        assert digest == expected['sha256'], 'Complete checkpoint SHA mismatch: '+name
        checked[name] = dict(size=target.stat().st_size,sha256=digest)
    index = load(path/'model.safetensors.index.json')
    assert set(index['weight_map'].values()) == {name for name in checked if name.endswith('.safetensors')}
    process = subprocess.run([PYTHONS[model],'-B',str(Path(__file__)),'render','--model',model],
        capture_output=True, text=True, env=dict(os.environ, HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'))
    if process.returncode:
        raise RuntimeError('Native rendering failed: '+process.stderr[-6000:])
    return dict(model=model,revision=expected_revision,files=checked,
                render_sha256=sha(ROOT/'preflight'/f'{model}-render.json'))


def preflight():
    validate_seal()
    assert not (ROOT/'preflight/RESULT.json').exists(), 'Preflight already exists'
    (ROOT/'preflight').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(verify_model, (PHI,QWEN)))
    result = dict(state='passed', checked_at=now(), models=reports,
                  design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'), scientific_inference_started=False)
    write_json(ROOT/'preflight/RESULT.json', result)
    print(canonical(dict(state=result['state'], models=[r['model'] for r in reports], checked_at=result['checked_at'])))


def run():
    validate_seal()
    pre = load(ROOT/'preflight/RESULT.json')
    assert pre['state']=='passed' and pre['design_seal_sha256']==sha(ROOT/'DESIGN_SEAL.json')
    sys.path.insert(0,str(ROOT/'queues/phi/source'))
    import run_extension as w
    with w.worker_lock(ROOT/'controller.lock'):
        children = []
        start = now()
        for label,model,gpus in (('phi',PHI,'0,1,2,3,4,5'),('qwen',QWEN,'6,7')):
            qroot = ROOT/'queues'/label
            assert not (qroot/'merged.sqlite').exists() and not (qroot/'queue_state.json').exists()
            stream = (qroot/'scheduler.stdout.log').open('xb')
            command = [PYTHONS[model],'-B',str(qroot/'source/schedule_runs.py'),'--queue',str(qroot),
                       '--gpus',gpus,'--gpu-lock-dir',GPU_LOCK_DIR,'--poll-seconds','10']
            proc = subprocess.Popen(command, cwd=qroot, env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((label,proc,stream))
            write_json(qroot/'scheduler.launch.json',dict(started_at=now(),identity=identity(proc.pid), command=command))
        try:
            while True:
                status = collect_status()
                status.update(state='running', started_at=start, controller_identity=identity(os.getpid()))
                done = [p.poll() is not None for _,p,_ in children]
                if all(done):
                    terminals = {}
                    for label,proc,_ in children:
                        terminals[label] = dict(returncode=proc.returncode,finished_at=now())
                        write_json(ROOT/'queues'/label/'scheduler.terminal.json',terminals[label])
                    state = 'inference_completed' if all(x['returncode']==0 for x in terminals.values()) else 'failed'
                    status.update(state=state,finished_at=now(),queue_terminals=terminals)
                    atomic_json(ROOT/'STATUS.json', status)
                    write_json(ROOT/'CONTROLLER_TERMINAL.json',dict(state=state,started_at=start,finished_at=now(),queues=terminals))
                    return 0 if state=='inference_completed' else 1
                atomic_json(ROOT/'STATUS.json', status)
                time.sleep(15)
        finally:
            for _,_,stream in children:
                stream.close()


def collect_status(write_identities=True):
    found = []
    identities_path = ROOT/'monitor/PROCESS_IDENTITIES.json'
    known = load(identities_path) if identities_path.exists() else {}
    for label in ('phi','qwen'):
        qroot = ROOT/'queues'/label
        queue = load(qroot/'queue.json')
        for job in queue['jobs']:
            folder = qroot/job['run_dir']
            progress = load(folder/'progress.json') if (folder/'progress.json').exists() else {}
            terminal = load(folder/'terminal.json') if (folder/'terminal.json').exists() else None
            execution = load(folder/'execution.json') if (folder/'execution.json').exists() else {}
            worker = identity(execution['worker_pid']) if 'worker_pid' in execution else None
            if worker:
                assert str(folder/'source/run_extension.py') in worker['command']
                assert job['model'] in worker['command']
                key = str(worker['pid'])
                if key in known:
                    assert known[key]['start_ticks'] == worker['start_ticks'], 'Worker PID was reused'
                else:
                    known[key] = dict(worker, first_seen_at=now(), job=job['id'])
            elif not terminal and execution.get('worker_pid'):
                # A wrapper may be publishing its terminal file. Do not treat PID alone as completion.
                worker = dict(pid=execution['worker_pid'], state='absent_waiting_terminal')
            found.append(dict(job=job['id'],model=job['model'],planned=job['count'],
                saved=progress.get('saved',0),progress=progress,terminal=terminal,
                worker_identity=worker,gpu=execution.get('gpu')))
    if write_identities:
        atomic_json(identities_path, known)
    return dict(checked_at=now(),planned=2112,saved=sum(x['saved'] for x in found),jobs=found)


def status():
    report = collect_status(write_identities=False)
    report['controller_state'] = load(ROOT/'STATUS.json').get('state') if (ROOT/'STATUS.json').exists() else None
    gpu = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    report['gpu_state'] = gpu.stdout.strip().splitlines()
    atomic_json(ROOT/'monitor/LATEST.json',report)
    print(canonical(dict(checked_at=report['checked_at'],state=report['controller_state'],planned=2112,
        saved=report['saved'],jobs=[dict(job=x['job'],saved=x['saved'],planned=x['planned'],gpu=x['gpu'],
            state=x['terminal'] or x['progress'].get('state','starting'),worker=x['worker_identity'],
            elapsed_sec=x['progress'].get('elapsed_sec'),samples_per_sec=x['progress'].get('samples_per_sec')) for x in report['jobs']],
        gpu_state=report['gpu_state'])))


def launch():
    validate_seal()
    assert load(ROOT/'preflight/RESULT.json')['state']=='passed'
    assert not (ROOT/'LAUNCH.json').exists(), 'Refusing duplicate launch'
    stream = (ROOT/'controller.stdout.log').open('xb')
    command = [PYTHONS[PHI],'-B',str(Path(__file__)),'run']
    child = subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                             stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    stream.close()
    report = dict(started_at=now(),identity=identity(child.pid),command=command,design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'))
    write_json(ROOT/'LAUNCH.json',report)
    print(canonical(report))


def pack():
    validate_seal()
    assert load(ROOT/'CONTROLLER_TERMINAL.json')['state']=='inference_completed'
    destination = ROOT/'completed_snapshot.zip'
    assert not destination.exists()
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path == destination:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in rel.split('/') for part in ('runtime-cache','__pycache__')) or rel.endswith('.lock'):
            continue
        if 'results.backup' in rel or 'pending' in rel or rel.endswith('.zip'):
            continue
        files.append(path)
    index = {p.relative_to(ROOT).as_posix():dict(size=p.stat().st_size,sha256=sha(p)) for p in files}
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr('SNAPSHOT_INDEX.json',json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    write_json(ROOT/'SNAPSHOT_SEAL.json',dict(created_at=now(),file_count=len(files),size=destination.stat().st_size,sha256=sha(destination)))
    print(canonical(load(ROOT/'SNAPSHOT_SEAL.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('preflight','render','run','status','launch','pack'))
    parser.add_argument('--model',choices=(PHI,QWEN))
    args = parser.parse_args()
    try:
        if args.action=='render':
            assert args.model
            render(args.model)
        else:
            result = globals()[args.action]()
            if args.action=='run':
                sys.exit(result)
    except BaseException as exc:
        if not isinstance(exc,SystemExit):
            atomic_json(ROOT/'ADMIN_FAILURE.json',dict(time=now(),action=args.action,type=type(exc).__name__,message=str(exc)))
        raise
