"""Durable eight-GPU orchestration of a frozen, nonadaptive two-stage study."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from common import (ROOT,PHI,QWEN,atomic_json,canonical,load,now,read_jsonl,sha,
                    text_hash,validate_seal,write_json)
from build_design import construct

PYTHONS={PHI:'/workspace/codex-auto-research/env/bin/python',
         QWEN:'/workspace/i16-review-supplement-20261011/fast_runtime/vllm085/venv/bin/python'}
GPU_LOCK_DIR='/tmp/i16-extension-gpu-locks'
LABELS={'phi':PHI,'qwen':QWEN}
GPUS={'phi':'0,1,2,3,4,5','qwen':'6,7'}


def identity(pid):
    path=Path('/proc')/str(pid)
    try:
        stat=(path/'stat').read_text()
        tail=stat[stat.rfind(')')+2:].split()
        cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
        return dict(pid=int(pid),state=tail[0],start_ticks=int(tail[19]),command=cmd,exe=os.readlink(path/'exe'))
    except (FileNotFoundError,ProcessLookupError):
        return None


def render(model,manifest,output):
    from transformers import AutoTokenizer
    config=load(ROOT/'inputs/runtime_configs.json')[model]
    expected=load(ROOT/'inputs/runtime_environments.json')[model]
    expected=expected.get('packages',expected)
    spec=config['models'][0]
    packages={name:importlib.metadata.version(name) for name in
              (('torch','transformers','vllm','tokenizers','huggingface-hub') if model==QWEN
               else ('torch','transformers','accelerate'))}
    for name,version in packages.items():
        if name in expected:
            assert version==expected[name],'Native runtime version changed: '+name
    if model==QWEN:
        assert packages['vllm']=='0.8.5' and packages['transformers']=='4.51.3'
    tokenizer=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True,trust_remote_code=False)
    assert tokenizer.chat_template
    model_cfg=load(Path(spec['path'])/'config.json')
    context=model_cfg.get('max_position_embeddings',model_cfg.get('text_config',{}).get('max_position_embeddings'))
    rows=[x for x in read_jsonl(manifest) if x['model_scope']==[model]]
    ledger=[]
    for item in rows:
        messages=item.get('messages',[dict(role='user',content=item.get('prompt'))])
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                                             **item.get('chat_template_kwargs',{}))
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        assert len(ids)<=item['max_input_tokens'],'Valid parent exceeds frozen input budget: '+item['id']
        assert len(ids)+item['max_new_tokens']<=context
        if model==QWEN:
            assert len(ids)+item['max_new_tokens']<=config['max_model_len']
        ledger.append(dict(item_id=item['id'],input_tokens=len(ids),rendered_prompt_sha256=text_hash(prompt),
                           rendered_token_ids_sha256=text_hash(canonical(ids))))
    write_json(output,dict(model=model,packages=packages,context_limit=context,n=len(ledger),
        min_input_tokens=min((x['input_tokens'] for x in ledger),default=None),
        max_input_tokens=max((x['input_tokens'] for x in ledger),default=None),ledger=ledger,
        native_chat_template_sha256=text_hash(canonical(tokenizer.chat_template)),checked_at=now()))


def render_subprocess(model,manifest,output):
    command=[PYTHONS[model],'-B',str(Path(__file__)),'render','--model',model,
             '--manifest',str(manifest),'--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'))
    write_json(output.with_name(output.name.replace('-render.json','-render-process.json')),
               dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,checked_at=now()))
    if result.returncode:
        raise RuntimeError('Native rendering failed: '+result.stderr[-6000:])


def verify_model(model):
    config=load(ROOT/'inputs/runtime_configs.json')[model]
    spec=config['models'][0]
    inventory=load(ROOT/'inputs/model_inventories.json')[model]
    revision=inventory.get('revision',inventory.get('spec',{}).get('revision'))
    assert revision==spec['revision']
    checkpoint=Path(spec['path'])
    checked={}
    for name,expected in inventory['files'].items():
        file=checkpoint/name
        assert file.is_file() and file.stat().st_size==expected['size'],name
        digest=sha(file)
        assert digest==expected['sha256'],'Complete checkpoint SHA mismatch: '+name
        checked[name]=dict(size=file.stat().st_size,sha256=digest)
    index=load(checkpoint/'model.safetensors.index.json')
    assert set(index['weight_map'].values())=={x for x in checked if x.endswith('.safetensors')}
    label='phi' if model==PHI else 'qwen'
    output=ROOT/'preflight'/(label+'-static-render.json')
    render_subprocess(model,ROOT/'manifests/preflight_static.jsonl',output)
    return dict(model=model,revision=revision,checkpoint=spec['path'],files=checked,render_sha256=sha(output))


def preflight():
    validate_seal()
    assert not (ROOT/'preflight/RESULT.json').exists()
    with ThreadPoolExecutor(max_workers=2) as pool:
        models=list(pool.map(verify_model,(PHI,QWEN)))
    report=dict(state='passed',checked_at=now(),models=models,
                design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'),scientific_inference_started=False)
    write_json(ROOT/'preflight/RESULT.json',report)
    print(canonical(dict(state='passed',checked_at=report['checked_at'],models=[x['model'] for x in models])))


def spawn_queue(label,phase):
    qroot=ROOT/'queues'/(phase+'_'+label)
    assert not (qroot/'scheduler.launch.json').exists(),'Refusing duplicate scheduler'
    command=[PYTHONS[LABELS[label]],'-B',str(qroot/'source/schedule_runs.py'),
             '--queue',str(qroot),'--gpus',GPUS[label],'--gpu-lock-dir',GPU_LOCK_DIR,'--poll-seconds','10']
    stream=(qroot/'scheduler.stdout.log').open('xb')
    proc=subprocess.Popen(command,cwd=qroot,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                          stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    stream.close()
    ident=identity(proc.pid)
    assert ident and str(qroot/'source/schedule_runs.py') in ident['command']
    write_json(qroot/'scheduler.launch.json',dict(started_at=now(),identity=ident,command=command))
    return proc


def collect_status(save_identities=True):
    known_file=ROOT/'monitor/PROCESS_IDENTITIES.json'
    known=load(known_file) if known_file.exists() else {}
    found=[]
    for qroot in sorted((ROOT/'queues').glob('*')):
        if not (qroot/'queue.frozen.json').exists():
            continue
        queue=load(qroot/'queue.json')
        for job in queue['jobs']:
            folder=qroot/job['run_dir']
            progress=load(folder/'progress.json') if (folder/'progress.json').exists() else {}
            terminal=load(folder/'terminal.json') if (folder/'terminal.json').exists() else None
            execution=load(folder/'execution.json') if (folder/'execution.json').exists() else {}
            ident=identity(execution['worker_pid']) if 'worker_pid' in execution else None
            if ident:
                assert str(folder/'source/run_extension.py') in ident['command']
                assert job['model'] in ident['command']
                key=str(ident['pid'])
                if key in known:
                    assert known[key]['start_ticks']==ident['start_ticks'],'Worker PID was reused'
                else:
                    known[key]=dict(ident,first_seen_at=now(),job=job['id'])
            elif execution.get('worker_pid') and not terminal:
                ident=dict(pid=execution['worker_pid'],state='absent_waiting_terminal')
            found.append(dict(queue=qroot.name,job=job['id'],model=job['model'],planned=job['count'],
                saved=progress.get('saved',job['count'] if terminal and terminal['returncode']==0 else 0),
                progress=progress,terminal=terminal,worker_identity=ident,gpu=execution.get('gpu')))
    if save_identities:
        atomic_json(known_file,known)
    return dict(checked_at=now(),max_planned=2880,constructed_planned=sum(x['planned'] for x in found),
                saved=sum(x['saved'] for x in found),jobs=found)


def complete_queue(label,phase,proc):
    report=dict(returncode=proc.returncode,finished_at=now())
    write_json(ROOT/'queues'/(phase+'_'+label)/'scheduler.terminal.json',report)
    if proc.returncode:
        raise RuntimeError('Queue failed: '+phase+'_'+label)
    return report


def finalize():
    for filename,output in (('analyze_results.py','analysis/RESULTS.json'),('audit_results.py','audit/AUDIT.json')):
        assert not (ROOT/output).exists(),'Existing final artifact requires inspection: '+output
        log=ROOT/'monitor'/(filename+'.log')
        with log.open('x',encoding='utf-8') as stream:
            command=[PYTHONS[PHI],'-B',str(ROOT/'source'/filename)]
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        write_json(ROOT/'monitor'/(filename+'.terminal.json'),dict(command=command,
            returncode=result.returncode,finished_at=now()))
        if result.returncode:
            raise RuntimeError('Final analysis/audit failed: '+filename)
    assert load(ROOT/'audit/AUDIT.json')['state']=='passed'
    write_json(ROOT/'FINAL_SEAL.json',dict(completed_at=now(),state='audited',
        design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'),
        analysis_sha256=sha(ROOT/'analysis/RESULTS.json'),audit_sha256=sha(ROOT/'audit/AUDIT.json')))


def run():
    validate_seal()
    pre=load(ROOT/'preflight/RESULT.json')
    assert pre['state']=='passed' and pre['design_seal_sha256']==sha(ROOT/'DESIGN_SEAL.json')
    sys.path.insert(0,str(ROOT/'worker_sources/phi'))
    import run_extension as native
    with native.worker_lock(ROOT/'controller.lock'):
        started=now()
        active={label:('parents',spawn_queue(label,'parents')) for label in LABELS}
        done={}
        failures=[]
        while active:
            for label,(phase,proc) in list(active.items()):
                if proc.poll() is None:
                    continue
                del active[label]
                try:
                    complete_queue(label,phase,proc)
                    if phase=='parents':
                        # This reads only the complete parent DB. Both scientific
                        # stages and all four interventions were frozen at launch.
                        construct(label)
                        construction=load(ROOT/'cohorts'/label/'CONSTRUCTION.json')
                        if construction['allocation']:
                            qroot=ROOT/'queues'/('followups_'+label)
                            rows=[]
                            for job in load(qroot/'queue.json')['jobs']:
                                rows+=read_jsonl(qroot/job['run_dir']/'manifest.jsonl')
                            manifest=ROOT/'cohorts'/label/'followups_preflight.jsonl'
                            from common import write_jsonl
                            write_jsonl(manifest,rows)
                            output=ROOT/'preflight'/(label+'-followups-render.json')
                            render_subprocess(LABELS[label],manifest,output)
                            write_json(ROOT/'cohorts'/label/'RENDER_SEAL.json',dict(checked_at=now(),
                                construction_sha256=sha(ROOT/'cohorts'/label/'CONSTRUCTION.json'),
                                manifest_sha256=sha(manifest),render_sha256=sha(output)))
                            active[label]=('followups',spawn_queue(label,'followups'))
                        else:
                            done[label]='no_valid_parents'
                    else:
                        done[label]='inference_completed'
                except Exception as exc:
                    failure=dict(label=label,phase=phase,time=now(),type=type(exc).__name__,message=str(exc),
                                 traceback=traceback.format_exc())
                    failures.append(failure)
                    write_json(ROOT/'monitor'/(label+'-'+phase+'-failure.json'),failure)
                    done[label]='failed'
            status=collect_status()
            status.update(state='running' if active else ('failed' if failures else 'inference_completed'),
                started_at=started,controller_identity=identity(os.getpid()),
                active_phases={label:phase for label,(phase,_) in active.items()},done=done,failures=failures)
            atomic_json(ROOT/'STATUS.json',status)
            if active:
                time.sleep(10)
        terminal=dict(state='failed' if failures else 'inference_completed',started_at=started,
                      finished_at=now(),done=done,failures=failures)
        write_json(ROOT/'INFERENCE_TERMINAL.json',terminal)
        if failures:
            write_json(ROOT/'CONTROLLER_TERMINAL.json',terminal)
            return 1
        atomic_json(ROOT/'STATUS.json',dict(status,state='analyzing'))
        finalize()
        terminal.update(state='audited',completed_at=now(),final_seal_sha256=sha(ROOT/'FINAL_SEAL.json'))
        write_json(ROOT/'CONTROLLER_TERMINAL.json',terminal)
        atomic_json(ROOT/'STATUS.json',dict(status,state='audited',completed_at=now()))
        return 0


def launch():
    validate_seal()
    assert load(ROOT/'preflight/RESULT.json')['state']=='passed'
    sys.path.insert(0,str(ROOT/'worker_sources/phi'))
    import run_extension as native
    with native.worker_lock(ROOT/'launch.lock'):
        assert not (ROOT/'LAUNCH.json').exists(),'Refusing duplicate launch'
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,check=True)
        rows=[x.split(',') for x in gpu.stdout.strip().splitlines()]
        assert len(rows)==8 and all(int(x[1].strip())<100 for x in rows),'GPU occupancy changed before launch'
        write_json(ROOT/'preflight/GPU_AT_LAUNCH.json',dict(checked_at=now(),rows=gpu.stdout.strip().splitlines()))
        stream=(ROOT/'controller.stdout.log').open('xb')
        command=[PYTHONS[PHI],'-B',str(Path(__file__)),'run']
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        stream.close()
        report=dict(started_at=now(),identity=identity(child.pid),command=command,
                    design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'),max_calls=2880)
        write_json(ROOT/'LAUNCH.json',report)
        print(canonical(report))


def status():
    report=collect_status(save_identities=False)
    report['controller_state']=load(ROOT/'STATUS.json').get('state') if (ROOT/'STATUS.json').exists() else None
    launch=load(ROOT/'LAUNCH.json') if (ROOT/'LAUNCH.json').exists() else None
    live=identity(launch['identity']['pid']) if launch else None
    if live:
        assert live['start_ticks']==launch['identity']['start_ticks']
        assert str(ROOT/'source/remote_admin.py') in live['command'] and live['command'].endswith(' run')
    report['controller_identity']=live
    report['controller_terminal']=load(ROOT/'CONTROLLER_TERMINAL.json') if (ROOT/'CONTROLLER_TERMINAL.json').exists() else None
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],
                        capture_output=True,text=True)
    report['gpu_state']=gpu.stdout.strip().splitlines()
    atomic_json(ROOT/'monitor/LATEST.json',report)
    print(canonical(dict(checked_at=report['checked_at'],state=report['controller_state'],
        max_planned=2880,constructed_planned=report['constructed_planned'],saved=report['saved'],
        controller=live,terminal=report['controller_terminal'],
        jobs=[dict(queue=x['queue'],job=x['job'],saved=x['saved'],planned=x['planned'],gpu=x['gpu'],
              state=x['terminal'] or x['progress'].get('state','starting'),worker=x['worker_identity'],
              elapsed_sec=x['progress'].get('elapsed_sec'),samples_per_sec=x['progress'].get('samples_per_sec')) for x in report['jobs']],
        gpu_state=report['gpu_state'])))


def pack():
    validate_seal()
    assert load(ROOT/'CONTROLLER_TERMINAL.json')['state']=='audited'
    assert load(ROOT/'audit/AUDIT.json')['state']=='passed'
    destination=ROOT/'snapshot.zip'
    assert not destination.exists()
    files=[]
    for file in sorted(ROOT.rglob('*')):
        if not file.is_file():
            continue
        rel=file.relative_to(ROOT).as_posix()
        if any(x in file.parts for x in ('runtime-cache','__pycache__')) or rel.endswith('.lock'):
            continue
        if 'results.backup' in rel or 'pending' in rel or rel.endswith('.zip'):
            continue
        files.append(file)
    index={p.relative_to(ROOT).as_posix():dict(size=p.stat().st_size,sha256=sha(p)) for p in files}
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files:
            archive.write(p,p.relative_to(ROOT).as_posix())
        archive.writestr('SNAPSHOT_INDEX.json',json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    write_json(ROOT/'SNAPSHOT_SEAL.json',dict(created_at=now(),file_count=len(files),
                size=destination.stat().st_size,sha256=sha(destination)))
    print(canonical(load(ROOT/'SNAPSHOT_SEAL.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('preflight','render','run','launch','status','pack'))
    parser.add_argument('--model',choices=(PHI,QWEN))
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        if args.action=='render':
            assert args.model and args.manifest and args.output
            render(args.model,args.manifest,args.output)
        else:
            result=globals()[args.action]()
            if args.action=='run':
                sys.exit(result)
    except BaseException as exc:
        if not isinstance(exc,SystemExit):
            atomic_json(ROOT/'ADMIN_FAILURE.json',dict(time=now(),action=args.action,
                        type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            if args.action=='run' and not (ROOT/'CONTROLLER_TERMINAL.json').exists():
                write_json(ROOT/'CONTROLLER_TERMINAL.json',dict(state='administrative_failure',
                           finished_at=now(),type=type(exc).__name__,message=str(exc)))
        raise
