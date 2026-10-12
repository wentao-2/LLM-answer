"""Owned GPU supervisor for the frozen R5 inference supplement."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import copy
import json
import os
import subprocess
import sys
import time

import inference_helpers as h
sys.path.insert(0,str(h.ROOT/'source'))
import analyze_extension as analyzer
import build_extension as base

ROOT=h.ROOT
PYTHON='python'


def now():
    return datetime.now(timezone.utc).isoformat()


def execute(job,gpu):
    folder=ROOT/'inference/jobs'/job['name']
    if (folder/'terminal.json').exists():
        raise ValueError('Refuse a blind restart of a terminal job '+job['name'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    started=now()
    with (folder/'worker.log').open('wb') as stream:
        child=subprocess.Popen([PYTHON,str(folder/'source/run_extension.py'),'--config',str(folder/'config.json'),'--model',job['model']],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        h.atomic_json(folder/'launch.json',dict(job=job,gpu=gpu,pid=child.pid,started_at=started))
        rc=child.wait()
    terminal=dict(job=job['name'],gpu=gpu,started_at=started,finished_at=now(),returncode=rc)
    h.atomic_json(folder/'terminal.json',terminal)
    if rc:
        raise RuntimeError('Worker failed: '+job['name']+' rc='+str(rc))
    return job


def model_inventory():
    inventory={}
    for model,spec in h.MODELS.items():
        directory=Path(spec['path'])
        files={}
        for p in sorted(directory.iterdir()):
            if p.is_file() and (p.suffix in ('.json','.jinja') or p.name.endswith('.safetensors')):
                files[p.name]=dict(size=p.stat().st_size,sha256=h.digest(p))
        assert any(n.endswith('.safetensors') for n in files)
        inventory[model]=dict(spec=spec,files=files)
    h.atomic_json(ROOT/'inference/MODEL_INVENTORY.json',inventory)


def main():
    status_path=ROOT/'inference/STATUS.json'
    if status_path.exists():
        raise ValueError('Existing supervisor state requires inspection; no blind restart')
    seal=json.loads((ROOT/'inference/DESIGN_SEAL.json').read_text())
    assert h.digest(ROOT/'INFERENCE_PROTOCOL.md')==seal['protocol_sha256']
    assert h.digest(ROOT/'inference_helpers.py')==seal['helper_sha256']
    assert h.digest(Path(__file__))==seal['controller_sha256']
    assert h.digest(ROOT/'PRE_EXECUTION_AMENDMENT.md')==seal['amendment_sha256']
    jobs=json.loads((ROOT/'inference/INITIAL_JOBS.json').read_text())
    stages={'phi-gate':[j for j in jobs if j['stage']=='phi-gate'],
            'qwen-parents':[j for j in jobs if j['stage']=='qwen-parents']}
    history={}
    h.atomic_json(ROOT/'inference/STARTING.json',dict(pid=os.getpid(),started_at=now(),state='recording_model_inventory'))
    model_inventory()
    for stage in stages:
        history[stage]=dict(state='prepared',n_planned=sum(j['count'] for j in stages[stage]))
    pending=[]
    for stage,gpus in (('phi-gate',range(4)),('qwen-parents',range(4,8))):
        pending.extend((j,gpu) for j,gpu in zip(stages[stage],gpus))
    done=set()
    stage_done=set()
    active={}
    started=now()
    with ThreadPoolExecutor(max_workers=8) as pool:
        def submit_stage(stage,rows,model,gpus):
            created=h.prepare_jobs(stage,rows,model,len(gpus))
            stages[stage]=created
            history[stage]=dict(state='prepared',n_planned=len(rows))
            pending.extend(zip(created,gpus))
        while pending or active:
            occupied={gpu for _,gpu in active.values()}
            for job,gpu in list(pending):
                if gpu not in occupied:
                    pending.remove((job,gpu))
                    future=pool.submit(execute,job,gpu)
                    active[future]=(job,gpu)
                    occupied.add(gpu)
                    history[job['stage']]['state']='running'
            for future,(job,gpu) in list(active.items()):
                if future.done():
                    future.result()
                    done.add(job['name'])
                    del active[future]
            for stage,stage_jobs in list(stages.items()):
                if stage in stage_done or not all(j['name'] in done for j in stage_jobs):
                    continue
                rows,database=h.merge_stage(stage,stage_jobs)
                stage_done.add(stage)
                history[stage].update(state='completed',status_counts=dict(Counter(r['status'] for r in rows)))
                manifest=ROOT/'inference/manifests'/f'{stage}.jsonl'
                if stage=='phi-gate':
                    gate=analyzer.analyze_gate(base.read_jsonl(manifest),database,'phi4-reasoning-r5')
                    h.atomic_json(ROOT/'inference/results/phi-gate-analysis.json',gate)
                    history[stage]['passes']=gate['passes']
                    if gate['passes']:
                        cross=[h.seal_item(r,'phi4-reasoning-r5') for r in base.read_jsonl(ROOT/'evidence/R4/cross-family.jsonl')]
                        parents=[]
                        for row in base.read_jsonl(ROOT/'evidence/R4/history-stage1.jsonl'):
                            x=copy.deepcopy(row); x['parent_condition']='numeric_native'
                            parents.append(h.seal_item(x,'phi4-reasoning-r5'))
                        h.write_manifest(ROOT/'inference/manifests/phi-cross.jsonl',cross)
                        h.write_manifest(ROOT/'inference/manifests/phi-parents.jsonl',parents)
                        submit_stage('phi-main',cross+parents,'phi4-reasoning-r5',list(range(4)))
                    else:
                        history['phi-downstream']=dict(state='not_constructed_failed_gate',reason='Frozen qualification criteria failed')
                elif stage=='qwen-parents':
                    parents=base.read_jsonl(manifest)
                    children=h.history_followups('qwen-history',parents,rows,'qwen3-8b-r5')
                    h.write_manifest(ROOT/'inference/manifests/qwen-history.jsonl',children)
                    paraphrases=base.read_jsonl(ROOT/'inference/manifests/paraphrase.jsonl')+base.read_jsonl(ROOT/'inference/manifests/paraphrase-neutral.jsonl')
                    submit_stage('qwen-main',children+paraphrases,'qwen3-8b-r5',list(range(4,8)))
                elif stage=='phi-main':
                    parents=base.read_jsonl(ROOT/'inference/manifests/phi-parents.jsonl')
                    parent_ids={r['id'] for r in parents}
                    producer_rows=[r for r in rows if r['item_id'] in parent_ids]
                    children=h.history_followups('phi-history',parents,producer_rows,'phi4-reasoning-r5')
                    if children:
                        submit_stage('phi-history',children,'phi4-reasoning-r5',list(range(4)))
                    else:
                        history['phi-history']=dict(state='not_constructed_no_valid_parents')
            progress=[]
            for job,gpu in active.values():
                p=ROOT/'inference/jobs'/job['name']/'progress.json'
                value=json.loads(p.read_text()) if p.exists() else dict(state='starting')
                progress.append(dict(job=job['name'],gpu=gpu,**value))
            state=dict(state='running',started_at=started,updated_at=now(),stages=history,
                       active_jobs=progress,completed_jobs=len(done),pending_jobs=len(pending))
            h.atomic_json(status_path,state)
            time.sleep(10)
    h.atomic_json(status_path,dict(state='completed',started_at=started,finished_at=now(),stages=history,
                                  completed_jobs=len(done),active_jobs=[],pending_jobs=0))


if __name__=='__main__':
    try:
        main()
    except BaseException as e:
        h.atomic_json(ROOT/'inference/FAILURE.json',dict(time=now(),type=type(e).__name__,message=str(e)))
        raise
