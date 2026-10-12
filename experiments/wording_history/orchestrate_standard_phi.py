"""Independent, sealed standard Phi-4 qualification and downstream study."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import copy, hashlib, json, os, random, shutil, sqlite3, subprocess, sys, time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import build_extension as b
import run_extension as w
import analyze_extension as a
MODEL='phi4-standard-r6'
STUDY=ROOT/'standard-phi/study'
PYTHON='python'
def now(): return datetime.now(timezone.utc).isoformat()
def digest(p): return w.digest(p)
def dump(p,v): w.atomic_json(p,v)
def seal(x):
    x=copy.deepcopy(x)
    x['origin_item_id']=x.pop('id',None)
    x.update(generator_version='i16-review-standard-phi-r6',model_scope=[MODEL],
             max_new_tokens=8192,max_input_tokens=8192,require_think_close=False)
    x.pop('chat_template_kwargs',None)
    x['id']=hashlib.sha256(b.canonical(x).encode()).hexdigest()[:24]
    return x
def manifest(path,rows): b.write_manifest(path,rows)
def prepare(stage,rows):
    assert len({r['id'] for r in rows})==len(rows)
    manifest(STUDY/'manifests'/f'{stage}.jsonl',rows)
    shuffled=list(rows); random.Random(16101101).shuffle(shuffled)
    jobs=[]
    for shard in range(4):
        items=shuffled[shard::4]
        if not items: continue
        folder=STUDY/'jobs'/f'{stage}-s{shard:02d}'
        assert not folder.exists()
        (folder/'source').mkdir(parents=True)
        shutil.copy2(ROOT/'source/run_extension.py',folder/'source/run_extension.py')
        protocols={}
        for name in ('STANDARD_PHI_PROTOCOL.md','RUNTIME_AMENDMENT.md'):
            shutil.copy2(ROOT/name,folder/name); protocols[name]=digest(folder/name)
        manifest(folder/'manifest.jsonl',items)
        model=dict(name=MODEL,path=str(ROOT/'standard-phi/checkpoint'),
            revision='2db69c1c3e91a05d2c64a3185acfbaf36f744e25',batch_size=2,
            require_think_close=False,mode='native_nonreasoning',chat_template_kwargs={})
        config=dict(name=folder.name,manifest='manifest.jsonl',experiment_protocol='STANDARD_PHI_PROTOCOL.md',
            seed=1601,batch_size=2,max_input_tokens=8192,max_new_tokens=8192,dtype='bfloat16',attention='sdpa',
            do_sample=False,num_beams=1,store_token_ids=True,cpu_threads=4,readout='probability',models=[model])
        dump(folder/'config.json',config)
        frozen=dict(config_sha256=digest(folder/'config.json'),manifest_sha256=digest(folder/'manifest.jsonl'),
            source_hashes={'run_extension.py':digest(folder/'source/run_extension.py')},protocol_hashes=protocols)
        dump(folder/'frozen.json',frozen)
        jobs.append(dict(name=folder.name,gpu=shard,count=len(items),stage=stage,model=MODEL,
                         frozen_sha256=digest(folder/'frozen.json')))
    return jobs
def execute(job):
    folder=STUDY/'jobs'/job['name']
    assert not (folder/'terminal.json').exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    start=now()
    with (folder/'worker.log').open('wb') as stream:
        child=subprocess.Popen([PYTHON,str(folder/'source/run_extension.py'),'--config',str(folder/'config.json'),
                                '--model',MODEL],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        dump(folder/'launch.json',dict(job=job,pid=child.pid,started_at=start))
        rc=child.wait()
    dump(folder/'terminal.json',dict(returncode=rc,started_at=start,finished_at=now(),job=job))
    if rc: raise RuntimeError(f"Worker {job['name']} failed: {rc}")
def merge(stage,jobs):
    path=STUDY/'results'/f'{stage}.sqlite'; assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    db=w.connect_db(path); rows=[]; ids=set(); hashes={}
    for job in jobs:
        folder=STUDY/'jobs'/job['name']
        assert json.loads((folder/'terminal.json').read_text())['returncode']==0
        w.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
        source=sqlite3.connect((folder/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True); source.row_factory=sqlite3.Row
        chunk=[dict(r) for r in source.execute('SELECT * FROM results')]; source.close()
        expected={r['id'] for r in b.read_jsonl(folder/'manifest.jsonl')}
        assert len(chunk)==job['count'] and {r['item_id'] for r in chunk}==expected
        assert not ids.intersection(expected); ids.update(expected)
        for row in chunk: w.insert_result(db,[row[c] for c in w.RESULT_COLUMNS])
        rows+=chunk; hashes[job['name']]=digest(folder/'results.sqlite')
    db.close()
    assert ids=={r['id'] for r in b.read_jsonl(STUDY/'manifests'/f'{stage}.jsonl')}
    dump(path.with_suffix('.seal.json'),dict(stage=stage,n_records=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
        source_databases=hashes,database_sha256=digest(path),manifest_sha256=digest(STUDY/'manifests'/f'{stage}.jsonl')))
    return rows,path
def runstage(stage,rows):
    jobs=prepare(stage,rows)
    dump(STUDY/'STATUS.json',dict(state='running',stage=stage,n_planned=len(rows),jobs=jobs,updated_at=now()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        for future in futures: future.result()
    return merge(stage,jobs)
def main():
    assert not STUDY.exists(),'No blind supervisor restart'
    verified=json.loads((ROOT/'standard-phi/DOWNLOAD_VERIFIED.json').read_text())
    # The downloader recorded actual complete file hashes; freeze this inventory as an input.
    STUDY.mkdir()
    dump(STUDY/'DESIGN_SEAL.json',dict(controller_sha256=digest(Path(__file__)),
        source_sha256=digest(ROOT/'source/run_extension.py'),protocol_sha256=digest(ROOT/'STANDARD_PHI_PROTOCOL.md'),
        amendment_sha256=digest(ROOT/'RUNTIME_AMENDMENT.md'),checkpoint_inventory_sha256=digest(ROOT/'standard-phi/DOWNLOAD_VERIFIED.json'),
        scientific_inference_started=False))
    gate=[seal(x) for x in b.read_jsonl(ROOT/'evidence/R4/gate.jsonl')]
    rows,db=runstage('gate',gate)
    report=a.analyze_gate(gate,db,MODEL); dump(STUDY/'results/gate-analysis.json',report)
    if not report['passes']:
        dump(STUDY/'STATUS.json',dict(state='completed',gate_passes=False,downstream='not_constructed_failed_gate',finished_at=now()))
        return
    cross=[seal(x) for x in b.read_jsonl(ROOT/'evidence/R4/cross-family.jsonl')]
    parents=[]
    for x in b.read_jsonl(ROOT/'evidence/R4/history-stage1.jsonl'):
        x=copy.deepcopy(x); x['parent_condition']='numeric_native'; parents.append(seal(x))
    manifest(STUDY/'manifests/cross.jsonl',cross); manifest(STUDY/'manifests/parents.jsonl',parents)
    rows,db=runstage('main',cross+parents)
    parent_ids={r['id'] for r in parents}; producers=[r for r in rows if r['item_id'] in parent_ids]
    children,cohort=b.history_children(parents,producers,[MODEL])
    for x in children: x['parent_condition']='numeric_native'
    children=[seal(x) for x in children]
    dump(STUDY/'results/history-cohort.json',cohort)
    if children: runstage('history',children)
    dump(STUDY/'STATUS.json',dict(state='completed',gate_passes=True,finished_at=now(),planned_cross=len(cross),
                                 planned_parents=len(parents),constructed_children=len(children)))
if __name__=='__main__':
    try: main()
    except BaseException as e:
        dump(STUDY/'FAILURE.json',dict(time=now(),type=type(e).__name__,message=str(e)))
        raise
