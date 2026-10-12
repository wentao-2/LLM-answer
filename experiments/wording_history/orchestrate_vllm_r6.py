"""Prepare and run a separate complete, uniform-runtime Qwen R6 design."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import copy,hashlib,json,os,random,shutil,sqlite3,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import build_extension as b
import run_extension as w
MODEL='qwen3-8b-r6';STUDY=ROOT/'vllm-r6'
PYTHON=str(ROOT/'fast_runtime/vllm085/venv/bin/python')
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,v):w.atomic_json(p,v)
def digest(p):return w.digest(p)
def prepare():
    assert not STUDY.exists(),'No blind restart of a sealed study'
    probe=json.loads((ROOT/'fast_runtime/probe/RESULT.json').read_text())
    assert probe['state']=='passed' and probe['scientific'] is False
    assert probe['packages']['vllm']=='0.8.5' and probe['packages']['transformers']=='4.51.3'
    STUDY.mkdir()
    inventory=json.loads((ROOT/'inference/MODEL_INVENTORY.json').read_text())['qwen3-8b-r5']
    assert inventory['spec']['path']=='models/Qwen/Qwen3-8B'
    for name,expected in inventory['files'].items():
        path=Path(inventory['spec']['path'])/name
        assert path.stat().st_size==expected['size'] and digest(path)==expected['sha256'],name
    dump(STUDY/'CHECKPOINT_VERIFIED.json',inventory)
    source=b.read_jsonl(ROOT/'inference/manifests/qwen-history.jsonl')+b.read_jsonl(ROOT/'inference/manifests/paraphrase.jsonl')+b.read_jsonl(ROOT/'inference/manifests/paraphrase-neutral.jsonl')
    assert len(source)==4000 and len({r['id'] for r in source})==4000
    rows=[]
    for old in source:
        x=copy.deepcopy(old);x['runtime_replication_of_r5_id']=x.pop('id')
        x.update(model_scope=[MODEL],generator_version='review-vllm-r6',backend='vllm-0.8.5')
        assert x['max_new_tokens']==12288 and x['max_input_tokens']==8192
        assert x['chat_template_kwargs']==dict(enable_thinking=True) and x['require_think_close'] is True
        x['id']=hashlib.sha256(b.canonical(x).encode()).hexdigest()[:24];rows.append(x)
    b.write_manifest(STUDY/'manifests/main.jsonl',rows)
    b.write_manifest(STUDY/'manifests/history.jsonl',[r for r in rows if r['module']=='history_stage2'])
    b.write_manifest(STUDY/'manifests/paraphrases.jsonl',[r for r in rows if r['module']=='paraphrase'])
    shuffled=list(rows);random.Random(16101101).shuffle(shuffled)
    jobs=[]
    for shard,gpu in enumerate(range(4,8)):
        items=shuffled[shard::4];folder=STUDY/'jobs'/f'main-s{shard:02d}'
        (folder/'source').mkdir(parents=True)
        shutil.copy2(ROOT/'vllm_worker.py',folder/'source/run_extension.py')
        shutil.copy2(ROOT/'source/run_extension.py',folder/'source/native_worker.py')
        protocols={}
        for name in ('VLLM_R6_PROTOCOL.md','RUNTIME_AMENDMENT.md','R6_ALLOCATION_NOTE.md'):
            shutil.copy2(ROOT/name,folder/name);protocols[name]=digest(folder/name)
        b.write_manifest(folder/'manifest.jsonl',items)
        model=dict(name=MODEL,path='models/Qwen/Qwen3-8B',revision='b968826d9c46dd6066d109eabc6255188de91218',
            batch_size=32,chat_template_kwargs=dict(enable_thinking=True),require_think_close=True)
        config=dict(name=folder.name,manifest='manifest.jsonl',experiment_protocol='VLLM_R6_PROTOCOL.md',
            seed=1601,temperature=0,max_input_tokens=8192,max_new_tokens=12288,max_model_len=20480,
            dtype='bfloat16',backend='vllm-0.8.5',chunk_size=64,readout='probability',models=[model])
        dump(folder/'config.json',config)
        frozen=dict(config_sha256=digest(folder/'config.json'),manifest_sha256=digest(folder/'manifest.jsonl'),
            source_hashes={p.name:digest(p) for p in (folder/'source').glob('*.py')},protocol_hashes=protocols)
        dump(folder/'frozen.json',frozen)
        w.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
        jobs.append(dict(name=folder.name,gpu=gpu,count=len(items),model=MODEL,frozen_sha256=digest(folder/'frozen.json')))
    dump(STUDY/'DESIGN_SEAL.json',dict(n_planned=len(rows),jobs=jobs,controller_sha256=digest(Path(__file__)),
        analysis_sha256=digest(ROOT/'analyze_new_inference.py'),probe_sha256=digest(ROOT/'fast_runtime/probe/RESULT.json'),
        package_list_sha256=digest(ROOT/'fast_runtime/vllm085/packages-final.txt'),
        source_manifests={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'inference/manifests').glob('*.jsonl')},
        derived_manifests={p.name:digest(p) for p in (STUDY/'manifests').glob('*.jsonl')},scientific_inference_started=False))
    return jobs,rows
def execute(job):
    folder=STUDY/'jobs'/job['name'];assert not (folder/'terminal.json').exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
    start=now()
    with (folder/'worker.log').open('wb') as stream:
        child=subprocess.Popen([PYTHON,str(folder/'source/run_extension.py'),'--config',str(folder/'config.json'),
                                '--model',MODEL],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        dump(folder/'launch.json',dict(job=job,pid=child.pid,started_at=start));rc=child.wait()
    dump(folder/'terminal.json',dict(job=job,returncode=rc,started_at=start,finished_at=now()))
    if rc:raise RuntimeError(f"Worker {job['name']} failed: {rc}")
def main():
    jobs,rows=prepare();start=now()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        while not all(f.done() for f in futures):
            progress=[]
            for job in jobs:
                folder=STUDY/'jobs'/job['name']
                path=folder/'progress.json'
                progress.append(dict(job=job['name'],gpu=job['gpu'],**(json.loads(path.read_text()) if path.exists() else dict(state='starting'))))
            dump(STUDY/'STATUS.json',dict(state='running',started_at=start,updated_at=now(),active=progress))
            time.sleep(10)
        for f in futures:f.result()
    destination=STUDY/'results/main.sqlite';destination.parent.mkdir()
    db=w.connect_db(destination);ids=set();hashes={};statuses=Counter()
    for job in jobs:
        folder=STUDY/'jobs'/job['name']
        assert json.loads((folder/'terminal.json').read_text())['returncode']==0
        w.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
        src=sqlite3.connect((folder/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True);src.row_factory=sqlite3.Row
        chunk=[dict(r) for r in src.execute('SELECT * FROM results')];src.close()
        expected={r['id'] for r in b.read_jsonl(folder/'manifest.jsonl')}
        assert {r['item_id'] for r in chunk}==expected and len(chunk)==job['count'] and not ids.intersection(expected)
        ids.update(expected)
        for r in chunk:w.insert_result(db,[r[c] for c in w.RESULT_COLUMNS]);statuses[r['status']]+=1
        hashes[job['name']]=digest(folder/'results.sqlite')
    db.close();assert ids=={r['id'] for r in rows}
    dump(destination.with_suffix('.seal.json'),dict(n_records=len(ids),status_counts=dict(statuses),
        database_sha256=digest(destination),manifest_sha256=digest(STUDY/'manifests/main.jsonl'),source_databases=hashes))
    dump(STUDY/'STATUS.json',dict(state='completed',started_at=start,finished_at=now(),n_terminal=len(ids),status_counts=dict(statuses)))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(STUDY/'FAILURE.json',dict(time=now(),type=type(e).__name__,message=str(e)));raise
