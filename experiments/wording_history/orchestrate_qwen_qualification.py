"""Independent registered diagnostic after the complete Qwen R6 comparison."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import copy,hashlib,json,os,random,shutil,sqlite3,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import build_extension as b
import run_extension as w
import analyze_extension as a
STUDY=ROOT/'qwen-r6-qualification';MODEL='qwen3-8b-r6'
PYTHON=str(ROOT/'fast_runtime/vllm085/venv/bin/python')
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,v):w.atomic_json(p,v)
def main():
    assert not STUDY.exists(),'No blind restart'
    STUDY.mkdir()
    dump(STUDY/'REGISTERED.json',dict(protocol_sha256=w.digest(ROOT/'VLLM_QUALIFICATION_ADDENDUM.md'),
        controller_sha256=w.digest(Path(__file__)),registered_at=now(),scientific_inference_started=False))
    while True:
        mainroot=ROOT/'vllm-r6'
        if (mainroot/'FAILURE.json').exists():raise RuntimeError('Comparison runtime has a technical failure')
        if (mainroot/'STATUS.json').exists() and json.loads((mainroot/'STATUS.json').read_text())['state']=='completed':break
        time.sleep(30)
    rows=[]
    for old in b.read_jsonl(ROOT/'evidence/R4/gate.jsonl'):
        x=copy.deepcopy(old);x['origin_item_id']=x.pop('id')
        x.update(model_scope=[MODEL],generator_version='review-r6-qualification',backend='vllm-0.8.5',
            max_new_tokens=12288,max_input_tokens=8192,chat_template_kwargs=dict(enable_thinking=True),require_think_close=True)
        x['id']=hashlib.sha256(b.canonical(x).encode()).hexdigest()[:24];rows.append(x)
    assert len(rows)==96
    b.write_manifest(STUDY/'manifest.jsonl',rows)
    random.Random(16101101).shuffle(rows);jobs=[]
    for shard,gpu in enumerate(range(4,8)):
        folder=STUDY/'jobs'/f'gate-s{shard:02d}';(folder/'source').mkdir(parents=True)
        source=ROOT/'vllm-r6/jobs/main-s00/source'
        for filename in ('run_extension.py','native_worker.py'):shutil.copy2(source/filename,folder/'source'/filename)
        shutil.copy2(ROOT/'VLLM_QUALIFICATION_ADDENDUM.md',folder/'PROTOCOL.md')
        b.write_manifest(folder/'manifest.jsonl',rows[shard::4])
        config=json.loads((ROOT/'vllm-r6/jobs/main-s00/config.json').read_text())
        config.update(name=folder.name,experiment_protocol='PROTOCOL.md')
        dump(folder/'config.json',config)
        frozen=dict(config_sha256=w.digest(folder/'config.json'),manifest_sha256=w.digest(folder/'manifest.jsonl'),
            source_hashes={p.name:w.digest(p) for p in (folder/'source').glob('*.py')},protocol_hashes={'PROTOCOL.md':w.digest(folder/'PROTOCOL.md')})
        dump(folder/'frozen.json',frozen);w.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
        jobs.append(dict(folder=str(folder),gpu=gpu,count=24,frozen_sha256=w.digest(folder/'frozen.json')))
    dump(STUDY/'DESIGN_SEAL.json',dict(jobs=jobs,manifest_sha256=w.digest(STUDY/'manifest.jsonl'),
        main_design_sha256=w.digest(ROOT/'vllm-r6/DESIGN_SEAL.json'),scientific_inference_started=False))
    def execute(job):
        folder=Path(job['folder']);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='4')
        with (folder/'worker.log').open('wb') as stream:
            child=subprocess.Popen([PYTHON,str(folder/'source/run_extension.py'),'--config',str(folder/'config.json'),'--model',MODEL],
                env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            dump(folder/'launch.json',dict(pid=child.pid,job=job,started_at=now()));rc=child.wait()
        dump(folder/'terminal.json',dict(returncode=rc,finished_at=now(),job=job))
        if rc:raise RuntimeError('Qualification worker failed '+folder.name)
    dump(STUDY/'STATUS.json',dict(state='running',jobs=jobs,started_at=now()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        for f in futures:f.result()
    destination=STUDY/'results.sqlite';db=w.connect_db(destination);ids=set();counts=Counter();hashes={}
    for job in jobs:
        folder=Path(job['folder']);src=sqlite3.connect((folder/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True);src.row_factory=sqlite3.Row
        chunk=[dict(r) for r in src.execute('SELECT * FROM results')];src.close()
        expected={r['id'] for r in b.read_jsonl(folder/'manifest.jsonl')}
        assert {r['item_id'] for r in chunk}==expected and len(chunk)==24 and not ids.intersection(expected)
        ids.update(expected)
        for r in chunk:w.insert_result(db,[r[c] for c in w.RESULT_COLUMNS]);counts[r['status']]+=1
        hashes[folder.name]=w.digest(folder/'results.sqlite')
    db.close();assert len(ids)==96
    report=a.analyze_gate(STUDY/'manifest.jsonl',destination,MODEL);dump(STUDY/'analysis.json',report)
    dump(STUDY/'RESULT_SEAL.json',dict(database_sha256=w.digest(destination),manifest_sha256=w.digest(STUDY/'manifest.jsonl'),source_databases=hashes))
    dump(STUDY/'STATUS.json',dict(state='completed',passes=report['passes'],status_counts=dict(counts),finished_at=now()))
if __name__=='__main__':
    try:main()
    except BaseException as e:dump(STUDY/'FAILURE.json',dict(time=now(),type=type(e).__name__,message=str(e)));raise
