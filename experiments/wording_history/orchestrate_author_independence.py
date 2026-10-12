"""Late-registered separate declaration paraphrase; all outcomes retained."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import copy,hashlib,json,os,random,shutil,sqlite3,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'source'))
import build_extension as b
import run_extension as w
MODEL='qwen3-8b-r6';STUDY=ROOT/'author-independence-r6'
PYTHON=str(ROOT/'fast_runtime/vllm085/venv/bin/python')
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,v):w.atomic_json(p,v)
def main():
    assert not STUDY.exists();STUDY.mkdir()
    dump(STUDY/'REGISTERED.json',dict(protocol_sha256=w.digest(ROOT/'AUTHOR_INDEPENDENCE_ADDENDUM.md'),
        controller_sha256=w.digest(Path(__file__)),analysis_sha256=w.digest(ROOT/'analyze_four_wordings.py'),registered_at=now()))
    source=b.read_jsonl(ROOT/'inference/manifests/paraphrase.jsonl')+b.read_jsonl(ROOT/'inference/manifests/paraphrase-neutral.jsonl')
    rows=[]
    old='Authorship in this hypothetical record is assigned independently of H, all sensor values, and the coin.'
    new='The author label was assigned without using H, any sensor reading, or the coin, and is statistically independent of all of them.'
    for item in source:
        if item['paraphrase_variant']!='original':continue
        x=copy.deepcopy(item);x['origin_item_id']=x.pop('id');assert x['prompt'].count(old)==1
        x['prompt']=x['prompt'].replace(old,new)
        x.update(model_scope=[MODEL],generator_version='review-r6-author-independence',backend='vllm-0.8.5',paraphrase_variant='author_independence')
        x['id']=hashlib.sha256(b.canonical(x).encode()).hexdigest()[:24];rows.append(x)
    assert len(rows)==576
    b.write_manifest(STUDY/'manifest.jsonl',rows)
    while True:
        previous=ROOT/'qwen-r6-qualification'
        if (previous/'FAILURE.json').exists():raise RuntimeError('Qualification has a technical failure')
        if (previous/'STATUS.json').exists() and json.loads((previous/'STATUS.json').read_text())['state']=='completed':break
        time.sleep(30)
    random.Random(16101101).shuffle(rows);jobs=[]
    for shard,gpu in enumerate(range(4,8)):
        folder=STUDY/'jobs'/f'variant-s{shard:02d}';(folder/'source').mkdir(parents=True)
        source=ROOT/'vllm-r6/jobs/main-s00/source'
        for name in ('run_extension.py','native_worker.py'):shutil.copy2(source/name,folder/'source'/name)
        shutil.copy2(ROOT/'AUTHOR_INDEPENDENCE_ADDENDUM.md',folder/'PROTOCOL.md')
        b.write_manifest(folder/'manifest.jsonl',rows[shard::4])
        config=json.loads((ROOT/'vllm-r6/jobs/main-s00/config.json').read_text());config.update(name=folder.name,experiment_protocol='PROTOCOL.md')
        dump(folder/'config.json',config)
        frozen=dict(config_sha256=w.digest(folder/'config.json'),manifest_sha256=w.digest(folder/'manifest.jsonl'),
            source_hashes={p.name:w.digest(p) for p in (folder/'source').glob('*.py')},protocol_hashes={'PROTOCOL.md':w.digest(folder/'PROTOCOL.md')})
        dump(folder/'frozen.json',frozen);w.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
        jobs.append(dict(folder=str(folder),gpu=gpu,count=144,frozen_sha256=w.digest(folder/'frozen.json')))
    dump(STUDY/'DESIGN_SEAL.json',dict(jobs=jobs,manifest_sha256=w.digest(STUDY/'manifest.jsonl'),
        analysis_sha256=w.digest(ROOT/'analyze_four_wordings.py'),scientific_inference_started=False))
    def execute(job):
        folder=Path(job['folder']);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='4')
        with (folder/'worker.log').open('wb') as stream:
            child=subprocess.Popen([PYTHON,str(folder/'source/run_extension.py'),'--config',str(folder/'config.json'),'--model',MODEL],
                env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            dump(folder/'launch.json',dict(pid=child.pid,job=job,started_at=now()));rc=child.wait()
        dump(folder/'terminal.json',dict(returncode=rc,finished_at=now(),job=job))
        if rc:raise RuntimeError('Paraphrase worker failed '+folder.name)
    dump(STUDY/'STATUS.json',dict(state='running',jobs=jobs,started_at=now()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        for f in futures:f.result()
    dest=STUDY/'results.sqlite';db=w.connect_db(dest);ids=set();counts=Counter();hashes={}
    for job in jobs:
        folder=Path(job['folder']);w.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
        src=sqlite3.connect((folder/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True);src.row_factory=sqlite3.Row
        chunk=[dict(r) for r in src.execute('SELECT * FROM results')];src.close()
        expected={r['id'] for r in b.read_jsonl(folder/'manifest.jsonl')}
        assert len(chunk)==144 and {r['item_id'] for r in chunk}==expected and not ids.intersection(expected)
        ids.update(expected)
        for r in chunk:w.insert_result(db,[r[c] for c in w.RESULT_COLUMNS]);counts[r['status']]+=1
        hashes[folder.name]=w.digest(folder/'results.sqlite')
    db.close();assert len(ids)==576
    dump(STUDY/'RESULT_SEAL.json',dict(database_sha256=w.digest(dest),manifest_sha256=w.digest(STUDY/'manifest.jsonl'),source_databases=hashes))
    dump(STUDY/'STATUS.json',dict(state='completed',status_counts=dict(counts),n_terminal=576,finished_at=now()))
if __name__=='__main__':
    try:main()
    except BaseException as e:dump(STUDY/'FAILURE.json',dict(time=now(),type=type(e).__name__,message=str(e)));raise
