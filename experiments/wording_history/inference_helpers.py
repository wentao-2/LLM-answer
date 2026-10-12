"""Frozen supplement construction and immutable worker preparation."""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import random
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
import build_extension as base
import run_extension as worker

MODELS = {
    'qwen3-8b-r5': dict(name='qwen3-8b-r5',path='models/Qwen/Qwen3-8B',
        revision='b968826d9c46dd6066d109eabc6255188de91218',batch_size=8,
        chat_template_kwargs=dict(enable_thinking=True),require_think_close=True),
    'phi4-reasoning-r5': dict(name='phi4-reasoning-r5',path='models/microsoft/Phi-4-reasoning',
        revision='1de18ec97600877ce63dbf60c73b998da99f0195',batch_size=2,require_think_close=True),
}


def digest(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    pending.replace(path)


def seal_item(item, model):
    item = copy.deepcopy(item)
    item['origin_item_id'] = item.pop('id', None)
    item['generator_version'] = 'i16-review-supplement-r5'
    item['model_scope'] = [model]
    item['max_new_tokens'] = 28672 if model.startswith('phi') else 12288
    item['max_input_tokens'] = 4096 if model.startswith('phi') else 8192
    item['readout'] = 'probability'
    item['id'] = hashlib.sha256(base.canonical(item).encode()).hexdigest()[:24]
    return item


def write_manifest(path, rows):
    path = Path(path)
    data = ('\n'.join(base.canonical(r) for r in rows)+'\n').encode()
    if path.exists():
        assert path.read_bytes() == data, 'Immutable manifest differs: '+str(path)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    return digest(path)


def prepare_jobs(stage, rows, model, n_shards):
    assert len({r['id'] for r in rows}) == len(rows)
    write_manifest(ROOT/'inference/manifests'/f'{stage}.jsonl',rows)
    selected = list(rows)
    random.Random(16101101).shuffle(selected)
    jobs = []
    for shard in range(n_shards):
        shard_rows = selected[shard::n_shards]
        if not shard_rows:
            continue
        name = f'{stage}-s{shard:02d}'
        folder = ROOT/'inference/jobs'/name
        if folder.exists():
            raise ValueError('Refuse to overwrite a prepared job: '+name)
        (folder/'source').mkdir(parents=True)
        shutil.copy2(ROOT/'source/run_extension.py',folder/'source/run_extension.py')
        shutil.copy2(ROOT/'INFERENCE_PROTOCOL.md',folder/'EXPERIMENT_PROTOCOL.md')
        if (ROOT/'PRE_EXECUTION_AMENDMENT.md').exists():
            shutil.copy2(ROOT/'PRE_EXECUTION_AMENDMENT.md',folder/'PRE_EXECUTION_AMENDMENT.md')
        write_manifest(folder/'manifest.jsonl',shard_rows)
        spec = copy.deepcopy(MODELS[model])
        cfg = dict(name=name,manifest='manifest.jsonl',experiment_protocol='EXPERIMENT_PROTOCOL.md',
            seed=1601,batch_size=spec['batch_size'],max_input_tokens=4096 if model.startswith('phi') else 8192,
            max_new_tokens=28672 if model.startswith('phi') else 12288,dtype='bfloat16',attention='sdpa',
            do_sample=False,num_beams=1,store_token_ids=True,cpu_threads=4,readout='probability',models=[spec])
        atomic_json(folder/'config.json',cfg)
        frozen = dict(config_sha256=digest(folder/'config.json'),manifest_sha256=digest(folder/'manifest.jsonl'),
            source_hashes={'run_extension.py':digest(folder/'source/run_extension.py')},
            protocol_hashes={'EXPERIMENT_PROTOCOL.md':digest(folder/'EXPERIMENT_PROTOCOL.md')})
        if (folder/'PRE_EXECUTION_AMENDMENT.md').exists():
            frozen['protocol_hashes']['PRE_EXECUTION_AMENDMENT.md']=digest(folder/'PRE_EXECUTION_AMENDMENT.md')
        atomic_json(folder/'frozen.json',frozen)
        jobs.append(dict(name=name,stage=stage,model=model,count=len(shard_rows),
                         frozen_sha256=digest(folder/'frozen.json')))
    return jobs


def merge_stage(stage, jobs):
    planned = base.read_jsonl(ROOT/'inference/manifests'/f'{stage}.jsonl')
    destination = ROOT/'inference/results'/f'{stage}.sqlite'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise ValueError('Refuse to overwrite completed stage '+stage)
    pending = destination.with_suffix('.pending.sqlite')
    db = worker.connect_db(pending)
    all_ids = set()
    all_rows = []
    source_hashes = {}
    for job in jobs:
        folder = ROOT/'inference/jobs'/job['name']
        terminal = json.loads((folder/'terminal.json').read_text())
        assert terminal['returncode'] == 0, job['name']
        worker.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
        source_db = sqlite3.connect((folder/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        source_db.row_factory = sqlite3.Row
        rows = [dict(row) for row in source_db.execute('SELECT * FROM results ORDER BY id')]
        source_db.close()
        expected = {job['model']+':'+r['id'] for r in base.read_jsonl(folder/'manifest.jsonl')}
        assert {r['id'] for r in rows} == expected and len(rows) == job['count']
        assert not all_ids.intersection(expected)
        all_ids.update(expected)
        assert all(r['status'] in ('ok','invalid','truncated','error') for r in rows)
        all_rows.extend(rows)
        for row in rows:
            worker.insert_result(db,[row[c] for c in worker.RESULT_COLUMNS])
        source_hashes[job['name']] = digest(folder/'results.sqlite')
    assert len(all_rows) == len(planned)
    db.commit()
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    pending.replace(destination)
    atomic_json(destination.with_suffix('.seal.json'),dict(stage=stage,n_records=len(all_rows),
        model_counts=dict(Counter(r['model'] for r in all_rows)),status_counts=dict(Counter(r['status'] for r in all_rows)),
        manifest_sha256=digest(ROOT/'inference/manifests'/f'{stage}.jsonl'),database_sha256=digest(destination),
        source_databases=source_hashes,scientific_rows_unchanged=True))
    return all_rows,destination


def history_followups(stage, parents, records, model):
    children,cohort = base.history_children(parents,records,[model])
    by_parent = {r['id']:r for r in parents}
    for child in children:
        parent = by_parent[child['parent_item_id']]
        child['parent_condition'] = parent['parent_condition']
        if model.startswith('qwen'):
            child['chat_template_kwargs'] = dict(enable_thinking=True)
            child['require_think_close'] = True
        child.update(native_mode='default')
    children = [seal_item(r,model) for r in children]
    atomic_json(ROOT/'inference/results'/f'{stage}-cohort.json',dict(parents=cohort,
        planned_parents=len(cohort),eligible_parents=sum(r['eligible'] for r in cohort),constructed_children=len(children)))
    return children


def initial_manifests():
    phi = 'phi4-reasoning-r5'
    qwen = 'qwen3-8b-r5'
    gate = [seal_item(r,phi) for r in base.read_jsonl(ROOT/'evidence/R4/gate.jsonl')]
    parents = []
    for row in base.read_jsonl(ROOT/'evidence/R4/history-stage1.jsonl'):
        for condition in ('numeric_on','numeric_off','explanation_on'):
            x = copy.deepcopy(row)
            x['parent_condition'] = condition
            x['chat_template_kwargs'] = dict(enable_thinking=condition!='numeric_off')
            x['require_think_close'] = condition!='numeric_off'
            if condition == 'explanation_on':
                prefix = x['prompt'].split('Your final visible answer')[0]
                x['prompt'] = prefix + ('In your visible answer, first give a concise explanation in two to four sentences for your probability. '
                    + base.FINAL_INSTRUCTION)
            parents.append(seal_item(x,qwen))
    original = [r for r in base.read_jsonl(ROOT/'evidence/R3/manifest.jsonl') if
        r['protocol']=='independent' and r['visibility']=='signal_visible' and r['signal']==r['answer']
        and r['attribution'] in ('self','other')]
    assert len(original)==384
    paraphrases=[]
    for row in original:
        for variant in ('original','authorship','dependence'):
            x=copy.deepcopy(row)
            x.update(module='paraphrase',analysis_modules=['paraphrase'],paraphrase_variant=variant,
                     chat_template_kwargs=dict(enable_thinking=True),require_think_close=True)
            if variant=='authorship':
                old='Attributed author of the earlier answer: '+('you' if x['attribution']=='self' else 'another assistant')+'.'
                new='The earlier answer is attributed to '+('the current assistant (you)' if x['attribution']=='self' else 'a different assistant')+'.'
                assert x['prompt'].count(old)==1
                x['prompt']=x['prompt'].replace(old,new)
            elif variant=='dependence':
                old='Given H, S and T are independent. The earlier answer A is set equal to T.'
                new='Conditional on the hidden state H, observing S leaves the distribution of T unchanged. The value reported as the earlier answer A is exactly the second reading T.'
                assert x['prompt'].count(old)==1
                x['prompt']=x['prompt'].replace(old,new)
            sealed=seal_item(x,qwen)
            assert abs(base.oracle(sealed['protocol'],sealed['visibility'],sealed['prior'],sealed['reliability'],sealed['signal'],sealed['answer'])-sealed['oracle'])<1e-12
            paraphrases.append(sealed)
    assert len(gate)==96 and len(parents)==144 and len(paraphrases)==1152
    write_manifest(ROOT/'inference/manifests/paraphrase.jsonl',paraphrases)
    jobs=prepare_jobs('phi-gate',gate,phi,4)+prepare_jobs('qwen-parents',parents,qwen,4)
    atomic_json(ROOT/'inference/INITIAL_JOBS.json',jobs)
    atomic_json(ROOT/'inference/DESIGN_SEAL.json',dict(protocol_sha256=digest(ROOT/'INFERENCE_PROTOCOL.md'),
        helper_sha256=digest(Path(__file__)),controller_sha256=digest(ROOT/'orchestrate_inference.py'),
        manifests={p.name:digest(p) for p in (ROOT/'inference/manifests').glob('*.jsonl')},
        scientific_inference_started=False))
    return jobs


if __name__=='__main__':
    print(json.dumps(dict(jobs=initial_manifests())))
