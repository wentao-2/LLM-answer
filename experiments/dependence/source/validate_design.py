"""Validate the newly connected prompt/oracle paths without model generation."""
from collections import Counter, defaultdict
import math
import sys

from common import ROOT, PHI, QWEN, canonical, load, read_jsonl, sha, text_hash, validate_seal
from oracle import target
from build_design import ARMS, RHOS, REMINDERS


def validate():
    seal = validate_seal()
    items = read_jsonl(ROOT/'manifests/all.jsonl')
    assert len(items) == len({x['id'] for x in items}) == 2112
    partial = [x for x in items if x['module']=='partial_dependence']
    intervention = [x for x in items if x['module']=='source_intervention']
    assert len(partial)==1344 and len(intervention)==768
    parents = {(x['continuation_model'],x['parent']['id']):x for x in read_jsonl(ROOT/'inputs/parents.jsonl')}
    controls = {x['source_item']['id']:x for x in read_jsonl(ROOT/'inputs/original_controls.jsonl')}
    selected = defaultdict(list)
    for item in items:
        data = dict(item)
        assert data.pop('id') == text_hash(canonical(data))[:24]
        assert str(target(item)) == item['oracle_fraction']
        assert math.isclose(float(target(item)),item['oracle'],rel_tol=0,abs_tol=1e-12)
        model = item['model_scope'][0]
        p = parents[(model,item['parent_item_id'])]
        assert item['parent_record_sha256']==text_hash(canonical(p['record']))
        assert item['parent_final_sha256']==p['parent_final_sha256']
        assert item['history_arm'] in ARMS
        if item['module']=='partial_dependence':
            assert item['template_id']=='frequency_v3'
            assert item['rho'] in RHOS
            task = item['messages'][-1]['content'] if 'messages' in item else item['prompt'].split('[NEW TASK]\n',1)[1]
            if item['branch']=='none':
                assert item['next_signal'] is None and 'Observed new record:' not in task
            else:
                assert f'Observed new record: B={item["next_signal"]}.' in task
            assert not (item['rho']==1 and item['branch']=='conflict')
            selected[(model,item['parent_item_id'])].append(item)
        else:
            source = controls[item['control_item_id']]['source_item']
            extra = REMINDERS[item['condition']]+'\n'
            if 'messages' in item:
                assert item['messages'][:2]==source['messages'][:2]
                assert item['messages'][-1]['content'].count(extra)==1
                assert item['messages'][-1]['content'].replace(extra,'')==source['messages'][-1]['content']
            else:
                assert item['prompt'].count(extra)==1
                assert item['prompt'].replace(extra,'')==source['prompt']
    assert len(selected)==48 and all(len(v)==28 for v in selected.values())
    for model in (PHI,QWEN):
        chosen = [p['parent'] for (m,_),p in parents.items() if m==model and p['selected_for_partial']]
        assert len(chosen)==24
        assert sorted(Counter(x['family_id'] for x in chosen).values())==[4]*6
        assert {(x['prior'],x['reliability'],x['signal']) for x in chosen}=={
            (p,r,s) for p in (.18,.32,.68,.82) for r in (.66,.78,.92) for s in (0,1)}
    # Invoke the unchanged scheduler's complete queue validation on both backends.
    for label in ('phi','qwen'):
        path = str(ROOT/'queues'/label/'source')
        sys.path.insert(0,path)
        import schedule_runs
        root,queue,_ = schedule_runs.validate_queue(ROOT/'queues'/label)
        assert queue['expected_records']==(1440 if label=='phi' else 672)
        sys.path.pop(0)
    assert sum(j['count'] for label in ('phi','qwen') for j in load(ROOT/'queues'/label/'queue.json')['jobs'])==2112
    print(canonical(dict(state='passed',planned_calls=2112,original_controls=len(controls),
        partial_parents=len(selected),exact_oracle_cases=load(ROOT/'design/BUILD_CHECKS.json')['exact_oracle_cases'],
        design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'))))


if __name__=='__main__':
    validate()
