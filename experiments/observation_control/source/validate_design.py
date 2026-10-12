"""Independent mathematical and construction checks, without model inference."""
from collections import Counter
from fractions import Fraction
import itertools
import json
import math
import sys

from common import ROOT, PHI, QWEN, canonical, load, read_jsonl, text_hash, validate_seal, write_json
import build_design as b


def exact(item):
    pi,r=Fraction(str(item['prior'])),Fraction(str(item['reliability']))
    s,t=item['signal'],item.get('next_signal')
    odds=pi/(1-pi)*(r/(1-r) if s else (1-r)/r)
    if t is not None:
        rho=Fraction(str(item['rho']))
        odds *= (rho*int(t==s)+(1-rho)*(r if t else 1-r)) / \
                (rho*int(t==s)+(1-rho)*(1-r if t else r))
    return odds/(1+odds)


def verify_item(item):
    content=dict(item)
    iid=content.pop('id')
    assert text_hash(canonical(content))[:24]==iid
    q=exact(item)
    assert math.isclose(float(q),item['oracle'],abs_tol=1e-12,rel_tol=0)
    assert str(q)==item['oracle_fraction']
    assert item['max_input_tokens']==8192
    assert item['max_new_tokens']==(8192 if item['model_scope']==[PHI] else 12288)
    if item.get('rho')==1:
        assert item['branch']!='conflict'
    if 'messages' in item:
        assert [x['role'] for x in item['messages']]==['user','assistant','user']
        assert text_hash(item['messages'][1]['content'])==item['parent_final_sha256']
        task=item['messages'][-1]['content']
        assert task.count('FINAL_JSON:')==1
        assert 'likelihood_h0' not in task or item['condition']=='full_steps'
        if item['branch']=='none':
            assert 'Observed new record: B=' not in task and 'Observed second report: B=' not in task
            assert 'Observed quotation: C=' not in task
        elif item['context_id']=='report':
            assert f'Observed quotation: C={item["next_signal"]}' in task
        if item['condition']=='explicit':
            assert ('P(H=1|S)' if item['next_signal'] is None else
                    ('P(H=1|S,B,C)' if item['context_id']=='report' else 'P(H=1|S,B)')) in task


def validate(save=True):
    if (ROOT/'DESIGN_SEAL.json').exists():
        validate_seal()
    items=read_jsonl(ROOT/'manifests/preflight_static.jsonl')
    for item in items:
        verify_item(item)
    assert Counter(x['stage'] for x in items)=={'expression':768,'transfer_parent':64}
    expressions=[x for x in items if x['stage']=='expression']
    assert len({x['parent_item_id'] for x in expressions})==24
    parents=[x for x in items if x['stage']=='transfer_parent']
    generated=[]
    # Mock text is used only to test the frozen constructor. It is never saved
    # as scientific parent output and never reaches a model execution queue.
    for parent in parents:
        final='FINAL_JSON: {"p_h1": 0.5}'
        rec=dict(model=parent['model_scope'][0],probability=.5,status='ok',
                 metadata_json=json.dumps(dict(final_output=final)))
        e=dict(parent=parent,record=rec,continuation_model=rec['model'],
               parent_record_sha256=text_hash(canonical(rec)),parent_final_sha256=text_hash(final))
        children=[b.child(e,'transfer',rho,branch,t,'selector',condition)
                  for rho,branch,t in b.legal_branches(parent['signal']) for condition in b.CONDITIONS]
        assert len(children)==len({x['id'] for x in children})==32
        for item in children:
            verify_item(item)
        generated += children
    assert len(generated)==len({x['id'] for x in generated})==2048
    # Explicitly enumerate the natural-language process using copy/noncopy paths
    # rather than consulting the execution oracle.
    natural_cases=0
    for pi,r,s in itertools.product(b.PRIORS,(.66,.78,.92)+b.NEW_RELIABILITIES,(0,1)):
        for rho,branch,t in b.legal_branches(s):
            weights=[]
            for h in (0,1):
                p_s=Fraction(str(r)) if s==h else 1-Fraction(str(r))
                second=Fraction(1)
                if t is not None:
                    second=Fraction(str(rho))*int(s==t)
                    second+=(1-Fraction(str(rho)))*(Fraction(str(r)) if t==h else 1-Fraction(str(r)))
                weights.append((Fraction(str(pi)) if h else 1-Fraction(str(pi)))*p_s*second)
            q=weights[1]/sum(weights)
            assert q==exact(dict(prior=pi,reliability=r,signal=s,next_signal=t,rho=rho))
            natural_cases+=1
    sys.path.insert(0,str(ROOT/'worker_sources/scheduler'))
    import schedule_runs as scheduler
    for label in ('parents_phi','parents_qwen'):
        scheduler.validate_queue(ROOT/'queues'/label)
    result=dict(state='passed',static_calls=len(items),expression_calls=768,parent_calls=64,
                hypothetical_constructor_cases=len(generated),natural_process_cases=natural_cases,
                legal_branches=8,children_per_valid_parent=32,unique_ids=True,
                no_impossible_conflict=True,none_no_observation_leakage=True,
                old_parent_answer_bytes_retained=True,scientific_inference=False)
    if save:
        write_json(ROOT/'preflight/LOCAL_DESIGN_CHECK.json',result)
    print(canonical(result))
    return result


if __name__=='__main__':
    validate()
