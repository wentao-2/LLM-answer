"""Frozen complete-phase analysis, retaining planned and constructed denominators."""
from collections import Counter,defaultdict
import csv
from fractions import Fraction
import json
import math
import re
from statistics import mean

from common import (ROOT,PHI,QWEN,canonical,load,now,read_db,read_jsonl,sha,validate_seal,write_json)
from oracle import posterior,sensor,logit,normative_weight
from build_design import CONDITIONS,legal_branches


def valid(rec):
    return (rec is not None and rec['status']=='ok' and type(rec['probability']) in (float,int)
            and math.isfinite(rec['probability']) and 0<=rec['probability']<=1)


def load_observations():
    validate_seal()
    items=read_jsonl(ROOT/'manifests/expression.jsonl')+read_jsonl(ROOT/'manifests/transfer_parents.jsonl')
    records={}
    cohorts={}
    for label,model in (('phi',PHI),('qwen',QWEN)):
        seal=load(ROOT/'cohorts'/label/'CONSTRUCTION.json')
        assert seal['design_seal_sha256']==sha(ROOT/'DESIGN_SEAL.json')
        for rel,digest in seal['files'].items():
            assert sha(ROOT/rel)==digest
        children=read_jsonl(ROOT/'cohorts'/label/'children.jsonl')
        items+=children
        cohorts[label]=read_jsonl(ROOT/'cohorts'/label/'coverage.jsonl')
        for queue in ('parents_'+label,'followups_'+label):
            qroot=ROOT/'queues'/queue
            if not qroot.exists():
                assert queue.startswith('followups_') and seal['allocation'] is None
                continue
            terminal=load(qroot/'scheduler.terminal.json')
            assert terminal['returncode']==0
            chunk=read_db(qroot/'merged.sqlite')
            for rec in chunk:
                key=(model,rec['item_id'])
                assert key not in records and rec['model']==model
                records[key]=rec
    assert set(records)=={(x['model_scope'][0],x['id']) for x in items}
    return items,records,cohorts


def fields(item):
    return dict(item_id=item['id'],model=item['model_scope'][0],stage=item['stage'],
        parent_item_id=item.get('parent_item_id'),family_id=item['family_id'],context=item['context_id'],
        prior=item['prior'],reliability=item['reliability'],signal=item['signal'],
        rule=item.get('rule_expression'),condition=item.get('condition'),rho=item.get('rho'),branch=item['branch'])


def error_row(item,rec):
    q=float(posterior(item['prior'],item['reliability'],item['signal'],item['next_signal'],item.get('rho',0)))
    p=rec['probability'] if valid(rec) else None
    err=abs(p-q) if p is not None else None
    x=fields(item)
    x.update(status=rec['status'] if rec else 'unconstructed_missing_parent',probability=p,oracle=q,
        posterior_error=err,error_lower=err if p is not None else 0,
        error_upper=err if p is not None else max(q,1-q),endpoint=p in (0,1) if p is not None else False,
        output_tokens=rec['output_tokens'] if rec else None,duration_sec=rec['duration_sec'] if rec else None)
    return x


def pair_row(item,rec,none,baseline):
    x=error_row(item,rec)
    base=error_row(none,baseline)
    pb,p0=x['probability'],base['probability']
    dq=x['oracle']-base['oracle']
    update=pb-p0 if pb is not None and p0 is not None else None
    residual=update-dq if update is not None else None
    lo=(pb if pb is not None else 0)-(p0 if p0 is not None else 1)-dq
    hi=(pb if pb is not None else 1)-(p0 if p0 is not None else 0)-dq
    norm=normative_weight(item['reliability'],item['rho'],item['branch'])
    weight=((2*item['next_signal']-1)*(logit(pb)-logit(p0))/math.log(item['reliability']/(1-item['reliability']))
            if pb is not None and p0 is not None and 0<pb<1 and 0<p0<1 else None)
    x.update(none_item_id=none['id'],none_probability=p0,none_oracle=base['oracle'],
        none_status=base['status'],probability_update=update,oracle_update=dq,update_residual=residual,
        absolute_update_residual=abs(residual) if residual is not None else None,
        update_lower=lo,update_upper=hi,absolute_update_lower=0 if lo<=0<=hi else min(abs(lo),abs(hi)),
        absolute_update_upper=max(abs(lo),abs(hi)),normative_weight=norm,effective_weight=weight,
        weight_residual=weight-norm if weight is not None else None,
        absolute_weight_residual=abs(weight-norm) if weight is not None else None,
        logit_state='finite' if weight is not None else ('endpoint' if update is not None else 'missing_pair'))
    return x


def grouped(rows,keys):
    groups=defaultdict(list)
    for x in rows:
        groups[tuple(x.get(k) for k in keys)].append(x)
    return sorted(groups.items(),key=lambda x:canonical(x[0]))


def metric(rows,key,expected_families=None):
    groups=defaultdict(list)
    for x in rows:
        if x.get(key) is not None:
            groups[x['family_id']].append(x[key])
    values={k:mean(v) for k,v in sorted(groups.items())}
    expected=set(expected_families if expected_families is not None else (x['family_id'] for x in rows))
    return dict(n=sum(len(x) for x in groups.values()),n_families=len(values),expected_families=len(expected),
        family_values=values,equal_family_mean=mean(values.values()) if values and set(values)==expected else None,
        family_range=[min(values.values()),max(values.values())] if values else None)


def summaries(rows,keys,metrics):
    return [dict(zip(keys,k),n_planned=len(group),status_counts=dict(Counter(x['status'] for x in group)),
        logit_states=dict(Counter(x['logit_state'] for x in group if 'logit_state' in x)),
        metrics={m:metric(group,m) for m in metrics},
        common_all={m:metric([x for x in group if x.get('common_all')],m,{x['family_id'] for x in group}) for m in metrics})
        for k,group in grouped(rows,keys)]


def unique_object(pairs):
    d={}
    for k,v in pairs:
        if k in d:
            raise ValueError('Duplicate key')
        d[k]=v
    return d


def work_row(item,rec):
    x=fields(item)
    text=json.loads(rec['metadata_json']).get('final_output','')
    matches=re.findall(r'(?:^|\n)[ \t]*WORK_JSON:[ \t]*(\{[^\r\n]*\})[ \t]*(?=\n|$)',text)
    work=None
    if len(matches)==1:
        try:
            work=json.loads(matches[0],object_pairs_hook=unique_object)
        except (ValueError,TypeError):
            pass
    expected_keys={'observed','sources','not_revealed_source_variables','likelihood_h0','likelihood_h1'}
    schema=(isinstance(work,dict) and set(work)==expected_keys and isinstance(work['observed'],dict)
            and isinstance(work['sources'],dict) and isinstance(work['not_revealed_source_variables'],list)
            and all(isinstance(v,str) for v in work['not_revealed_source_variables'])
            and all(type(work[k]) in (int,float) and math.isfinite(work[k]) and 0<=work[k]<=1
                    for k in ('likelihood_h0','likelihood_h1')))
    obs={'S':item['signal']}
    if item['next_signal'] is not None:
        obs['B']=item['next_signal']
        if item['context_id']=='report':
            obs['C']=item['next_signal']
    ll=[]
    for h in (0,1):
        v=sensor(item['signal'],h,item['reliability'])
        if item['next_signal'] is not None:
            rho=Fraction(str(item['rho']))
            v*=rho*int(item['next_signal']==item['signal'])+(1-rho)*sensor(item['next_signal'],h,item['reliability'])
        ll.append(float(v))
    x.update(final_status=rec['status'],work_schema_valid=bool(schema),work=work,
        expected_observed=obs,observed_correct=(work['observed']==obs) if schema else None,
        not_revealed_names_correct=(sorted(work['not_revealed_source_variables'])==['D','T']) if schema else None,
        expected_likelihood_h0=ll[0],expected_likelihood_h1=ll[1],
        likelihood_h0_error=abs(work['likelihood_h0']-ll[0]) if schema else None,
        likelihood_h1_error=abs(work['likelihood_h1']-ll[1]) if schema else None,
        manual_source_code=None,manual_visible_quote=None,manual_coder=None)
    return x


def contrast(a,b,kind):
    x={k:a[k] for k in ('model','stage','parent_item_id','family_id','context','prior','reliability','signal','rule','rho','branch')}
    x.update(comparison=kind,condition=a['condition'],reference_condition=b['condition'],
             item_id=a['item_id'],reference_item_id=b['item_id'],common_all=a.get('common_all') and b.get('common_all'))
    for key in ('posterior_error','absolute_update_residual','effective_weight','absolute_weight_residual'):
        va,vb=a.get(key),b.get(key)
        x[key+'_change']=va-vb if va is not None and vb is not None else None
    x.update(error_change_lower=a['error_lower']-b['error_upper'],error_change_upper=a['error_upper']-b['error_lower'])
    if 'absolute_update_lower' in a:
        x.update(absolute_update_change_lower=a['absolute_update_lower']-b['absolute_update_upper'],
                 absolute_update_change_upper=a['absolute_update_upper']-b['absolute_update_lower'])
    return x


def write_csv(name,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for x in rows for k in x))
    with (ROOT/'analysis'/name).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for x in rows:
            writer.writerow({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in x.items()})


def analyze():
    items,records,cohorts=load_observations()
    errors=[error_row(x,records[(x['model_scope'][0],x['id'])]) for x in items]
    children=[x for x in items if x['stage']!='transfer_parent']
    key=lambda x:(x['model_scope'][0],x['stage'],x['parent_item_id'],x['rule_expression'],x['condition'],x['rho'])
    baselines={key(x):x for x in children if x['branch']=='none'}
    pairs=[pair_row(x,records[(x['model_scope'][0],x['id'])],baselines[key(x)],
                   records[(x['model_scope'][0],baselines[key(x)]['id'])])
           for x in children if x['branch']!='none']
    parent_sets=defaultdict(list)
    for x in errors:
        if x['stage']!='transfer_parent':
            parent_sets[(x['model'],x['stage'],x['parent_item_id'])].append(x)
    complete={k for k,v in parent_sets.items() if len(v)==32 and all(x['posterior_error'] is not None for x in v)}
    assert all(len(v)==32 for v in parent_sets.values())
    for x in errors+pairs:
        x['common_all']=(x['model'],x['stage'],x['parent_item_id']) in complete
    # Contrasts include none posterior comparisons as well as paired updates.
    all_rows={x['item_id']:x for x in errors}
    all_rows.update({x['item_id']:x for x in pairs})
    index={(x['model'],x['stage'],x['parent_item_id'],x['rule'],x['condition'],x['rho'],x['branch']):x
           for x in all_rows.values() if x['stage']!='transfer_parent'}
    contrasts=[]
    interactions=[]
    for k,a in index.items():
        m,stage,pid,rule,condition,rho,branch=k
        if stage=='expression':
            if condition=='explicit':
                contrasts.append(contrast(a,index[(m,stage,pid,rule,'general',rho,branch)],'explicit_minus_general'))
            if rule=='selector':
                contrasts.append(contrast(a,index[(m,stage,pid,'direct',condition,rho,branch)],'selector_minus_direct'))
            if rule=='selector' and condition=='explicit':
                rows=[a,index[(m,stage,pid,rule,'general',rho,branch)],
                      index[(m,stage,pid,'direct','explicit',rho,branch)],index[(m,stage,pid,'direct','general',rho,branch)]]
                x={k:a[k] for k in ('model','stage','parent_item_id','family_id','context','prior','reliability','signal','rho','branch')}
                for metric_name in ('posterior_error','absolute_update_residual','effective_weight'):
                    vals=[v.get(metric_name) for v in rows]
                    x[metric_name+'_interaction']=vals[0]-vals[1]-vals[2]+vals[3] if all(v is not None for v in vals) else None
                interactions.append(x)
        else:
            if condition!='ordinary':
                contrasts.append(contrast(a,index[(m,stage,pid,rule,'ordinary',rho,branch)],'intervention_minus_ordinary'))
            if condition=='full_steps':
                for ref in ('general_bayes','explicit'):
                    contrasts.append(contrast(a,index[(m,stage,pid,rule,ref,rho,branch)],'full_steps_minus_'+ref))
    references=[]
    for item in children:
        if item['next_signal'] is None:
            continue
        rec=records[(item['model_scope'][0],item['id'])]
        p=rec['probability'] if valid(rec) else None
        x=fields(item)
        targets=dict(joint=item['oracle'],s_only=float(posterior(item['prior'],item['reliability'],item['signal'])),
                     b_only=float(posterior(item['prior'],item['reliability'],item['next_signal'])),
                     independent=float(posterior(item['prior'],item['reliability'],item['signal'],item['next_signal'],0)))
        x.update(status=rec['status'],probability=p,targets=targets,
                 errors={k:abs(p-v) if p is not None else None for k,v in targets.items()},
                 matches_0001=[k for k,v in targets.items() if p is not None and abs(p-v)<=.0001],
                 matches_0005=[k for k,v in targets.items() if p is not None and abs(p-v)<=.0005])
        references.append(x)
    works=[work_row(x,records[(x['model_scope'][0],x['id'])]) for x in children if x['condition']=='full_steps']
    # Intended-child rows for unavailable parents carry known design values but
    # deliberately have no fabricated result ID or conversation history.
    intended=[dict(x,cohort='constructed') for x in errors if x['stage']=='transfer']
    intended_pairs=[dict(x,cohort='constructed') for x in pairs if x['stage']=='transfer']
    parent_items={x['id']:x for x in items if x['stage']=='transfer_parent'}
    for coverage in cohorts.values():
        for c in coverage:
            if c['eligible']:
                continue
            p=parent_items[c['parent_item_id']]
            virtual=[]
            for rho,branch,b in legal_branches(p['signal']):
                for condition in CONDITIONS:
                    spec=dict(p,id=None,stage='transfer',parent_item_id=p['id'],condition=condition,
                              rule_expression='selector',rho=rho,branch=branch,next_signal=b)
                    row=error_row(spec,None)
                    row.update(common_all=False,cohort='unconstructed')
                    intended.append(row)
                    virtual.append(spec)
            virtual_none={(x['rho'],x['condition']):x for x in virtual if x['branch']=='none'}
            for spec in virtual:
                if spec['branch']!='none':
                    row=pair_row(spec,None,virtual_none[(spec['rho'],spec['condition'])],None)
                    row.update(common_all=False,cohort='unconstructed')
                    intended_pairs.append(row)
    assert len(intended)==2048
    assert len(intended_pairs)==1280
    keys=('model','stage','context','rule','condition','rho','branch')
    em=('posterior_error','error_lower','error_upper','output_tokens','duration_sec')
    pm=('update_residual','absolute_update_residual','effective_weight','absolute_weight_residual',
        'update_lower','update_upper','absolute_update_lower','absolute_update_upper')
    ckeys=('model','stage','context','rule','comparison','condition','reference_condition','rho','branch')
    cm=('posterior_error_change','absolute_update_residual_change','effective_weight_change',
        'absolute_weight_residual_change','error_change_lower','error_change_upper',
        'absolute_update_change_lower','absolute_update_change_upper')
    result=dict(analyzed_at=now(),design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'),
        constructed_calls=len(items),max_calls=2880,planned_parents=64,intended_children=2048,
        stage1_calls=768,constructed_stage2_children=sum(x['stage']=='transfer' for x in items),
        cohorts=cohorts,status_counts={m:dict(Counter(r['status'] for (model,_),r in records.items() if model==m)) for m in (PHI,QWEN)},
        error_rows=errors,pair_rows=pairs,contrast_rows=contrasts,interaction_rows=interactions,
        summaries=summaries(errors,keys,em),pair_summaries=summaries(pairs,keys,pm),
        direction_summaries=summaries(errors,keys+('signal',),em),direction_pair_summaries=summaries(pairs,keys+('signal',),pm),
        numeric_summaries=summaries(errors,keys+('prior','reliability','signal'),em),
        contrast_summaries=[dict(zip(ckeys,k),n=len(v),metrics={q:metric(v,q) for q in cm}) for k,v in grouped(contrasts,ckeys)],
        intended_child_summaries=summaries(intended,keys,em),
        intended_pair_summaries=summaries(intended_pairs,keys,pm),
        complete_parent_ids=[dict(model=k[0],stage=k[1],parent_item_id=k[2]) for k in sorted(complete)],
        reference_rows=references,work_rows=works,manual_source_coding='pending, no automatic label inferred',
        inference_scope='fixed synthetic grid, descriptive equal-family means and ranges; no hidden mechanism identification',
        endpoint_policy='count without clipping; logit has no finite universal missing bounds')
    write_json(ROOT/'analysis/RESULTS.json',result)
    for filename,rows in (('posterior_errors.csv',errors),('paired_updates.csv',pairs),
                          ('condition_contrasts.csv',contrasts),('expression_interactions.csv',interactions),
                          ('reference_comparisons_new.csv',references),('visible_work.csv',works),
                          ('intended_children.csv',intended),('intended_pairs.csv',intended_pairs)):
        write_csv(filename,rows)
    print(canonical(dict(constructed_calls=len(items),status_counts=result['status_counts'],complete_parents=len(complete))))
    return result


if __name__=='__main__':
    analyze()
