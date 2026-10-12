"""Additional independent audit of common cohorts and paired instruction contrasts.

This checker does not change the frozen estimator, parser, design or raw data.
"""
from pathlib import Path
from collections import defaultdict
import json,math,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from common import PHI,QWEN,canonical,load,now,read_db,read_jsonl,sha,validate_seal,write_json


def same(a,b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a,b,rel_tol=0,abs_tol=1e-10),(a,b)


def family_report(rows, metric):
    families=defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            families[row['family_id']].append(row[metric])
    values={f:math.fsum(v)/len(v) for f,v in families.items()}
    return dict(n=sum(len(v) for v in families.values()),n_families=len(values),family_values=values,
                equal_family_mean=math.fsum(values.values())/6 if len(values)==6 else None,
                family_range=[min(values.values()),max(values.values())] if values else None)


def compare(report,rows,metrics,count_field):
    assert report['n_planned']==len(rows)
    assert report[count_field]==sum(row.get(metrics[0]) is not None for row in rows)
    for metric in metrics:
        expected=family_report(rows,metric)
        actual=report[metric]
        assert expected['n']==actual['n'] and expected['n_families']==actual['n_families']
        assert set(expected['family_values'])==set(actual['family_values'])
        for family,value in expected['family_values'].items():same(value,actual['family_values'][family])
        same(expected['equal_family_mean'],actual['equal_family_mean'])
        if expected['family_range']:
            for a,b in zip(expected['family_range'],actual['family_range']):same(a,b)
        else:assert actual['family_range'] is None


def audit():
    validate_seal()
    initial=load(ROOT/'audit/AUDIT.json')
    assert initial['state']=='passed' and initial['analysis_sha256']==sha(ROOT/'analysis/RESULTS.json')
    data=load(ROOT/'analysis/RESULTS.json')
    errors,pairs=data['error_rows'],data['pair_rows']
    group=defaultdict(list)
    for row in errors:group[(row['model'],row['module'],row['parent_item_id'])].append(row)
    common=defaultdict(set)
    within=defaultdict(set)
    for (model,module,parent),rows in group.items():
        expected=28 if module=='partial_dependence' else 24
        assert len(rows)==expected
        if all(row['status']=='ok' for row in rows):common[(model,module)].add(parent)
        if module=='partial_dependence':
            for rho in (0,.25,.5,.75,1):
                current=[row for row in rows if row['rho']==rho]
                assert len(current)==(4 if rho==1 else 6)
                if all(row['status']=='ok' for row in current):within[(model,rho)].add(parent)
    for report in data['common_complete_sets']:
        expected=(within[(report['model'],report['rho'])] if report['module']=='partial_dependence_within_rho'
                  else common[(report['model'],report['module'])])
        assert set(report['parent_ids'])==expected and report['n_complete']==len(expected)
    for row in errors+pairs:
        assert row['common_parent_all_cells']==(row['parent_item_id'] in common[(row['model'],row['module'])])
        if row['module']=='partial_dependence':
            assert row['common_parent_within_rho']==(row['parent_item_id'] in within[(row['model'],row['rho'])])
        else:assert row['common_parent_within_rho'] is None
    keys=('model','module','history_arm','rho','condition','branch')
    metrics_error=('posterior_absolute_error','choice_correct','error_lower','error_upper','accuracy_lower','accuracy_upper')
    metrics_pair=('probability_update_residual','absolute_probability_update_residual','effective_weight',
        'weight_residual','absolute_weight_residual','absolute_weight','update_residual_lower','update_residual_upper',
        'absolute_update_residual_lower','absolute_update_residual_upper')
    counts=0
    for reports,rows,metrics,count_field in ((data['summaries'],errors,metrics_error,'n_valid'),
            (data['pair_summaries'],pairs,metrics_pair,'n_complete_probability_pairs')):
        groups=defaultdict(list)
        for row in rows:groups[tuple(row[k] for k in keys)].append(row)
        assert len(reports)==len(groups)
        for report in reports:
            members=groups[tuple(report[k] for k in keys)]
            compare(report,members,metrics,count_field)
            common_members=[row for row in members if row['common_parent_all_cells']]
            compare(report['common_all'],common_members,metrics if rows is pairs else metrics[:2],count_field)
            if rows is pairs and report['module']=='partial_dependence':
                within_members=[row for row in members if row['common_parent_within_rho']]
                compare(report['common_within_rho'],within_members,metrics,count_field)
            counts+=1
    # The normative log odds increment is independently derived from the stored exact posteriors.
    new_items={x['id']:x for x in read_jsonl(ROOT/'manifests/all.jsonl')}
    old_items={x['source_item']['id']:x['source_item'] for x in read_jsonl(ROOT/'inputs/original_controls.jsonl')}
    for row in pairs:
        source=old_items[row['item_id']] if row.get('condition')=='original' else new_items[row['item_id']]
        q,q0=row['oracle'],row['none_oracle']
        delta=math.log((q/(1-q))/(q0/(1-q0)))
        ref=(2*source['next_signal']-1)*delta/math.log(source['reliability']/(1-source['reliability']))
        same(ref,row['normative_weight'])
    index={(row['parent_item_id'],row['history_arm'],row['condition'],row['branch']):row
           for row in pairs if row['module']=='source_intervention'}
    contrast_metrics=('posterior_absolute_error','absolute_probability_update_residual','effective_weight','absolute_weight_residual')
    for row in data['contrast_rows']:
        treated=index[(row['parent_item_id'],row['history_arm'],row['condition'],row['branch'])]
        control=index[(row['parent_item_id'],row['history_arm'],'original',row['branch'])]
        for metric in contrast_metrics:
            a,b=treated[metric],control[metric]
            same(a-b if a is not None and b is not None else None,row[metric+'_change'])
        assert row['common_parent_all_cells']==treated['common_parent_all_cells']
    for report in data['contrast_summaries']:
        rows=[row for row in data['contrast_rows'] if all(row[k]==report[k] for k in ('history_arm','condition','branch'))]
        metrics=tuple(m+'_change' for m in contrast_metrics)
        compare(report,rows,metrics,'n_complete')
        compare(report['common_all'],[row for row in rows if row['common_parent_all_cells']],metrics,'n_complete')
        counts+=1
    # Connect actual rendered prompt strings to the complete preflight ledger.
    rendered=0
    for label,model in (('phi',PHI),('qwen',QWEN)):
        reference=load(ROOT/'preflight'/f'{model}-render.json')
        expected={row['item_id']:row for row in reference['ledger']}
        for rec in read_db(ROOT/'queues'/label/'merged.sqlite'):
            metadata=json.loads(rec['metadata_json'])
            row=expected[rec['item_id']]
            assert metadata['rendered_prompt_sha256']==row['rendered_prompt_sha256']
            assert rec['input_tokens']==row['input_tokens']
            rendered+=1
    assert rendered==2112
    report=dict(state='passed',checked_at=now(),analysis_sha256=sha(ROOT/'analysis/RESULTS.json'),
        primary_audit_sha256=sha(ROOT/'audit/AUDIT.json'),aggregate_groups=counts,
        contrasts=len(data['contrast_rows']),actual_rendered_prompts=rendered,
        checks=['all common parent ID sets','all complete-set and conditional family means',
                'normative log odds weights','paired instruction contrast values','actual native rendering versus preflight'],
        estimator_changes=False,original_data_changes=False)
    destination=ROOT/'audit/AGGREGATES.json'
    if destination.exists():
        saved=load(destination)
        assert {k:v for k,v in saved.items() if k!='checked_at'}=={k:v for k,v in report.items() if k!='checked_at'}
        report=saved
    else:
        write_json(destination,report)
    print(canonical(report))


if __name__=='__main__':audit()
