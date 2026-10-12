"""Strict-readout analysis of terminal R5 records under frozen cohort rules."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import sys

import inference_helpers as h
sys.path.insert(0,str(h.ROOT/'source'))
import analyze_extension as a
import build_extension as b

ROOT=h.ROOT
OUT=ROOT/'inference/analysis'


def dump(name,value):
    h.atomic_json(OUT/name,value)


def snapshot(stage,manifest,model):
    records=a.snapshot(b.read_jsonl(ROOT/'inference/manifests'/manifest),ROOT/'inference/results'/f'{stage}.sqlite',model)
    assert not any(r.get('exclusion') for r in records)
    assert all(r['status'] in ('ok','invalid','truncated','error') for r in records)
    return records


def history_results(parents,children):
    by_id={r['item']['id']:r for r in parents}
    parent_quality={}
    parent_export=[]
    for r in parents:
        x=r['item']
        valid=a.valid(r)
        prefix=r['metadata'].get('final_output','')
        prefix=prefix.split('FINAL_JSON:',1)[0].strip() if isinstance(prefix,str) else ''
        quality=dict(valid=valid,visible_prefix=bool(prefix),parent_condition=x['parent_condition'])
        if valid:
            p,q=r['probability'],x['oracle']
            c=int(p>=.5)
            quality.update(choice_correct=c==int(q>=.5),probability_error=abs(p-q),
                large_error=abs(p-q)>.05,overconfident=(2*c-1)*(p-q)>.05,choice=c)
        parent_quality[x['id']]=quality
        parent_export.append(dict(id=x['id'],model=r['model'],family=x['family_id'],oracle=x['oracle'],
            probability=r['probability'],status=r['status'],**quality))
    cohorts={'all':set(k for k,v in parent_quality.items() if v['valid'])}
    for condition in sorted({v['parent_condition'] for v in parent_quality.values()}):
        ids={k for k,v in parent_quality.items() if v['valid'] and v['parent_condition']==condition}
        cohorts['producer:'+condition]=ids
    for label,fn in (
        ('correct_choice',lambda v:v['choice_correct']),('wrong_choice',lambda v:not v['choice_correct']),
        ('small_error',lambda v:not v['large_error']),('large_error',lambda v:v['large_error']),
        ('overconfident',lambda v:v['overconfident']),('other_confidence',lambda v:not v['overconfident']),
        ('visible_prefix',lambda v:v['visible_prefix']),('no_visible_prefix',lambda v:not v['visible_prefix'])):
        cohorts[label]={k for k,v in parent_quality.items() if v['valid'] and fn(v)}
    results=[]
    item_updates=[]
    for label,ids in cohorts.items():
        records=[r for r in children if r['item']['parent_item_id'] in ids]
        assert len(records)==16*len(ids),(label,len(records),len(ids))
        result=dict(cohort=label,n_parents=len(ids),n_planned_children=len(records),
            family_counts=dict(Counter(by_id[k]['item']['family_id'] for k in ids)),
            parent_mae=a.estimate([(by_id[k]['item']['family_id'],abs(by_id[k]['probability']-by_id[k]['item']['oracle'])) for k in ids]))
        if not records:
            result['outcome']='empty_cohort_not_tested'
            results.append(result)
            continue
        result['analysis']=a.history_analyses(records)
        result['arms']=[dict(history_arm=arm,evidence_kind=kind,**a.describe(rows)) for (arm,kind),rows in
            sorted(a.groups_by(records,['history_arm','evidence_kind']).items())]
        baseline={}
        for r in records:
            x=r['item']
            if x['evidence_kind']=='none':
                key=(x['parent_item_id'],x['history_arm'])
                assert key not in baseline
                baseline[key]=r
        updates=defaultdict(list)
        corrections=defaultdict(list)
        for r in records:
            x=r['item']
            if x['evidence_kind']=='none':
                parent=by_id[x['parent_item_id']]
                if a.valid(r):
                    c=parent_quality[x['parent_item_id']]['choice']
                    corrections[x['history_arm']].append((x['family_id'],(2*c-1)*(r['probability']-parent['probability'])))
                continue
            hidden=baseline[(x['parent_item_id'],x['history_arm'])]
            if not (a.valid(r) and a.valid(hidden)):
                continue
            direction=2*x['next_signal']-1
            residual=direction*((r['probability']-hidden['probability'])-(x['oracle']-hidden['item']['oracle']))
            updates[(x['history_arm'],x['evidence_kind'])].append((x['family_id'],residual))
            if label=='all':
                item_updates.append(dict(model=r['model'],parent_id=x['parent_item_id'],visible_id=x['id'],
                    baseline_id=hidden['item']['id'],history_arm=x['history_arm'],evidence_kind=x['evidence_kind'],
                    family=x['family_id'],residual=residual,absolute_residual=abs(residual)))
        result['individual_update_errors']=[dict(history_arm=arm,evidence_kind=kind,
            signed=a.estimate(values),absolute=a.estimate([(f,abs(v)) for f,v in values]),
            max_absolute=max(abs(v) for _,v in values),
            planned_pairs=(2 if kind=='independent' else 1)*len(ids)) for (arm,kind),values in sorted(updates.items())]
        result['no_new_evidence_change_from_initial']={arm:a.estimate(values) for arm,values in corrections.items()}
        results.append(result)
    return dict(parent_planned=len(parents),parent_status_counts=dict(Counter(r['status'] for r in parents)),
        eligible_parents=len(cohorts['all']),attained_cohorts={k:len(v) for k,v in cohorts.items()},
        parent_records=parent_export,cohorts=results,item_update_errors=item_updates)


def paraphrase_results(records):
    def selectors():
        for coords,subset in sorted(a.groups_by(records,['reminder','output_instruction']).items()):
            label=dict(zip(['reminder','output_instruction'],coords))
            yield label,subset
            for (template,),rows in sorted(a.groups_by(subset,['template_id']).items()):
                yield dict(label,template_id=template),rows
    all_rows=[]
    for label,subset in selectors():
        grouped=a.groups_by(subset,a.MATCH)
        nine=[]
        for _,rows in grouped.items():
            cells={(r['item']['paraphrase_variant'],r['item']['attribution']):r for r in rows}
            assert len(cells)==9 and len(rows)==9
            assert len({r['item']['oracle'] for r in rows})==1
            nine.append(cells)
        result=dict(stratum=label,planned_sets=len(nine),complete_nine_sets=sum(all(a.valid(r) for r in c.values()) for c in nine),
                    versions={},paired_version_changes={})
        for version in ('original','authorship','dependence'):
            metrics={}
            for metric in ('gap','self_error','other_error','neutral_error','self_mae','other_mae','neutral_mae'):
                units=[]
                for c in nine:
                    first=c[(version,'self')]['item']; d=2*first['answer']-1
                    if metric=='gap':
                        terms=[(c[(version,'self')],d),(c[(version,'other')],-d)]
                        scale='probability'
                    else:
                        author=metric.split('_')[0]
                        terms=[(c[(version,author)],1 if metric.endswith('mae') else d)]
                        scale='absolute_error' if metric.endswith('mae') else 'residual'
                    units.append(a.unit(terms,first['family_id'],required=list(c.values())))
                metrics[metric]=a.summarize_units(units,scale)
            metrics['complete_triads']=sum(all(a.valid(c[(version,g)]) for g in ('self','other','neutral')) for c in nine)
            result['versions'][version]=metrics
        for version in ('authorship','dependence'):
            units=[]
            for c in nine:
                first=c[('original','self')]['item']; d=2*first['answer']-1
                terms=[(c[(version,'self')],d),(c[(version,'other')],-d),
                       (c[('original','self')],-d),(c[('original','other')],d)]
                units.append(a.unit(terms,first['family_id'],required=list(c.values())))
            result['paired_version_changes'][version]=a.summarize_units(units)
        gaps=[result['versions'][v]['gap']['point']['estimate'] for v in ('original','authorship','dependence')]
        result['observed_gap_range_same_nine_sets']=[min(gaps),max(gaps)] if all(v is not None for v in gaps) else None
        all_rows.append(result)
    return dict(overall=a.describe(records),strata=all_rows,comparison_unit='nine reports for primary points; triad coverage also reported')


def main():
    status=json.loads((ROOT/'inference/STATUS.json').read_text())
    assert status['state']=='completed','All stages must terminate before final analysis'
    assert not (ROOT/'inference/FAILURE.json').exists()
    OUT.mkdir(exist_ok=True)
    qwen='qwen3-8b-r5'
    parents=snapshot('qwen-parents','qwen-parents.jsonl',qwen)
    children=snapshot('qwen-main','qwen-history.jsonl',qwen)
    para_manifest=b.read_jsonl(ROOT/'inference/manifests/paraphrase.jsonl')+b.read_jsonl(ROOT/'inference/manifests/paraphrase-neutral.jsonl')
    para=a.snapshot(para_manifest,ROOT/'inference/results/qwen-main.sqlite',qwen)
    assert not any(r.get('exclusion') for r in para)
    dump('qwen_history.json',history_results(parents,children))
    dump('paraphrases.json',paraphrase_results(para))
    gate=json.loads((ROOT/'inference/results/phi-gate-analysis.json').read_text())
    dump('phi_gate.json',gate)
    if gate['passes']:
        phi='phi4-reasoning-r5'
        cross=a.analyze(ROOT/'inference/manifests/phi-cross.jsonl',ROOT/'inference/results/phi-main.sqlite',[phi])
        dump('phi_cross.json',cross)
        phi_parents=snapshot('phi-main','phi-parents.jsonl',phi)
        phi_children=snapshot('phi-history','phi-history.jsonl',phi) if (ROOT/'inference/results/phi-history.sqlite').exists() else []
        dump('phi_history.json',history_results(phi_parents,phi_children))
    else:
        dump('phi_downstream.json',dict(state='not_constructed_failed_gate',criterion=gate))
    seals={p.name:json.loads(p.read_text()) for p in (ROOT/'inference/results').glob('*.seal.json')}
    dump('PROVENANCE.json',dict(analysis_code_sha256=h.digest(Path(__file__)),helper_sha256=h.digest(ROOT/'inference_helpers.py'),
        stage_seals=seals,model_inventory_sha256=h.digest(ROOT/'inference/MODEL_INVENTORY.json'),
        original_primary_records_changed=False,all_stages_terminal=True,
        files={p.name:h.digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='PROVENANCE.json'}))
    print(json.dumps(dict(qwen_valid_parents=sum(a.valid(r) for r in parents),history_children=len(children),paraphrase_calls=len(para),phi_passes=gate['passes'])))


if __name__=='__main__':
    main()
