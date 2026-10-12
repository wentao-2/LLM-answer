"""Export only audited dependence and observation-control results for the paper."""
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import argparse
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('phi4-standard-r6', 'qwen3-8b-r6')
CONDITIONS = ('ordinary', 'general_bayes', 'explicit', 'full_steps')
NAMES = {'ordinary': 'Ordinary', 'general_bayes': 'Bayes reminder',
         'explicit': 'Explicit observations', 'full_steps': 'Full steps'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row['family_id']].append(row)
    values = {family: mean(x[key] for x in group if x.get(key) is not None)
              for family, group in groups.items() if any(x.get(key) is not None for x in group)}
    return dict(n=sum(x.get(key) is not None for x in rows), families=len(values),
                planned_families=len(groups), family_values=values,
                estimate=mean(values.values()) if values and len(values) == len(groups) else None,
                family_range=[min(values.values()), max(values.values())] if values else None)


def grouped(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    return [dict(zip(keys, key), planned=len(group),
                 statuses=dict(Counter(x['status'] for x in group if 'status' in x)),
                 metrics={name: metric(group, name) for name in metrics})
            for key, group in sorted(groups.items(), key=lambda pair: json.dumps(pair[0]))]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def presentation_stats(path):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    matched = defaultdict(dict)
    for entry in records:
        item, rec = entry['source_item'], entry['source_record']
        assert rec['status']=='ok'
        key = item['parent_item_id'], entry['analysis_fields']['branch']
        assert item['history_arm'] not in matched[key]
        matched[key][item['history_arm']] = (item, rec)
    values = []
    for (parent, branch), arms in matched.items():
        assert set(arms)=={'native_self','replay_self'}
        item, native = arms['native_self']
        other_item, replay = arms['replay_self']
        assert all(item[k]==other_item[k] for k in ('initial_answer','family_id','signal','next_signal'))
        difference = native['probability']-replay['probability']
        values.append(dict(parent=parent,branch=branch,family_id=item['family_id'],
                           signed=(2*item['initial_answer']-1)*difference,absolute=abs(difference)))
    return dict(source_sha256=sha(path), direction='initial choice',
                rows=values, summaries=grouped(values,('branch',),('signed','absolute')))


def original_history_example(path):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    parents = defaultdict(dict)
    for entry in records:
        item = entry['source_item']
        if item['history_arm'] == 'native_self':
            branch = entry['analysis_fields']['branch']
            assert branch not in parents[item['parent_item_id']]
            parents[item['parent_item_id']][branch] = entry
    assert len(parents) == 48
    for branches in parents.values():
        assert set(branches) == {'none', 'copy', 'independent_agree', 'independent_conflict'}
        assert all(x['source_record']['status'] == 'ok' for x in branches.values())
        assert branches['none']['source_record']['probability'] == branches['none']['source_item']['initial_probability']
    ranked = sorted((abs(rows['copy']['source_record']['probability'] - rows['none']['source_record']['probability']), parent)
                    for parent, rows in parents.items())
    residual, parent = ranked[(len(ranked)-1)//2]
    selected = parents[parent]
    endpoints = {}
    for name, branches in (('no_reading', ('none',)), ('copy', ('copy',)),
                           ('independent', ('independent_agree', 'independent_conflict'))):
        errors = [dict(family_id=rows[b]['source_item']['family_id'],
                       error=abs(rows[b]['source_record']['probability'] - rows[b]['source_item']['oracle']))
                  for rows in parents.values() for b in branches]
        endpoints[name] = metric(errors, 'error')
    example = {}
    for branch, entry in selected.items():
        item, rec = entry['source_item'], entry['source_record']
        example[branch] = dict(item_id=item['id'], probability=rec['probability'],
            target=item['oracle'], status=rec['status'], messages=item['messages'],
            visible_output=json.loads(rec['metadata_json'])['final_output'])
    return dict(source_sha256=sha(path), selection='lower median absolute native copy update, parent ID breaks ties',
                parent_item_id=parent, absolute_copy_update=residual,
                endpoint_errors=endpoints, complete_visible_records=example)


def num(x):
    return '--' if x is None else f'{x:.4f}'


def tables(data):
    lines = ['% Generated by scripts/export_followup.py.', r'\begin{table*}[t]', r'\centering',
             r'{\small\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}', r'\toprule',
             r'Model & Question & Valid & MAE & \shortstack{Full design\\MAE bounds} & Pairs & \shortstack{MAE\\change} & \shortstack{Full design\\change bounds} \\', r'\midrule']
    compact = []
    for model in MODELS:
        for condition in CONDITIONS:
            row = next(x for x in data['condition_errors'] if x['model']==model and x['condition']==condition)
            m = row['metrics']
            estimate = m['posterior_error']['estimate']
            lo, hi = (m[k]['estimate'] for k in ('error_lower','error_upper'))
            if condition == 'ordinary':
                change, change_bounds, paired_n = '--', '--', 0
            else:
                contrast = next(x for x in data['condition_contrasts'] if x['model']==model
                    and x['condition']==condition and x['reference_condition']=='ordinary')
                cm = contrast['metrics']
                change = num(cm['posterior_error_change']['estimate'])
                change_bounds = f"[{num(cm['error_change_lower']['estimate'])}, {num(cm['error_change_upper']['estimate'])}]"
                paired_n = cm['posterior_error_change']['n']
            pair_count = '--' if condition == 'ordinary' else f'{paired_n}/{row["planned"]}'
            display_name = 'Explicit' if condition == 'explicit' else NAMES[condition]
            lines.append(f"{'Phi-4' if model==MODELS[0] else 'Qwen3-8B'} & {display_name} & {row['statuses'].get('ok',0)}/{row['planned']} & {num(estimate)} & [{num(lo)}, {num(hi)}] & {pair_count} & {change} & {change_bounds} " + r'\\')
            compact.append(dict(model=model,condition=condition,planned=row['planned'],valid=row['statuses'].get('ok',0),
                mae=estimate,mae_lower=lo,mae_upper=hi,paired_n=paired_n,paired_change=change,change_bounds=change_bounds))
        if model==MODELS[0]: lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular*}', '}',
              r'\caption{Probability error on new numerical settings and two contexts. Negative paired changes indicate improvement over ordinary questions. Bounds retain all planned outputs.}',
              r'\label{tab:followup-main}', r'\end{table*}', '']
    (ROOT/'sections/followup_main_generated.tex').write_text('\n'.join(lines), encoding='utf-8')
    with (ROOT/'data/followup/conditions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(compact[0])); writer.writeheader();writer.writerows(compact)
    for stem, rows, label, caption, keys in (
        ('followup_updates',data['branch_updates'],'tab:followup-updates',
         'Update errors by evidence branch. Counts refer to complete probability pairs. Bounds use all 32 planned pairs in each cell.',
         ('absolute_update_residual','absolute_update_lower','absolute_update_upper','effective_weight')),
    ):
        lines=['% Generated by scripts/export_followup.py.', r'\begin{table*}[t]',r'\centering',
               r'{\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrlrrrrr@{}}',r'\toprule',
               r'Model & Question & $\rho$ & Branch & Pairs & Mean $|R_B|$ & Full design bounds & Finite & Weight \\',r'\midrule']
        for model in MODELS:
            for condition in CONDITIONS:
                for row in sorted([x for x in rows if x['model']==model and x['condition']==condition],key=lambda x:(x['rho'],x['branch'])):
                    m=row['metrics'];v=[m[k]['estimate'] for k in keys]
                    lines.append(f"{'Phi-4' if model==MODELS[0] else 'Qwen3-8B'} & {NAMES[condition]} & {row['rho']:g} & {row['branch'].capitalize()} & {m[keys[0]]['n']}/{row['planned']} & {num(v[0])} & [{num(v[1])}, {num(v[2])}] & {m[keys[3]]['n']}/{row['planned']} & {num(v[3])} "+r'\\')
            # Split each model into its own full-width table to avoid a tall float.
            lines += [r'\bottomrule',r'\end{tabular*}','}',r'\caption{'+caption+'}',r'\label{'+label+('-phi' if model==MODELS[0] else '-qwen')+'}',r'\end{table*}','']
            (ROOT/f'sections/{stem}_{"phi" if model==MODELS[0] else "qwen"}_generated.tex').write_text('\n'.join(lines),encoding='utf-8')
            lines=lines[:8]
    return compact


def supplementary_tables(data):
    expression = ['% Generated by scripts/export_followup.py.', r'\begin{table}[t]',
        r'\centering', r'\begin{tabular}{llrr}', r'\toprule',
        r'Expression & Question & Valid & MAE \\',r'\midrule']
    for rule in ('direct','selector'):
        for condition in ('general','explicit'):
            row=next(x for x in data['expression_errors'] if x['rule']==rule and x['condition']==condition)
            expression.append(f"{rule.capitalize()} & {condition.capitalize()} & {row['statuses']['ok']}/{row['planned']} & {num(row['metrics']['posterior_error']['estimate'])} "+r'\\')
    expression += [r'\bottomrule',r'\end{tabular}',
        r'\caption{Equivalent expressions and conditioning requests on the same 24 Phi-4 histories. Each condition covers all 192 planned reports.}',
        r'\label{tab:followup-expression}',r'\end{table}','']
    (ROOT/'sections/followup_expression_generated.tex').write_text('\n'.join(expression),encoding='utf-8')
    lines=['% Generated by scripts/export_followup.py.',r'\begin{table*}[t]',r'\centering',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llllrrr@{}}',r'\toprule',
        r'Model & Question & Context & $S$ & Valid & MAE & Full design MAE bounds \\',r'\midrule']
    for model in MODELS:
        for condition in ('ordinary','explicit'):
            for row in [x for x in data['direction_context'] if x['model']==model and x['condition']==condition]:
                m=row['metrics']
                lo,hi=(m[k]['estimate'] for k in ('error_lower','error_upper'))
                lines.append(f"{'Phi-4' if model==MODELS[0] else 'Qwen3-8B'} & {NAMES[condition]} & {row['context'].capitalize()} & {row['signal']} & {m['posterior_error']['n']}/{row['planned']} & {num(m['posterior_error']['estimate'])} & [{num(lo)}, {num(hi)}] "+r'\\')
        if model==MODELS[0]:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular*}',
        r'\caption{Context and direction subdivisions of ordinary and explicit questions. Each cell includes eight branches for each numerical setting. Bounds retain all planned reports.}',
        r'\label{tab:followup-context}',r'\end{table*}','']
    (ROOT/'sections/followup_context_generated.tex').write_text('\n'.join(lines),encoding='utf-8')


def main(args):
    previous, latest = args.dependence.resolve(), args.followup.resolve()
    old=load(previous/'analysis/RESULTS.json');new=load(latest/'analysis/RESULTS.json')
    for folder, result in ((previous,old),(latest,new)):
        for name in ('AUDIT','LOCAL_AUDIT'):
            audit=load(folder/f'audit/{name}.json')
            assert audit['state']=='passed' and audit['analysis_sha256']==sha(folder/'analysis/RESULTS.json')
        for rel,digest in load(folder/'DESIGN_SEAL.json')['files'].items():
            assert sha(folder/rel)==digest,rel
    assert new['constructed_calls']==2880 and old['n_new_calls']==2112 and old['n_original_controls']==384
    transfer=[x for x in new['error_rows'] if x['stage']=='transfer']
    pairs=[x for x in new['pair_rows'] if x['stage']=='transfer']
    expression=[x for x in new['error_rows'] if x['stage']=='expression']
    contrasts=[x for x in new['contrast_rows'] if x['stage']=='transfer']
    em=('posterior_error','error_lower','error_upper')
    pm=('absolute_update_residual','absolute_update_lower','absolute_update_upper','effective_weight','normative_weight','absolute_weight_residual')
    cm=('posterior_error_change','error_change_lower','error_change_upper')
    data=dict(schema='Audited probability updating evidence',
        sources={'dependence':dict(analysis_sha256=sha(previous/'analysis/RESULTS.json'),design_sha256=sha(previous/'DESIGN_SEAL.json')),
                 'followup':dict(analysis_sha256=sha(latest/'analysis/RESULTS.json'),design_sha256=sha(latest/'DESIGN_SEAL.json'))},
        new_calls=dict(dependence=2112,followup=2880,total=4992),reused_controls=384,
        history_presentation=presentation_stats(previous/'inputs/original_controls.jsonl'),
        dependence_status=old['new_status_counts'],followup_status=new['status_counts'],
        dependence_points=load(previous/'figures/partial_dependence.data.json')['points'],
        reminder_points=load(previous/'figures/source_intervention.data.json')['points'],
        dependence_summaries=old['pair_summaries'],
        dependence_pairs=old['pair_rows'],dependence_errors=old['error_rows'],
        condition_errors=grouped(transfer,('model','condition'),em),
        expression_errors=grouped(expression,('rule','condition'),em),
        branch_updates=grouped(pairs,('model','condition','rho','branch'),pm),
        direction_context=grouped(transfer,('model','condition','context','signal'),em),
        condition_contrasts=grouped(contrasts,('model','comparison','condition','reference_condition'),cm),
        errors=new['error_rows'],pairs=new['pair_rows'],contrasts=new['contrast_rows'],
        expression_interactions=new['interaction_rows'],references=new['reference_rows'],
        complete_parent_ids=new['complete_parent_ids'],manual_source_coding='not used for manuscript claims',
        work_counts=[dict(model=m,n=len(rows),schema_valid=sum(x['work_schema_valid'] for x in rows),
                          observed_correct=sum(x['observed_correct'] is True for x in rows))
                     for m in MODELS for rows in [[x for x in new['work_rows'] if x['model']==m]]])
    destination=ROOT/'data/followup';destination.mkdir(parents=True,exist_ok=True)
    save_json(destination/'evidence.json',data)
    save_json(destination/'history_example.json',original_history_example(previous/'inputs/original_controls.jsonl'))
    compact=tables(data)
    supplementary_tables(data)
    # Compare publication aggregates with the separately saved experiment summary.
    summary=load(latest/'analysis/SUMMARY.json')
    for row in compact:
        original=next(x for x in summary['transfer_conditions'] if x['model']==row['model'] and x['condition']==row['condition'])
        assert row['valid']==original['status_counts'].get('ok',0)
        assert row['mae']==original['metrics']['posterior_error']['equal_family_mean']
        assert row['mae_lower']==original['metrics']['error_lower']['equal_family_mean']
        assert row['mae_upper']==original['metrics']['error_upper']['equal_family_mean']
    save_json(ROOT/'build/followup-export-check.json',dict(state='passed',new_calls=4992,reused_controls=384,
        evidence_sha256=sha(destination/'evidence.json'),sources=data['sources'],condition_rows=len(compact)))
    print(json.dumps(dict(state='passed',new_calls=4992,condition_rows=len(compact))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dependence',type=Path,required=True)
    parser.add_argument('--followup',type=Path,required=True)
    main(parser.parse_args())
