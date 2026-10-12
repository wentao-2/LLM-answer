"""Frozen descriptive analysis of all planned outputs, including missing bounds."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from statistics import mean

from common import (ROOT, PHI, QWEN, canonical, load, now, read_db, read_jsonl,
                    sha, validate_seal, write_json)
from oracle import target, logit, normative_weight


def valid(rec):
    return (rec is not None and rec['status'] == 'ok' and type(rec['probability']) in (int, float)
            and math.isfinite(rec['probability']) and 0 <= rec['probability'] <= 1)


def family_mean(rows, value):
    groups = defaultdict(list)
    for x in rows:
        v = x.get(value)
        if v is not None:
            groups[x['family_id']].append(v)
    family = {f: mean(v) for f, v in sorted(groups.items())}
    return dict(n=len([x for x in rows if x.get(value) is not None]),
                n_families=len(family), family_values=family,
                equal_family_mean=mean(family.values()) if len(family)==6 else None,
                family_range=[min(family.values()), max(family.values())] if family else None)


def summarize(rows, values, count_name):
    result = dict(n_planned=len(rows), **{count_name: sum(x.get(values[0]) is not None for x in rows)})
    for value in values:
        result[value] = family_mean(rows, value)
    return result


def error_row(item, rec):
    q = float(target(item))
    ok = valid(rec)
    p = rec['probability'] if ok else None
    error = abs(p-q) if ok else None
    correct = int((p >= .5) == (q >= .5)) if ok else None
    return dict(item_id=item['id'], family_id=item['family_id'], parent_item_id=item['parent_item_id'],
        model=item['model_scope'][0], module=item['module'], condition=item.get('condition'),
        rho=item.get('rho'), history_arm=item['history_arm'], branch=item['branch'],
        status=rec['status'] if rec else 'missing_record', probability=p, oracle=q,
        posterior_absolute_error=error, choice_correct=correct,
        error_lower=error if ok else 0, error_upper=error if ok else max(q, 1-q),
        accuracy_lower=correct if ok else 0, accuracy_upper=correct if ok else 1,
        endpoint=ok and p in (0, 1))


def pair_row(item, rec, none_item, none_rec):
    q_b, q_0 = float(target(item)), float(target(none_item))
    dq = q_b-q_0
    ok_b, ok_0 = valid(rec), valid(none_rec)
    pb, p0 = rec['probability'] if ok_b else None, none_rec['probability'] if ok_0 else None
    update = pb-p0 if ok_b and ok_0 else None
    residual = update-dq if update is not None else None
    low_b, high_b = (pb, pb) if ok_b else (0, 1)
    low_0, high_0 = (p0, p0) if ok_0 else (0, 1)
    lo, hi = low_b-high_0-dq, high_b-low_0-dq
    abs_lo = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    abs_hi = max(abs(lo), abs(hi))
    x = error_row(item, rec)
    x.update(none_item_id=none_item['id'], none_status=none_rec['status'] if none_rec else 'missing_record',
        none_probability=p0, none_oracle=q_0, probability_update=update, oracle_update=dq,
        probability_update_residual=residual,
        absolute_probability_update_residual=abs(residual) if residual is not None else None,
        update_residual_lower=lo, update_residual_upper=hi,
        absolute_update_residual_lower=abs_lo, absolute_update_residual_upper=abs_hi)
    r, b = item['reliability'], item['next_signal']
    norm = normative_weight(r, item.get('rho', 0), item['branch'])
    weight = ((2*b-1)*(logit(pb)-logit(p0))/math.log(r/(1-r))
              if ok_b and ok_0 and 0 < pb < 1 and 0 < p0 < 1 else None)
    x.update(normative_weight=norm, effective_weight=weight,
        weight_residual=weight-norm if weight is not None else None,
        absolute_weight_residual=abs(weight-norm) if weight is not None else None,
        absolute_weight=abs(weight) if weight is not None else None,
        logit_state='finite' if weight is not None else ('endpoint' if ok_b and ok_0 else 'missing_pair'),
        endpoint_pattern=[pb, p0] if ok_b and ok_0 and weight is None else None)
    return x


def load_observations():
    items = read_jsonl(ROOT/'manifests/all.jsonl')
    records = {}
    for model, queue in ((PHI, 'phi'), (QWEN, 'qwen')):
        path = ROOT/'queues'/queue/'merged.sqlite'
        assert path.is_file(), 'Only complete merged stages may enter final analysis'
        chunk = read_db(path)
        assert len(chunk) == sum(x['model_scope'] == [model] for x in items)
        for rec in chunk:
            key = (model, rec['item_id'])
            assert key not in records and rec['model'] == model
            records[key] = rec
    assert set(records) == {(x['model_scope'][0], x['id']) for x in items}
    for c in read_jsonl(ROOT/'inputs/original_controls.jsonl'):
        item = dict(c['source_item'], **c['analysis_fields'])
        rec = c['source_record']
        key = (PHI, item['id'])
        assert key not in records
        items.append(item)
        records[key] = rec
    return items, records


def subgroup(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return sorted(groups.items(), key=lambda x: canonical(x[0]))


def analyze():
    seal = validate_seal()
    items, records = load_observations()
    errors = [error_row(x, records[(x['model_scope'][0], x['id'])]) for x in items]
    # Each rho/condition and presentation gets its own no record baseline.
    baselines = {}
    for x in items:
        if x['branch'] == 'none':
            key = (x['model_scope'][0], x['module'], x['parent_item_id'], x['history_arm'],
                   x.get('rho'), x.get('condition'))
            assert key not in baselines
            baselines[key] = x
    pairs = []
    for x in items:
        if x['branch'] == 'none':
            continue
        model = x['model_scope'][0]
        key = (model, x['module'], x['parent_item_id'], x['history_arm'], x.get('rho'), x.get('condition'))
        n = baselines[key]
        pairs.append(pair_row(x, records[(model, x['id'])], n, records[(model, n['id'])]))
    complete_parents, complete_within = {}, {}
    for (model, module), rows in subgroup(errors, ('model', 'module')):
        parent_rows = defaultdict(list)
        for row in rows:
            parent_rows[row['parent_item_id']].append(row)
        expected = 28 if module == 'partial_dependence' else 24
        assert all(len(v) == expected for v in parent_rows.values())
        complete_parents[(model, module)] = {pid for pid, v in parent_rows.items()
            if all(x['posterior_absolute_error'] is not None for x in v)}
        if module == 'partial_dependence':
            for rho in (0, .25, .5, .75, 1):
                complete_within[(model, rho)] = {pid for pid, v in parent_rows.items()
                    if all(x['posterior_absolute_error'] is not None for x in v if x['rho'] == rho)}
    for row in errors+pairs:
        row['common_parent_all_cells'] = row['parent_item_id'] in complete_parents[(row['model'], row['module'])]
        row['common_parent_within_rho'] = (row['parent_item_id'] in complete_within[(row['model'], row['rho'])]
                                          if row['module'] == 'partial_dependence' else None)
    key_fields = ('model', 'module', 'history_arm', 'rho', 'condition', 'branch')
    summaries = []
    for key, rows in subgroup(errors, key_fields):
        summaries.append(dict(zip(key_fields, key), status_counts=dict(Counter(x['status'] for x in rows)),
            n_endpoints=sum(x['endpoint'] for x in rows),
            **summarize(rows, ('posterior_absolute_error', 'choice_correct', 'error_lower', 'error_upper',
                               'accuracy_lower', 'accuracy_upper'), 'n_valid'),
            common_all=summarize([x for x in rows if x['common_parent_all_cells']],
                                ('posterior_absolute_error', 'choice_correct'), 'n_valid')))
    pair_summaries = []
    metrics = ('probability_update_residual', 'absolute_probability_update_residual', 'effective_weight',
               'weight_residual', 'absolute_weight_residual', 'absolute_weight',
               'update_residual_lower', 'update_residual_upper',
               'absolute_update_residual_lower', 'absolute_update_residual_upper')
    for key, rows in subgroup(pairs, key_fields):
        pair_summaries.append(dict(zip(key_fields, key), logit_states=dict(Counter(x['logit_state'] for x in rows)),
            **summarize(rows, metrics, 'n_complete_probability_pairs'),
            common_all=summarize([x for x in rows if x['common_parent_all_cells']], metrics, 'n_complete_probability_pairs'),
            common_within_rho=summarize([x for x in rows if x['common_parent_within_rho']], metrics, 'n_complete_probability_pairs')
                              if key[1] == 'partial_dependence' else None))
    # Pair instruction comparisons on precisely the same parent and presentation.
    intervention = {(x['parent_item_id'], x['history_arm'], x['condition'], x['branch']): x
                    for x in pairs if x['module'] == 'source_intervention'}
    contrasts = []
    for (pid, arm, condition, branch), row in sorted(intervention.items()):
        if condition == 'original':
            continue
        control = intervention[(pid, arm, 'original', branch)]
        x = dict(parent_item_id=pid, family_id=row['family_id'], history_arm=arm, condition=condition, branch=branch,
                 common_parent_all_cells=row['common_parent_all_cells'])
        for metric in ('posterior_absolute_error', 'absolute_probability_update_residual', 'effective_weight', 'absolute_weight_residual'):
            a, b = row.get(metric), control.get(metric)
            x[metric+'_change'] = a-b if a is not None and b is not None else None
        contrasts.append(x)
    contrast_summaries = []
    contrast_metrics = ('posterior_absolute_error_change', 'absolute_probability_update_residual_change',
                        'effective_weight_change', 'absolute_weight_residual_change')
    for key, rows in subgroup(contrasts, ('history_arm', 'condition', 'branch')):
        contrast_summaries.append(dict(zip(('history_arm', 'condition', 'branch'), key),
            **summarize(rows, contrast_metrics, 'n_complete'),
            common_all=summarize([x for x in rows if x['common_parent_all_cells']], contrast_metrics, 'n_complete')))
    common = []
    for (model, module), parents in sorted(complete_parents.items()):
        common.append(dict(model=model, module=module, n_complete=len(parents), parent_ids=sorted(parents)))
    for (model, rho), parents in sorted(complete_within.items()):
        common.append(dict(model=model, module='partial_dependence_within_rho', rho=rho,
                           n_complete=len(parents), parent_ids=sorted(parents)))
    result = dict(analyzed_at=now(), design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'),
        n_new_calls=2112, n_original_controls=384, source_hashes=load(ROOT/'inputs/source_hashes.json'),
        status_counts={model: dict(Counter(r['status'] for (m, _), r in records.items() if m == model)) for model in (PHI, QWEN)},
        new_status_counts={model: dict(Counter(records[(model,x['id'])]['status'] for x in items
            if x['model_scope'][0] == model and x.get('condition') != 'original')) for model in (PHI,QWEN)},
        summaries=summaries, pair_summaries=pair_summaries, common_complete_sets=common,
        contrast_summaries=contrast_summaries, error_rows=errors, pair_rows=pairs, contrast_rows=contrasts,
        inference_scope='fixed factorial grid; descriptive family means and ranges; no population significance claim',
        logit_missing_bounds='No finite universal bounds. Finite logit estimates exclude and count endpoint probabilities without clipping.')
    (ROOT/'analysis').mkdir(exist_ok=True)
    write_json(ROOT/'analysis/RESULTS.json', result)
    for name, rows in (('probability_errors', errors), ('paired_updates', pairs), ('intervention_contrasts', contrasts)):
        path = ROOT/'analysis'/f'{name}.csv'
        with path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    print(json.dumps(dict(new_status_counts=result['new_status_counts'], common_complete_sets=common)))
    return result


if __name__ == '__main__':
    analyze()
