"""Frozen-field analysis for I16 extension. No generation-output reparsing."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3

from build_extension import canonical, file_hash, oracle, probability, read_jsonl


def valid(record):
    return record['status'] == 'ok' and not record.get('exclusion') and probability(record.get('probability'))


def key(record):
    return record['model'] + ':' + record['item']['id']


def snapshot(manifest, db_path, model):
    planned = [x for x in manifest if not x.get('model_scope') or model in x['model_scope']]
    if not planned or len({r['id'] for r in planned}) != len(planned):
        raise ValueError('No eligible items or duplicate manifest IDs')
    c = sqlite3.connect(Path(db_path).resolve().as_uri()+'?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    index = defaultdict(list)
    for row in c.execute('SELECT * FROM results WHERE model=?', (model,)):
        index[row['item_id']].append(dict(row))
    c.close()
    records = []
    for item in planned:
        hits = index.get(item['id'], [])
        rec = dict(model=model, item=item, status='unobserved', probability=None, metadata={})
        if len(hits) > 1:
            rec.update(status='duplicate_records', exclusion='duplicate_model_item')
        elif hits:
            hit = hits[0]
            meta = json.loads(hit['metadata_json'])
            rec.update(status=hit['status'], probability=hit['probability'], metadata=meta,
                       output_tokens=hit['output_tokens'])
            mismatch = [k for k, v in item.items() if meta.get(k) != v]
            if mismatch:
                rec['exclusion'] = 'metadata_mismatch'
                rec['mismatched_fields'] = mismatch
            elif rec['status'] == 'ok' and not probability(rec['probability']):
                rec['exclusion'] = 'invalid_stored_probability'
        q = oracle(item['protocol'], item['visibility'], item['prior'], item['reliability'],
                   item['signal'], item['answer'])
        if not math.isclose(q, item['oracle'], abs_tol=1e-12, rel_tol=0):
            raise ValueError('Oracle mismatch '+item['id'])
        records.append(rec)
    return records


def estimate(values):
    groups = defaultdict(list)
    for family, value in values:
        groups[family].append(value)
    means = {f: sum(v)/len(v) for f, v in sorted(groups.items())}
    return dict(estimate=sum(means.values())/len(means) if means else None,
                n_observations=len(values), n_families=len(means), family_means=means)


def describe(records):
    ok = [r for r in records if valid(r)]
    result = dict(n_planned=len(records), n_valid=len(ok),
                  valid_rate=len(ok)/len(records) if records else None,
                  status_counts=dict(Counter(r['status'] for r in records)),
                  exclusions=dict(Counter(r['exclusion'] for r in records if r.get('exclusion'))),
                  mae=estimate([(r['item']['family_id'], abs(r['probability']-r['item']['oracle'])) for r in ok]),
                  signed_residual=estimate([(r['item']['family_id'], r['probability']-r['item']['oracle']) for r in ok]))
    directional = [r for r in ok if r['item'].get('answer') in (0, 1)]
    result['directional_error'] = estimate([(r['item']['family_id'], (2*r['item']['answer']-1)*
                                            (r['probability']-r['item']['oracle'])) for r in directional])
    result['generation'] = dict(
        output_tokens_total=sum(r.get('output_tokens') or 0 for r in records),
        closed_thinking=sum(r['metadata'].get('thinking_segment_closed') is True for r in records),
        empty_thinking=sum(r['metadata'].get('reasoning_nonempty') is False for r in records))
    return result


def analyze_gate(manifest_path, db_path, model):
    """Public scheduler API: frozen manifest + merged terminal SQLite + model ID."""
    manifest = read_jsonl(manifest_path) if isinstance(manifest_path, (str, Path)) else manifest_path
    records = snapshot(manifest, db_path, model)
    groups = defaultdict(list)
    for r in records:
        if r['item']['module'] != 'gate':
            raise ValueError('Gate API received non-gate items')
        groups[r['item'].get('gate_group', r['item']['protocol'])].append(r)
    expected = {'positive': 16, 'zero': 16, 'negative': 16, 'baseline': 16, 'independent': 32}
    if set(groups) not in (set(expected), {'independent'}):
        raise ValueError('Unexpected gate groups')
    details = []
    for name, rows in sorted(groups.items()):
        stats = describe(rows)
        mae = stats['mae']['estimate']
        passed = (len(rows) == expected[name] and all(r['status'] in ('ok', 'invalid', 'truncated', 'error') for r in rows)
                  and stats['valid_rate'] >= .95
                  and mae is not None and mae <= .05 and not stats['exclusions'])
        detail = dict(group=name, passes=passed, **stats)
        if name == 'independent':
            detail['relation'] = {rel: describe([r for r in rows if
                ('agree' if r['item']['signal'] == r['item']['answer'] else 'conflict') == rel])
                for rel in ('agree', 'conflict')}
        details.append(detail)
    return dict(model=model, passes=all(d['passes'] for d in details),
                qualification_scope='five_groups' if len(groups) == 5 else 'independent_extension_only',
                thresholds=dict(minimum_valid_rate=.95, maximum_family_mae=.05), groups=details)


def unit(terms, family, offset=0.0, **meta):
    return dict(terms=terms, family=family, offset=offset, **meta)


def score(u, metric, normative=False):
    value = u.get('offset', 0.0)
    for r, coef in u['terms']:
        q = r['item']['oracle']
        p = q if normative else r['probability']
        value += coef * (abs(p-q) if metric == 'absolute_error' else p-q if metric == 'residual' else p)
    return value


def full_bounds(units, metric='probability'):
    """Sharp finite-design linear bounds, merging every shared record coefficient."""
    by_family = Counter(u['family'] for u in units)
    if not by_family:
        return None
    coefs = defaultdict(float)
    refs = {}
    constant = 0.0
    for u in units:
        weight = 1 / (len(by_family)*by_family[u['family']])
        constant += weight*u.get('offset', 0.0)
        for r, coef in u['terms']:
            rid = key(r)
            if rid in refs and refs[rid]['item']['oracle'] != r['item']['oracle']:
                raise ValueError('A shared record has inconsistent oracles')
            refs[rid] = r
            coefs[rid] += weight*coef
    lo = hi = constant
    for rid, coef in coefs.items():
        r, q = refs[rid], refs[rid]['item']['oracle']
        if valid(r):
            p = r['probability']
            z = abs(p-q) if metric == 'absolute_error' else p-q if metric == 'residual' else p
            lo += coef*z
            hi += coef*z
        else:
            a, b = ((0., max(q, 1-q)) if metric == 'absolute_error' else
                    (-q, 1-q) if metric == 'residual' else (0., 1.))
            lo += min(coef*a, coef*b)
            hi += max(coef*a, coef*b)
    return dict(lower=lo, upper=hi, n_planned_units=len(units), n_families=len(by_family),
                n_unique_records=len(refs), n_missing_records=sum(not valid(r) for r in refs.values()),
                n_missing_nonzero_coefficients=sum(not valid(refs[k]) and abs(v) > 1e-15 for k, v in coefs.items()),
                method='fixed finite design bounds; distinct missing probabilities in [0,1]; shared coefficients merged')


def summarize_units(units, metric='probability'):
    complete = [u for u in units if all(valid(r) for r in u.get('required', [r for r, _ in u['terms']]))]
    result = dict(metric=metric, candidate_sets=len(units), complete_valid_sets=len(complete),
                  point=estimate([(u['family'], score(u, metric)) for u in complete]),
                  bounds=full_bounds(units, metric))
    if metric == 'probability':
        result['normative_same_sets'] = estimate([(u['family'], score(u, metric, True)) for u in complete])
        result['residual'] = estimate([(u['family'], score(u, metric)-score(u, metric, True)) for u in complete])
        residual_units = [dict(u, offset=0.) for u in units]
        result['residual_bounds'] = full_bounds(residual_units, 'residual')
    return result


def groups_by(records, fields):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r['item'].get(f) for f in fields)].append(r)
    return groups


def paired_units(records, group_fields, cell_fn, coefficients, direction_fn=lambda _: 1):
    units = []
    for _, rows in groups_by(records, group_fields).items():
        cells = {}
        for r in rows:
            label = cell_fn(r['item'])
            if label not in coefficients:
                continue
            if label in cells:
                raise ValueError('Duplicate design cell '+str(label))
            cells[label] = r
        if set(cells) != set(coefficients):
            raise ValueError('Manifest lacks planned contrast cells: '+str(set(coefficients)-set(cells)))
        first = next(iter(cells.values()))['item']
        d = direction_fn(first)
        units.append(unit([(cells[k], d*v) for k, v in coefficients.items()], first['family_id']))
    return units


def strata(records, fields, include_pooled_template=True):
    for values, rows in sorted(groups_by(records, fields).items(), key=lambda z: str(z[0])):
        label = dict(zip(fields, values))
        selections = [(label, rows)]
        if rows and all(r['item']['protocol'] == 'independent' and
                        r['item']['visibility'] == 'signal_visible' for r in rows):
            selections += [(dict(label, relation=relation), [r for r in rows if
                            ('agree' if r['item']['signal'] == r['item']['answer'] else 'conflict') == relation])
                           for relation in ('agree', 'conflict')]
        for selected_label, selected in selections:
            yield selected_label, selected
            if include_pooled_template:
                for (template,), subset in sorted(groups_by(selected, ['template_id']).items()):
                    yield dict(selected_label, template_id=template), subset


MATCH = ['world_id', 'template_id', 'signal', 'answer', 'protocol', 'visibility']


def attribution_analyses(records):
    rows = [r for r in records if r['item'].get('attribution') in ('self', 'other')
            and r['item'].get('answer') in (0, 1)]
    output = []
    for label, subset in strata(rows, ['protocol', 'visibility', 'reminder', 'output_instruction']):
        units = paired_units(subset, MATCH, lambda x: x['attribution'], {'self': 1, 'other': -1},
                             lambda x: 2*x['answer']-1)
        result = dict(stratum=label, gap=summarize_units(units))
        for index, name in ((0, 'self'), (1, 'other')):
            # Coefficient for other in the gap is negative; turn it back to +d.
            sign = 1 if index == 0 else -1
            one = [unit([(u['terms'][index][0], sign*u['terms'][index][1])], u['family'],
                        required=[r for r, _ in u['terms']]) for u in units]
            result[name+'_directional_error'] = summarize_units(one, 'residual')
        complete = [u for u in units if all(valid(r) for r, _ in u['terms'])]
        signs, tolerance = Counter(), Counter()
        for u in complete:
            rs, ro = u['terms'][0][0], u['terms'][1][0]
            d = u['terms'][0][1]
            es, eo = d*(rs['probability']-rs['item']['oracle']), d*(ro['probability']-ro['item']['oracle'])
            signs[{2: 'both_negative', 1: 'one_negative', 0: 'both_nonnegative'}[(es < 0)+(eo < 0)]] += 1
            tolerance['both_near_norm' if abs(es) <= .02 and abs(eo) <= .02 else
                      'self_near_other_below' if abs(es) <= .02 and eo < -.02 else 'other_pattern'] += 1
        result.update(pair_sign_counts=dict(signs), descriptive_tolerance=.02,
                      descriptive_category_counts=dict(tolerance))
        output.append(result)
    return output


def information_analyses(records):
    output = []
    hidden = [r for r in records if r['item']['visibility'] == 'signal_hidden']
    for label, subset in strata(hidden, ['protocol', 'attribution']):
        units = []
        for r in subset:
            x = r['item']; d = 2*x['answer']-1
            units.append(unit([(r, d)], x['family_id'], offset=-d*x['prior']))
        output.append(dict(kind='signal_hidden_from_prior', stratum=label, result=summarize_units(units)))
    visible = [r for r in records if r['item']['visibility'] == 'signal_visible'
               and r['item']['reminder'] == 'scoped' and r['item']['output_instruction'] == 'compatible']
    baselines = {}
    bfields = ['world_id', 'template_id', 'signal', 'protocol', 'attribution', 'reminder', 'output_instruction']
    for r in records:
        if r['item']['visibility'] == 'answer_hidden':
            k = tuple(r['item'][f] for f in bfields)
            if k in baselines:
                raise ValueError('Duplicate hidden baseline')
            baselines[k] = r
    for label, subset in strata(visible, ['protocol', 'attribution']):
        units = []
        for r in subset:
            x = r['item']; b = baselines[tuple(x[f] for f in bfields)]; d = 2*x['answer']-1
            units.append(unit([(r, d), (b, -d)], x['family_id']))
        output.append(dict(kind='answer_visibility_increment', stratum=label, result=summarize_units(units)))
    return output


def cross_factorial(records):
    rows = [r for r in records if r['item']['protocol'] == 'independent'
            and r['item']['visibility'] == 'signal_visible' and r['item']['attribution'] in ('self', 'other')]
    cells = [(rem, wording, attr) for rem in ('scoped', 'legacy')
             for wording in ('compatible', 'restrictive') for attr in ('self', 'other')]
    output = []
    for label, subset in strata(rows, ['protocol']):
        mae = {k: (1 if k[1] == 'compatible' else -1)/4 for k in cells}
        gap = {k: (1 if k[1] == 'compatible' else -1)*(1 if k[2] == 'self' else -1)/2 for k in cells}
        fn = lambda x: (x['reminder'], x['output_instruction'], x['attribution'])
        output.append(dict(stratum=label,
            instruction_mae=summarize_units(paired_units(subset, MATCH, fn, mae), 'absolute_error'),
            instruction_attribution=summarize_units(paired_units(subset, MATCH, fn, gap,
                                                    lambda x: 2*x['answer']-1))))
    return output


def native_analyses(records):
    import itertools
    cells = list(itertools.product(('on', 'off'), ('compatible', 'restrictive'), (4096, 12288), ('self', 'other')))
    specifications = {
        'native_mode_mae': (lambda k: (1 if k[0] == 'on' else -1)/8, 'absolute_error', False),
        'wording_mae': (lambda k: (1 if k[1] == 'compatible' else -1)/8, 'absolute_error', False),
        'budget_mae': (lambda k: (1 if k[2] == 12288 else -1)/8, 'absolute_error', False),
        'native_mode_attribution': (lambda k: (1 if k[0] == 'on' else -1)*(1 if k[3] == 'self' else -1)/4, 'probability', True),
        'wording_attribution': (lambda k: (1 if k[1] == 'compatible' else -1)*(1 if k[3] == 'self' else -1)/4, 'probability', True),
        'budget_attribution': (lambda k: (1 if k[2] == 12288 else -1)*(1 if k[3] == 'self' else -1)/4, 'probability', True),
        'mode_wording_mae': (lambda k: (1 if k[0] == 'on' else -1)*(1 if k[1] == 'compatible' else -1)/4, 'absolute_error', False),
        'mode_budget_mae': (lambda k: (1 if k[0] == 'on' else -1)*(1 if k[2] == 12288 else -1)/4, 'absolute_error', False),
        'wording_budget_mae': (lambda k: (1 if k[1] == 'compatible' else -1)*(1 if k[2] == 12288 else -1)/4, 'absolute_error', False),
        'mode_wording_budget_mae': (lambda k: (1 if k[0] == 'on' else -1)*(1 if k[1] == 'compatible' else -1)*(1 if k[2] == 12288 else -1)/2, 'absolute_error', False),
    }
    output = []
    for label, subset in strata(records, ['protocol']):
        result = dict(stratum=label, contrasts={})
        for name, (coef, metric, directional) in specifications.items():
            units = paired_units(subset, MATCH,
                lambda x: (x['native_mode'], x['output_instruction'], x['budget'], x['attribution']),
                {k: coef(k) for k in cells}, lambda x: 2*x['answer']-1 if directional else 1)
            result['contrasts'][name] = summarize_units(units, metric)
        output.append(result)
    return output


def history_analyses(records):
    output = dict(presentation=[], information=[])
    match = ['parent_item_id', 'parent_model', 'evidence_kind', 'next_signal']
    for label, subset in strata(records, ['evidence_kind']):
        result = dict(stratum=label, contrasts={})
        for name, coeff in (
            ('history', {'native_self': 1, 'replay_self': -1}),
            ('replay_attribution', {'replay_self': 1, 'replay_other': -1}),
            ('replay_self_neutral', {'replay_self': 1, 'replay_neutral': -1}),
            ('replay_other_neutral', {'replay_other': 1, 'replay_neutral': -1})):
            selected = [r for r in subset if r['item']['history_arm'] in coeff]
            units = paired_units(selected, match, lambda x: x['history_arm'], coeff,
                                 lambda x: 2*x['initial_answer']-1)
            result['contrasts'][name+'_initial_choice_direction'] = summarize_units(units)
            units = paired_units(selected, match, lambda x: x['history_arm'], coeff)
            result['contrasts'][name+'_mae'] = summarize_units(units, 'absolute_error')
        output['presentation'].append(result)
    baseline = {(r['item']['parent_item_id'], r['item']['history_arm']): r for r in records
                if r['item']['evidence_kind'] == 'none'}
    new_info = [r for r in records if r['item']['evidence_kind'] != 'none']
    for label, subset in strata(new_info, ['evidence_kind', 'history_arm']):
        units = []
        for r in subset:
            x = r['item']; d = 2*x['next_signal']-1
            b = baseline[x['parent_item_id'], x['history_arm']]
            units.append(unit([(r, d), (b, -d)], x['family_id']))
        output['information'].append(dict(stratum=label, update=summarize_units(units)))
    output['estimand'] = 'Fixed cohort of valid producer records; parent-generation coverage reported separately'
    return output


def analyze(manifest_path, db_path, models):
    manifest = read_jsonl(manifest_path)
    output = dict(schema_version=1, analysis_kind='i16-extension-v1',
                  manifest_sha256=file_hash(manifest_path), database_sha256=file_hash(db_path),
                  uncertainty='Fixed finite family means and deterministic missingness bounds; no sampling intervals',
                  models=[])
    for model in models:
        records = snapshot(manifest, db_path, model)
        module_set = {r['item']['module'] for r in records}
        if len(module_set) != 1:
            raise ValueError('Analyze one experiment module at a time')
        module = next(iter(module_set))
        result = dict(model=model, module=module, overall=describe(records))
        fields = ['protocol', 'visibility', 'attribution', 'reminder', 'output_instruction',
                  'template_id', 'native_mode', 'max_new_tokens', 'history_arm']
        result['conditions'] = [dict(stratum=dict(zip(fields, coords)), **describe(rows))
                                for coords, rows in sorted(groups_by(records, fields).items(), key=lambda z: str(z[0]))]
        if module == 'gate':
            result['qualification'] = analyze_gate(manifest, db_path, model)
        elif module == 'cross_family':
            result.update(attribution=attribution_analyses(records), information=information_analyses(records),
                          factorial=cross_factorial(records))
        elif module == 'native_factorial':
            result['factorial'] = native_analyses(records)
        elif module == 'history_stage2':
            result['history'] = history_analyses(records)
        elif module == 'history_stage1':
            result['derived_choice_rule'] = 'C=1[p_h1>=0.5], including exact ties'
            result['choice_counts'] = dict(Counter(int(r['probability'] >= .5) for r in records if valid(r)))
        else:
            raise ValueError(module)
        output['models'].append(result)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('gate', 'analyze'))
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--db', type=Path, required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    result = ([analyze_gate(args.manifest, args.db, m) for m in args.models] if args.command == 'gate'
              else analyze(args.manifest, args.db, args.models))
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    else:
        print(canonical(result))


if __name__ == '__main__':
    main()
