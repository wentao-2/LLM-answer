"""Independent complete-stage integrity audit, with exact posterior arithmetic."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
import re
import sqlite3
import sys

from common import (ROOT, PHI, QWEN, canonical, load, now, read_db, read_jsonl,
                    sha, text_hash, validate_seal, write_json)


def exact(item):
    # Independent odds computation, separate from the enumerating execution oracle.
    pi, r = Fraction(str(item['prior'])), Fraction(str(item['reliability']))
    s, b = item['signal'], item['next_signal']
    odds = pi/(1-pi)*(r/(1-r) if s else (1-r)/r)
    rho = Fraction(str(item['rho'])) if item['module'] == 'partial_dependence' else Fraction(int(item['branch'] == 'copy'))
    if b is not None:
        num = rho*int(b == s)+(1-rho)*(r if b else 1-r)
        den = rho*int(b == s)+(1-rho)*(1-r if b else r)
        assert num > 0 and den > 0
        odds *= num/den
    return odds/(1+odds)


def verify_new_records():
    planned = read_jsonl(ROOT/'manifests/all.jsonl')
    records = {}
    evidence = []
    sys.path.insert(0, str(ROOT/'queues/phi/source'))
    import run_extension as original
    for label, model in (('phi', PHI), ('qwen', QWEN)):
        queue_root = ROOT/'queues'/label
        queue = load(queue_root/'queue.json')
        actual = read_db(queue_root/'merged.sqlite')
        assert len(actual) == queue['expected_records']
        merged = {r['item_id']: r for r in actual}
        assert set(merged) == {x['id'] for x in planned if x['model_scope'] == [model]}
        worker_ids = set()
        for job in queue['jobs']:
            folder = queue_root/job['run_dir']
            terminal = load(folder/'terminal.json')
            assert terminal['returncode'] == 0
            cfg, freeze = original.validate_frozen(folder/'config.json', folder/'source/run_extension.py')
            assert sha(folder/'frozen.json') == job['frozen_sha256']
            assert cfg['models'][0]['name'] == model
            shard = read_db(folder/'results.sqlite')
            ids = {r['item_id'] for r in shard}
            assert ids == set(job['expected_item_ids']) and len(shard) == job['count']
            assert not ids.intersection(worker_ids)
            worker_ids.update(ids)
            db = sqlite3.connect((folder/'results.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
            try:
                metadata = {k: json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
            finally:
                db.close()
            for key in ('config_sha256', 'manifest_sha256', 'source_hashes', 'protocol_hashes'):
                assert metadata[key] == freeze[key]
            expected_profile = dict(max_new_tokens=8192 if model == PHI else 12288, max_input_tokens=8192,
                chat_template_kwargs={} if model == PHI else dict(enable_thinking=True), require_think_close=model == QWEN,
                readout='probability', do_sample=False, num_beams=1)
            for rec in shard:
                assert rec == merged[rec['item_id']]
                assert rec['id'] == model+':'+rec['item_id'] and rec['model'] == model
                meta = json.loads(rec['metadata_json'])
                assert meta['generation_profile'] == expected_profile
                assert rec['output_tokens'] == len(meta['generated_token_ids'])
                assert rec['input_tokens'] <= 8192 and rec['output_tokens'] <= expected_profile['max_new_tokens']
                assert meta['rendered_prompt_sha256'] == text_hash(meta['rendered_prompt'])
                eos = metadata['effective_model_readout']['eos_token_ids']
                has_eos = any(v in eos for v in meta['generated_token_ids'])
                assert has_eos == (rec['status'] != 'truncated')
                if rec['status'] == 'ok':
                    parsed = original.parse_final(meta['final_output'])
                    assert parsed and parsed['probability'] == rec['probability']
                    if model == QWEN:
                        assert meta['thinking_segment_closed'] is True and '</think>' in rec['raw_output']
                else:
                    assert rec['status'] in ('invalid', 'truncated', 'error') and rec['probability'] is None
                    if rec['status']=='invalid':
                        assert original.parse_final(meta['final_output']) is None
                provenance = meta['model_provenance']['model_spec']
                assert provenance['revision'] == cfg['models'][0]['revision']
                assert provenance['path'] == cfg['models'][0]['path']
            evidence.append(dict(job=job['id'], count=len(shard), database_sha256=sha(folder/'results.sqlite'),
                terminal_sha256=sha(folder/'terminal.json'), status_counts=dict(Counter(r['status'] for r in shard))))
        assert worker_ids == set(merged)
        records.update({(model, iid): rec for iid, rec in merged.items()})
    parents = {(x['continuation_model'], x['parent']['id']): x for x in read_jsonl(ROOT/'inputs/parents.jsonl')}
    module_counts = Counter()
    for item in planned:
        model = item['model_scope'][0]
        rec = records[(model, item['id'])]
        meta = json.loads(rec['metadata_json'])
        assert all(meta.get(k) == v for k,v in item.items())
        content = dict(item)
        iid = content.pop('id')
        assert iid == text_hash(canonical(content))[:24]
        q = exact(item)
        assert math.isclose(float(q), item['oracle'], abs_tol=1e-12, rel_tol=0)
        assert str(q) == item['oracle_fraction']
        parent = parents[(model, item['parent_item_id'])]
        assert item['parent_record_sha256'] == parent['parent_record_sha256']
        assert item['parent_final_sha256'] == parent['parent_final_sha256']
        final = json.loads(parent['record']['metadata_json'])['final_output']
        if item['history_arm'] == 'native_self':
            assert item['messages'][0]['content'] == parent['parent']['prompt']
            assert item['messages'][1]['content'] == final
        else:
            assert '\n[BEGIN EARLIER ASSISTANT RESPONSE]\n'+final+'\n[END EARLIER ASSISTANT RESPONSE]' in item['prompt']
        module_counts[(model, item['module'])] += 1
    assert module_counts == {(PHI, 'partial_dependence'): 672, (QWEN, 'partial_dependence'): 672, (PHI, 'source_intervention'): 768}
    return planned, records, evidence


def audit():
    validate_seal()
    planned, records, evidence = verify_new_records()
    analysis = load(ROOT/'analysis/RESULTS.json')
    assert analysis['n_new_calls'] == 2112 and analysis['n_original_controls'] == 384
    original = read_jsonl(ROOT/'inputs/original_controls.jsonl')
    for entry in original:
        item = dict(entry['source_item'], **entry['analysis_fields'])
        rec = entry['source_record']
        records[(PHI, item['id'])] = rec
        planned.append(item)
        assert all(json.loads(rec['metadata_json']).get(k) == v for k,v in entry['source_item'].items())
    item_index = {(x['model_scope'][0], x['id']): x for x in planned}
    checks = 0
    for row in analysis['error_rows']:
        key = (row['model'], row['item_id'])
        item, rec = item_index[key], records[key]
        q = float(exact(item))
        assert math.isclose(row['oracle'], q, abs_tol=1e-12)
        if rec['status'] == 'ok':
            err = abs(rec['probability']-q)
            assert math.isclose(row['posterior_absolute_error'], err, abs_tol=1e-12)
            assert row['choice_correct'] == int((rec['probability'] >= .5) == (q >= .5))
        else:
            assert row['posterior_absolute_error'] is None and row['error_lower'] == 0
            assert math.isclose(row['error_upper'], max(q,1-q), abs_tol=1e-12)
        checks += 1
    for row in analysis['pair_rows']:
        model = row['model']
        item, none = item_index[(model,row['item_id'])], item_index[(model,row['none_item_id'])]
        rec, base = records[(model,row['item_id'])], records[(model,row['none_item_id'])]
        assert item['parent_item_id'] == none['parent_item_id'] and item['history_arm'] == none['history_arm']
        assert item.get('rho') == none.get('rho') and item.get('condition') == none.get('condition')
        qb, q0 = float(exact(item)), float(exact(none))
        pb = rec['probability'] if rec['status']=='ok' else None
        p0 = base['probability'] if base['status']=='ok' else None
        dq = qb-q0
        lo = (pb if pb is not None else 0)-(p0 if p0 is not None else 1)-dq
        hi = (pb if pb is not None else 1)-(p0 if p0 is not None else 0)-dq
        abslo = 0 if lo <= 0 <= hi else min(abs(lo),abs(hi))
        assert math.isclose(row['update_residual_lower'], lo, abs_tol=1e-12)
        assert math.isclose(row['update_residual_upper'], hi, abs_tol=1e-12)
        assert math.isclose(row['absolute_update_residual_lower'], abslo, abs_tol=1e-12)
        assert math.isclose(row['absolute_update_residual_upper'], max(abs(lo),abs(hi)), abs_tol=1e-12)
        if pb is not None and p0 is not None:
            assert math.isclose(row['probability_update_residual'], pb-p0-dq, abs_tol=1e-12)
            if 0 < pb < 1 and 0 < p0 < 1:
                w = (2*item['next_signal']-1)*math.log((pb/(1-pb))/(p0/(1-p0)))/math.log(item['reliability']/(1-item['reliability']))
                assert math.isclose(row['effective_weight'],w,abs_tol=1e-10)
            else:
                assert row['effective_weight'] is None and row['logit_state']=='endpoint'
        else:
            assert row['effective_weight'] is None and row['logit_state']=='missing_pair'
        checks += 1
    # Independently reconcile each aggregate with the recorded per-item values.
    for series, rows, fields in ((analysis['summaries'], analysis['error_rows'],
            ('posterior_absolute_error','choice_correct','error_lower','error_upper','accuracy_lower','accuracy_upper')),
        (analysis['pair_summaries'], analysis['pair_rows'],
            ('probability_update_residual','absolute_probability_update_residual','effective_weight',
             'weight_residual','absolute_weight_residual','absolute_weight','update_residual_lower','update_residual_upper',
             'absolute_update_residual_lower','absolute_update_residual_upper'))):
        for report in series:
            group = [x for x in rows if all(x[k]==report[k] for k in ('model','module','history_arm','rho','condition','branch'))]
            assert len(group) == report['n_planned']
            for metric in fields:
                family = defaultdict(list)
                for x in group:
                    if x.get(metric) is not None:
                        family[x['family_id']].append(x[metric])
                values = {k:sum(v)/len(v) for k,v in family.items()}
                got = report[metric]
                assert got['n'] == sum(len(v) for v in family.values()) and got['n_families']==len(values)
                for k,v in values.items():
                    assert math.isclose(got['family_values'][k],v,abs_tol=1e-10)
                if len(values)==6:
                    assert math.isclose(got['equal_family_mean'],sum(values.values())/len(values),abs_tol=1e-10)
                else:
                    assert got['equal_family_mean'] is None
                checks += 1
    report = dict(state='passed', audited_at=now(), n_new_records=2112, n_original_controls=384,
        independent_numerical_checks=checks, design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'),
        analysis_sha256=sha(ROOT/'analysis/RESULTS.json'), jobs=evidence,
        new_status_counts=analysis['new_status_counts'],
        checks=['exact IDs and denominators','frozen sources/configurations','independent rational posterior',
                'saved parent and template identity','strict original FINAL parser and actual EOS',
                'probability errors and log odds without clipping','missing probability bounds','family means'],
        limits=['Local tokenizer decoding was not rerun. Native rendering and full weight hashes were verified by the server preflight.',
                'The fixed factorial grid is descriptive and does not define a random population sample.'])
    write_json(ROOT/'audit/AUDIT.json', report)
    print(json.dumps({k:report[k] for k in ('state','n_new_records','independent_numerical_checks','new_status_counts')}))
    return report


if __name__ == '__main__':
    audit()
