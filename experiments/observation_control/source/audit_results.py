"""Independent terminal-record, posterior, pairing and analysis integrity audit."""
from collections import Counter,defaultdict
from fractions import Fraction
import json
import math
import sqlite3
import sys
from statistics import mean

from common import ROOT,PHI,QWEN,canonical,load,now,read_db,read_jsonl,sha,text_hash,validate_seal,write_json
from validate_design import exact,verify_item


def close(a,b):
    assert math.isclose(a,b,abs_tol=1e-12,rel_tol=0),(a,b)


def audit():
    validate_seal()
    sys.path.insert(0,str(ROOT/'worker_sources/phi'))
    import run_extension as native
    planned=read_jsonl(ROOT/'manifests/expression.jsonl')+read_jsonl(ROOT/'manifests/transfer_parents.jsonl')
    records={}
    evidence=[]
    parents={(PHI,x['parent']['id']):x for x in read_jsonl(ROOT/'inputs/parents.jsonl')
             if x['continuation_model']==PHI and x['selected_for_partial']}
    for label,model in (('phi',PHI),('qwen',QWEN)):
        construction=load(ROOT/'cohorts'/label/'CONSTRUCTION.json')
        assert construction['design_seal_sha256']==sha(ROOT/'DESIGN_SEAL.json')
        for rel,digest in construction['files'].items():
            assert sha(ROOT/rel)==digest
        entries=read_jsonl(ROOT/'cohorts'/label/'parents.jsonl')
        children=read_jsonl(ROOT/'cohorts'/label/'children.jsonl')
        coverage=read_jsonl(ROOT/'cohorts'/label/'coverage.jsonl')
        assert len(entries)==len(coverage)==32
        for entry in entries:
            parents[(model,entry['parent']['id'])]=entry
            assert text_hash(canonical(entry['record']))==entry['parent_record_sha256']
            assert text_hash(json.loads(entry['record']['metadata_json'])['final_output'])==entry['parent_final_sha256']
        assert len(children)==32*sum(x['eligible'] for x in coverage)==construction['constructed_children']
        assert Counter(x['parent_item_id'] for x in children)=={x['parent_item_id']:32 for x in coverage if x['eligible']}
        planned+=children
        for label_queue in ('parents_'+label,'followups_'+label):
            qroot=ROOT/'queues'/label_queue
            if not qroot.exists():
                assert construction['allocation'] is None and label_queue=='followups_qwen'
                continue
            assert load(qroot/'scheduler.terminal.json')['returncode']==0
            queue=load(qroot/'queue.json')
            merged={r['item_id']:r for r in read_db(qroot/'merged.sqlite')}
            worker_ids=set()
            for job in queue['jobs']:
                folder=qroot/job['run_dir']
                assert load(folder/'terminal.json')['returncode']==0
                assert sha(folder/'frozen.json')==job['frozen_sha256']
                cfg,freeze=native.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
                chunk=read_db(folder/'results.sqlite')
                ids={r['item_id'] for r in chunk}
                assert ids==set(job['expected_item_ids']) and len(chunk)==job['count'] and not ids&worker_ids
                worker_ids.update(ids)
                db=sqlite3.connect((folder/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
                try:
                    metadata={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
                finally:
                    db.close()
                for k in ('config_sha256','manifest_sha256','source_hashes','protocol_hashes'):
                    assert metadata[k]==freeze[k]
                eos=metadata['effective_model_readout']['eos_token_ids']
                for rec in chunk:
                    assert rec==merged[rec['item_id']]
                    meta=json.loads(rec['metadata_json'])
                    assert rec['id']==model+':'+rec['item_id'] and rec['model']==model
                    assert meta['generation_profile']==dict(max_new_tokens=8192 if model==PHI else 12288,
                        max_input_tokens=8192,chat_template_kwargs={} if model==PHI else dict(enable_thinking=True),
                        require_think_close=model==QWEN,readout='probability',do_sample=False,num_beams=1)
                    assert rec['output_tokens']==len(meta['generated_token_ids'])
                    assert rec['input_tokens']<=8192 and rec['output_tokens']<=(8192 if model==PHI else 12288)
                    assert text_hash(meta['rendered_prompt'])==meta['rendered_prompt_sha256']
                    if rec['status']!='error':
                        assert any(t in eos for t in meta['generated_token_ids'])==(rec['status']!='truncated')
                    parsed=native.parse_final(meta['final_output'])
                    if rec['status']=='ok':
                        assert parsed and parsed['probability']==rec['probability']
                        if model==QWEN:
                            assert meta['thinking_segment_closed'] and '</think>' in rec['raw_output']
                    else:
                        assert rec['status'] in ('invalid','truncated','error') and rec['probability'] is None
                        if rec['status']=='invalid':
                            assert parsed is None
                    model_spec=meta['model_provenance']['model_spec']
                    assert model_spec['revision']==cfg['models'][0]['revision']
                    assert model_spec['path']==cfg['models'][0]['path']
                    key=(model,rec['item_id'])
                    assert key not in records
                    records[key]=rec
                evidence.append(dict(queue=label_queue,job=job['id'],count=len(chunk),
                    database_sha256=sha(folder/'results.sqlite'),terminal_sha256=sha(folder/'terminal.json'),
                    status_counts=dict(Counter(x['status'] for x in chunk))))
            assert worker_ids==set(merged) and len(merged)==queue['expected_records']
    assert len(planned)==len({x['id'] for x in planned})==len(records)
    index={x['id']:x for x in planned}
    assert set(records)=={(x['model_scope'][0],x['id']) for x in planned}
    rendered={}
    for p in (ROOT/'preflight').glob('*-render.json'):
        report=load(p)
        for r in report['ledger']:
            if r['item_id'] in rendered:
                assert rendered[r['item_id']]==r
            rendered[r['item_id']]=r
    for item in planned:
        verify_item(item)
        model=item['model_scope'][0]
        rec=records[(model,item['id'])]
        meta=json.loads(rec['metadata_json'])
        assert all(meta.get(k)==v for k,v in item.items())
        assert meta['rendered_prompt_sha256']==rendered[item['id']]['rendered_prompt_sha256']
        assert rec['input_tokens']==rendered[item['id']]['input_tokens']
        if item['stage']!='transfer_parent':
            p=parents[(model,item['parent_item_id'])]
            assert item['parent_record_sha256']==p['parent_record_sha256']
            assert item['parent_final_sha256']==p['parent_final_sha256']
            assert item['messages'][0]['content']==p['parent']['prompt']
            assert item['messages'][1]['content']==json.loads(p['record']['metadata_json'])['final_output']
    result=load(ROOT/'analysis/RESULTS.json')
    assert result['design_seal_sha256']==sha(ROOT/'DESIGN_SEAL.json')
    assert result['constructed_calls']==len(planned)
    checks=0
    for row in result['error_rows']:
        item=index[row['item_id']]
        rec=records[(row['model'],row['item_id'])]
        q=float(exact(item))
        close(row['oracle'],q)
        if rec['status']=='ok':
            close(row['posterior_error'],abs(rec['probability']-q))
            close(row['error_lower'],row['posterior_error'])
            close(row['error_upper'],row['posterior_error'])
        else:
            assert row['posterior_error'] is None and row['error_lower']==0
            close(row['error_upper'],max(q,1-q))
        checks+=1
    for row in result['pair_rows']:
        item,base=index[row['item_id']],index[row['none_item_id']]
        assert all(item[k]==base[k] for k in ('parent_item_id','rule_expression','condition','rho'))
        assert base['branch']=='none'
        pb,p0=row['probability'],row['none_probability']
        dq=float(exact(item))-float(exact(base))
        close(row['oracle_update'],dq)
        lo=(pb if pb is not None else 0)-(p0 if p0 is not None else 1)-dq
        hi=(pb if pb is not None else 1)-(p0 if p0 is not None else 0)-dq
        close(row['update_lower'],lo)
        close(row['update_upper'],hi)
        close(row['absolute_update_lower'],0 if lo<=0<=hi else min(abs(lo),abs(hi)))
        close(row['absolute_update_upper'],max(abs(lo),abs(hi)))
        if pb is not None and p0 is not None:
            close(row['update_residual'],pb-p0-dq)
            if 0<pb<1 and 0<p0<1:
                ell=math.log(item['reliability']/(1-item['reliability']))
                w=(2*item['next_signal']-1)*(math.log(pb/(1-pb))-math.log(p0/(1-p0)))/ell
                close(row['effective_weight'],w)
            else:
                assert row['effective_weight'] is None and row['logit_state']=='endpoint'
        r,rho=item['reliability'],item['rho']
        norm=1 if item['branch']=='conflict' else math.log((rho+(1-rho)*r)/(rho+(1-rho)*(1-r)))/math.log(r/(1-r))
        close(row['normative_weight'],norm)
        checks+=1
    row_index={x['item_id']:x for x in result['error_rows']}
    row_index.update({x['item_id']:x for x in result['pair_rows']})
    for row in result['contrast_rows']:
        a,b=row_index[row['item_id']],row_index[row['reference_item_id']]
        for key in ('posterior_error','absolute_update_residual','effective_weight','absolute_weight_residual'):
            va,vb=a.get(key),b.get(key)
            if va is not None and vb is not None:
                close(row[key+'_change'],va-vb)
            else:
                assert row[key+'_change'] is None
        close(row['error_change_lower'],a['error_lower']-b['error_upper'])
        close(row['error_change_upper'],a['error_upper']-b['error_lower'])
        checks+=1
    # Recompute equal-family metrics rather than accepting aggregate labels.
    aggregation_checks=0
    for name,rows,extra in (('summaries',result['error_rows'],()),('pair_summaries',result['pair_rows'],()),
                            ('direction_summaries',result['error_rows'],('signal',)),
                            ('direction_pair_summaries',result['pair_rows'],('signal',)),
                            ('numeric_summaries',result['error_rows'],('prior','reliability','signal'))):
        keys=('model','stage','context','rule','condition','rho','branch')+extra
        for summary in result[name]:
            selected=[x for x in rows if all(x.get(k)==summary[k] for k in keys)]
            assert len(selected)==summary['n_planned']
            assert dict(Counter(x['status'] for x in selected))==summary['status_counts']
            for metric,report in summary['metrics'].items():
                groups=defaultdict(list)
                for x in selected:
                    if x.get(metric) is not None:
                        groups[x['family_id']].append(x[metric])
                vals={k:mean(v) for k,v in groups.items()}
                assert set(vals)==set(report['family_values'])
                for k,v in vals.items():
                    close(report['family_values'][k],v)
                expected={x['family_id'] for x in selected}
                if vals and set(vals)==expected:
                    close(report['equal_family_mean'],mean(vals.values()))
                else:
                    assert report['equal_family_mean'] is None
                aggregation_checks+=1
    audit=dict(state='passed',audited_at=now(),design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'),
        analysis_sha256=sha(ROOT/'analysis/RESULTS.json'),constructed_calls=len(planned),
        record_counts=dict(Counter(x['model'] for x in records.values())),item_numeric_checks=checks,
        aggregation_checks=aggregation_checks,workers=evidence,manual_source_coding='pending')
    write_json(ROOT/'audit/AUDIT.json',audit)
    print(canonical(dict(state='passed',constructed_calls=len(planned),checks=checks,aggregations=aggregation_checks)))
    return audit


if __name__=='__main__':
    audit()
