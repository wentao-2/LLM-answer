from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'source'))
import analyze_extension as a
import build_extension as b
from analyze_inference import history_results,paraphrase_results
folder=ROOT/'vllm-r6';assert json.loads((folder/'STATUS.json').read_text())['state']=='completed'
db=folder/'results/main.sqlite';seal=json.loads(db.with_suffix('.seal.json').read_text())
assert hashlib.sha256(db.read_bytes()).hexdigest()==seal['database_sha256']
parents=a.snapshot(b.read_jsonl(ROOT/'inference/manifests/qwen-parents.jsonl'),ROOT/'inference/results/qwen-parents.sqlite','qwen3-8b-r5')
children=a.snapshot(b.read_jsonl(folder/'manifests/history.jsonl'),db,'qwen3-8b-r6')
para=a.snapshot(b.read_jsonl(folder/'manifests/paraphrases.jsonl'),db,'qwen3-8b-r6')
assert len(children)==2272 and len(para)==1728 and not any(r.get('exclusion') for r in parents+children+para)
h=history_results(parents,children);p=paraphrase_results(para)
out=ROOT/'completed_module_analysis';out.mkdir(exist_ok=True)
(out/'qwen_history.json').write_text(json.dumps(h,indent=2)+'\n')
(out/'paraphrases.json').write_text(json.dumps(p,indent=2)+'\n')
for label in ('all','wrong_choice','large_error','overconfident','visible_prefix'):
    c=next(r for r in h['cohorts'] if r['cohort']==label)
    print(json.dumps(dict(cohort=label,n=c['n_parents'],parent_mae=c['parent_mae']['estimate'],
        no_new=[dict(arm=x['history_arm'],valid=x['n_valid'],planned=x['n_planned'],mae=x['mae']['estimate'],
            change=c['no_new_evidence_change_from_initial'].get(x['history_arm'],{}).get('estimate')) for x in c.get('arms',[]) if x['evidence_kind']=='none'],
        update_errors=[dict(arm=x['history_arm'],evidence=x['evidence_kind'],n=x['absolute']['n_observations'],planned=x['planned_pairs'],
            mean_abs=x['absolute']['estimate'],max_abs=x['max_absolute']) for x in c.get('individual_update_errors',[])])))
for x in p['strata']:
    if 'template_id' in x['stratum']:continue
    print(json.dumps(dict(stratum=x['stratum'],complete=x['complete_nine_sets'],planned=x['planned_sets'],
        range=x['observed_gap_range_same_nine_sets'],gaps={v:r['gap']['point']['estimate'] for v,r in x['versions'].items()})))
