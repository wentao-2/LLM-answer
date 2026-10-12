"""Export completed supplement evidence for the manuscript, without inference."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def csvfile(name,rows):
    with (ROOT/'data/review_supplement'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args();e=args.evidence_root.resolve();aud=load(args.audit)
    assert aud['state']=='passed'
    out=ROOT/'data/review_supplement';out.mkdir(exist_ok=True)
    names=('phi_static','four_wordings','qwen_history','phi_history','phi_gate','qwen_runtime_qualification')
    docs={n:load(e/'completed_inference_analysis'/f'{n}.json') for n in names}
    for n in ('qwen_history','phi_history'):
        docs[n].pop('parent_records');docs[n].pop('item_update_errors')
    save(out/'summaries.json',docs)
    save(out/'provenance.json',dict(analysis_sources={n:sha(e/'completed_inference_analysis'/f'{n}.json') for n in names},
         audit_sha256=sha(args.audit),database_sources=aud['stages'],
         selection='All conditions, fixed quality cohorts, and constructed followups. No outcome selected replacement.',
         primary_wording_comparison='Common complete twelve reports. Bounds cover all planned settings.'))
    sys.path.insert(0,str(e/'source'));import analyze_extension as a;import build_extension as b
    records=a.snapshot(b.read_jsonl(e/'standard-phi/study/manifests/cross.jsonl'),e/'standard-phi/study/results/main.sqlite','phi4-standard-r6')
    rows=[]
    for s in docs['phi_static']['attribution']:
        st=s['stratum']
        if st.get('protocol')!='independent' or st.get('visibility')!='signal_visible' or st.get('relation')!='agree' or 'template_id' not in st: continue
        subset=[r for r in records if all(r['item'].get(k)==v for k,v in st.items() if k!='relation') and r['item']['signal']==r['item']['answer'] and r['item']['attribution'] in ('self','other')]
        pairs=defaultdict(dict)
        for r in subset: pairs[tuple(r['item'][k] for k in a.MATCH)][r['item']['attribution']]=r
        complete=[c for c in pairs.values() if all(a.valid(r) for r in c.values())]
        assert all(len(c)==2 for c in pairs.values())
        mae={g:a.estimate([(c[g]['item']['family_id'],abs(c[g]['probability']-c[g]['item']['oracle'])) for c in complete])['estimate'] for g in ('self','other')}
        rows.append(dict(reminder=st['reminder'],wording=st['output_instruction'],representation=st['template_id'].split('_')[0],complete=len(complete),planned=len(pairs),self_error=s['self_directional_error']['point']['estimate'],other_error=s['other_directional_error']['point']['estimate'],gap=s['gap']['point']['estimate'],self_mae=mae['self'],other_mae=mae['other'],gap_lower=s['gap']['bounds']['lower'],gap_upper=s['gap']['bounds']['upper']))
    assert len(rows)==8;csvfile('phi_static.csv',rows)
    rows=[]
    for s in docs['four_wordings']['strata']:
        for v,metrics in s['versions'].items():
            for metric,x in metrics.items():
                if not isinstance(x,dict):continue
                rows.append(dict(stratum_json=json.dumps(s['stratum'],sort_keys=True),version=v,metric=metric,point=x['point']['estimate'],complete=x['complete_valid_sets'],planned=x['candidate_sets'],lower=x['bounds']['lower'],upper=x['bounds']['upper']))
    csvfile('wordings.csv',rows)
    coverage=[];updates=[];present=[]
    for name in ('qwen_history','phi_history'):
        for c in docs[name]['cohorts']:
            coverage.append(dict(model=name,cohort=c['cohort'],parents=c['n_parents'],families=c['parent_mae']['n_families'],parent_mae=c['parent_mae']['estimate'],children=c['n_planned_children'],valid=sum(z['n_valid'] for z in c.get('arms',[]))))
            for x in c.get('individual_update_errors',[]):
                updates.append(dict(model=name,cohort=c['cohort'],arm=x['history_arm'],evidence=x['evidence_kind'],signed=x['signed']['estimate'],absolute=x['absolute']['estimate'],complete=x['absolute']['n_observations'],families=x['absolute']['n_families'],planned=x['planned_pairs'],max_absolute=x['max_absolute']))
            for s in c.get('analysis',{}).get('presentation',[]):
                for metric,x in s['contrasts'].items():
                    present.append(dict(model=name,cohort=c['cohort'],stratum_json=json.dumps(s['stratum'],sort_keys=True),metric=metric,point=x['point']['estimate'],complete=x['complete_valid_sets'],planned=x['candidate_sets'],lower=x['bounds']['lower'],upper=x['bounds']['upper']))
    csvfile('history_coverage.csv',coverage);csvfile('history_update_errors.csv',updates);csvfile('history_presentation.csv',present)
    print('Exported six audited summaries and five unrounded CSVs.')

if __name__=='__main__':main()
