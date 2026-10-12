"""Four semantic wordings on a common twelve-report comparison set."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'source'))
import analyze_extension as a
import build_extension as b
VERSIONS=('original','authorship','dependence','author_independence')
def analyze(records):
    results=[]
    selectors=[]
    for coords,rows in sorted(a.groups_by(records,['reminder','output_instruction']).items()):
        label=dict(zip(('reminder','output_instruction'),coords));selectors.append((label,rows))
        for (template,),subset in sorted(a.groups_by(rows,['template_id']).items()):
            selectors.append((dict(label,template_id=template),subset))
    for label,rows in selectors:
        sets=[]
        for _,subset in a.groups_by(rows,a.MATCH).items():
            cells={(r['item']['paraphrase_variant'],r['item']['attribution']):r for r in subset}
            assert len(cells)==12 and len(subset)==12
            assert set(cells)=={(v,g) for v in VERSIONS for g in ('self','other','neutral')}
            assert len({r['item']['oracle'] for r in subset})==1
            sets.append(cells)
        result=dict(stratum=label,planned_sets=len(sets),
            complete_twelve_sets=sum(all(a.valid(r) for r in cells.values()) for cells in sets),versions={},paired_version_changes={})
        for version in VERSIONS:
            metrics={}
            for metric in ('gap','self_error','other_error','neutral_error','self_mae','other_mae','neutral_mae'):
                units=[]
                for cells in sets:
                    item=cells[(version,'self')]['item'];d=2*item['answer']-1
                    if metric=='gap':
                        terms=[(cells[(version,'self')],d),(cells[(version,'other')],-d)];scale='probability'
                    else:
                        author=metric.split('_')[0];terms=[(cells[(version,author)],1 if metric.endswith('mae') else d)]
                        scale='absolute_error' if metric.endswith('mae') else 'residual'
                    units.append(a.unit(terms,item['family_id'],required=list(cells.values())))
                metrics[metric]=a.summarize_units(units,scale)
            metrics['complete_triads']=sum(all(a.valid(cells[(version,g)]) for g in ('self','other','neutral')) for cells in sets)
            result['versions'][version]=metrics
        for version in VERSIONS[1:]:
            units=[]
            for cells in sets:
                item=cells[('original','self')]['item'];d=2*item['answer']-1
                units.append(a.unit([(cells[(version,'self')],d),(cells[(version,'other')],-d),
                    (cells[('original','self')],-d),(cells[('original','other')],d)],item['family_id'],required=list(cells.values())))
            result['paired_version_changes'][version]=a.summarize_units(units)
        gaps=[result['versions'][v]['gap']['point']['estimate'] for v in VERSIONS]
        result['observed_gap_range_same_twelve_sets']=[min(gaps),max(gaps)] if all(n is not None for n in gaps) else None
        results.append(result)
    return dict(overall=a.describe(records),strata=results,comparison_unit='twelve reports across four wordings and three author labels',versions=VERSIONS)
def main():
    original=ROOT/'vllm-r6';additional=ROOT/'author-independence-r6'
    for study in (original,additional):
        assert json.loads((study/'STATUS.json').read_text())['state']=='completed' and not (study/'FAILURE.json').exists()
    records=a.snapshot(b.read_jsonl(original/'manifests/paraphrases.jsonl'),original/'results/main.sqlite','qwen3-8b-r6')
    records+=a.snapshot(b.read_jsonl(additional/'manifest.jsonl'),additional/'results.sqlite','qwen3-8b-r6')
    assert len(records)==2304 and not any(r.get('exclusion') for r in records)
    output=ROOT/'completed_inference_analysis/four_wordings.json'
    assert not output.exists();output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(analyze(records),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
