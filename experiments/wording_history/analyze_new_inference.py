"""Complete uniform R6 and qualified standard Phi-4 outputs, no reparsing."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import analyze_extension as a
import build_extension as b
from analyze_inference import history_results,paraphrase_results
OUT=ROOT/'completed_inference_analysis'
def save(name,value):
    path=OUT/name
    assert not path.exists(),'Immutable completed analysis already exists'
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
    q=ROOT/'vllm-r6';phi=ROOT/'standard-phi/study'
    assert json.loads((q/'STATUS.json').read_text())['state']=='completed' and not (q/'FAILURE.json').exists()
    assert json.loads((phi/'STATUS.json').read_text())['state']=='completed' and not (phi/'FAILURE.json').exists()
    OUT.mkdir(exist_ok=True)
    parents=a.snapshot(b.read_jsonl(ROOT/'inference/manifests/qwen-parents.jsonl'),ROOT/'inference/results/qwen-parents.sqlite','qwen3-8b-r5')
    history=a.snapshot(b.read_jsonl(q/'manifests/history.jsonl'),q/'results/main.sqlite','qwen3-8b-r6')
    para=a.snapshot(b.read_jsonl(q/'manifests/paraphrases.jsonl'),q/'results/main.sqlite','qwen3-8b-r6')
    assert not any(r.get('exclusion') for r in parents+history+para)
    save('qwen_history.json',history_results(parents,history));save('paraphrases.json',paraphrase_results(para))
    gate=json.loads((phi/'results/gate-analysis.json').read_text());save('phi_gate.json',gate)
    if gate['passes']:
        cross=a.snapshot(b.read_jsonl(phi/'manifests/cross.jsonl'),phi/'results/main.sqlite','phi4-standard-r6')
        assert not any(r.get('exclusion') for r in cross)
        save('phi_static.json',dict(overall=a.describe(cross),attribution=a.attribution_analyses(cross),information=a.information_analyses(cross)))
        pp=a.snapshot(b.read_jsonl(phi/'manifests/parents.jsonl'),phi/'results/main.sqlite','phi4-standard-r6')
        hh=a.snapshot(b.read_jsonl(phi/'manifests/history.jsonl'),phi/'results/history.sqlite','phi4-standard-r6')
        assert not any(r.get('exclusion') for r in pp+hh)
        save('phi_history.json',history_results(pp,hh))
    save('closure.json',dict(state='complete',training=False,weight_updates=False,
        qwen_status=json.loads((q/'STATUS.json').read_text()),phi_status=json.loads((phi/'STATUS.json').read_text()),
        r5_long_phi_termination=json.loads((ROOT/'inference/phi_futility/STOP_DECISION.json').read_text()),
        r5_followup_termination=json.loads((ROOT/'inference/qwen_pilot_closure/STOP_DECISION.json').read_text())))
if __name__=='__main__':main()
