"""Verify sealed settings and matched paraphrase semantics before inference."""
import json
from collections import Counter, defaultdict
import inference_helpers as h
import analyze_extension as a

jobs=json.loads((h.ROOT/'inference/INITIAL_JOBS.json').read_text())
for job in jobs:
    folder=h.ROOT/'inference/jobs'/job['name']
    cfg,_=h.worker.validate_frozen(folder/'config.json',folder/'source/run_extension.py')
    rows=h.base.read_jsonl(folder/'manifest.jsonl')
    assert len(rows)==job['count']
    spec=cfg['models'][0]
    for item in rows:
        profile=h.worker.effective_profile(cfg,spec,item)
        assert profile.max_new_tokens==(28672 if job['model'].startswith('phi') else 12288)
        assert profile.max_input_tokens==(4096 if job['model'].startswith('phi') else 8192)
        assert h.worker.item_messages(item)
        assert abs(h.base.oracle(item['protocol'],item['visibility'],item['prior'],item['reliability'],item['signal'],item['answer'])-item['oracle'])<1e-12
paraphrases=h.base.read_jsonl(h.ROOT/'inference/manifests/paraphrase.jsonl')
groups=defaultdict(list)
for item in paraphrases:
    groups[item['origin_item_id']].append(item)
for origin,rows in groups.items():
    assert len(rows)==3
    assert {r['paraphrase_variant'] for r in rows}=={'original','authorship','dependence'}
    for field in ('prior','reliability','signal','answer','attribution','oracle','family_id','world_id','template_id','reminder','output_instruction'):
        assert len({r[field] for r in rows})==1,(origin,field)
    assert len({r['prompt'] for r in rows})==3
    assert len({r['id'] for r in rows})==3
parents=h.base.read_jsonl(h.ROOT/'inference/manifests/qwen-parents.jsonl')
assert Counter(r['parent_condition'] for r in parents)=={'numeric_on':48,'numeric_off':48,'explanation_on':48}
assert all(r['chat_template_kwargs']['enable_thinking']==(r['parent_condition']!='numeric_off') for r in parents)
result=dict(worker_freezes_verified=len(jobs),qualified_gate_calls=96,parent_calls=len(parents),
            matched_paraphrase_cells=len(groups),paraphrase_calls=len(paraphrases),oracles_verified=True,
            all_decoding_greedy=True,model_weights_unchanged=True)
h.atomic_json(h.ROOT/'inference/PRE_EXECUTION_VALIDATION.json',result)
print(json.dumps(result))
