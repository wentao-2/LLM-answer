"""A dated pre-execution amendment; preserve the original sealed design."""
import copy
import json
from collections import defaultdict
import inference_helpers as h

assert not (h.ROOT/'inference/STATUS.json').exists()
old=json.loads((h.ROOT/'inference/DESIGN_SEAL.json').read_text())
assert not old['scientific_inference_started']
rows=[]
for original in h.base.read_jsonl(h.ROOT/'inference/manifests/paraphrase.jsonl'):
    if original['attribution']!='self':
        continue
    item=copy.deepcopy(original)
    item['source_r3_id']=original['origin_item_id']
    item['id']=original['origin_item_id']+':neutral-extension'
    item['attribution']='neutral'
    item['label_extension']=True
    if item['paraphrase_variant']=='authorship':
        old_line='The earlier answer is attributed to the current assistant (you).'
        new_line='The earlier answer is attributed to an undisclosed author.'
    else:
        old_line='Attributed author of the earlier answer: you.'
        new_line='Attributed author of the earlier answer: undisclosed.'
    assert item['prompt'].count(old_line)==1
    item['prompt']=item['prompt'].replace(old_line,new_line)
    rows.append(h.seal_item(item,'qwen3-8b-r5'))
assert len(rows)==576
h.write_manifest(h.ROOT/'inference/manifests/paraphrase-neutral.jsonl',rows)
combined=rows+h.base.read_jsonl(h.ROOT/'inference/manifests/paraphrase.jsonl')
groups=defaultdict(list)
for r in combined:
    key=tuple(r[f] for f in ('world_id','template_id','signal','answer','reminder','output_instruction'))
    groups[key].append(r)
assert len(groups)==192
for records in groups.values():
    assert len(records)==9
    assert {(r['attribution'],r['paraphrase_variant']) for r in records}=={(a,v) for a in ('self','other','neutral') for v in ('original','authorship','dependence')}
    assert len({r['oracle'] for r in records})==1
old.update(helper_sha256=h.digest(h.ROOT/'inference_helpers.py'),controller_sha256=h.digest(h.ROOT/'orchestrate_inference.py'),
    amendment_sha256=h.digest(h.ROOT/'PRE_EXECUTION_AMENDMENT.md'),
    manifests={p.name:h.digest(p) for p in (h.ROOT/'inference/manifests').glob('*.jsonl')})
h.atomic_json(h.ROOT/'inference/DESIGN_SEAL.json',old)
h.atomic_json(h.ROOT/'inference/NEUTRAL_EXTENSION_VALIDATION.json',dict(added_calls=len(rows),total_paraphrase_calls=len(combined),
    planned_nine_report_units=len(groups),matching_numeric_worlds=True,original_self_other_manifest_unchanged=True,scientific_inference_started=False))
print(json.dumps(dict(added_neutral_calls=len(rows),paraphrase_total=len(combined),common_nine_report_sets=len(groups))))
