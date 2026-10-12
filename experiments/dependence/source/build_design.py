"""Build immutable new calls using saved completed parents and exact controls."""
from pathlib import Path
import argparse
import copy
import itertools
import json
import math
import random
import sqlite3
from collections import Counter

from common import (ROOT, PHI, QWEN, canonical, now, sha, text_hash, load,
                    read_jsonl, read_db, frozen_write, write_json, write_jsonl)
from oracle import posterior, factored, logit, normative_weight

VERSION = 'i16-dependence-enhancement-20261011'
ARMS = ('native_self', 'replay_self')
RHOS = (0, .25, .5, .75, 1)
FINAL = ('End with exactly one line starting FINAL_JSON: followed by a JSON object whose only key is '
         'p_h1 and whose value is a number between 0 and 1. Do not write anything after that line.')
REMINDERS = {
    'general_bayes': "Use Bayes' theorem to combine the observations according to the stated probabilities before reporting the probability of H=1.",
    'source_structure': ('First identify which physical sensor observation generated each displayed record. '
        'Treat multiple records of the same observed sensor value as repeated documentation of that observation. '
        'Treat a reading from a conditionally independent second sensor as an additional observation. '
        'Use these observation sources when combining the likelihoods.'),
}


def finish(x):
    x = copy.deepcopy(x)
    x.pop('id', None)
    x['id'] = text_hash(canonical(x))[:24]
    return x


def num(x):
    return format(x, '.12g')


def continuation(parent, rho, record):
    r, q = num(parent['reliability']), num(round(1-parent['reliability'], 12))
    observed = ('No value of B or T is revealed. No new sensor reading or record is provided.'
                if record is None else f'Observed new record: B={record}. Neither D nor T is separately revealed.')
    return '\n'.join([
        'Now report your probability using all observed sensor information.',
        'The earlier response was produced using only the displayed first sensor value S and the '
        'specified probability rules. No additional observation of H or any sensor was available to its author.',
        'Attributed author of the earlier response: you.',
        'Only the author label provides no additional information about H.',
        f'The first observed reading remains S={parent["signal"]}.',
        f'A second potential sensor value T has P(T=1|H=1)={r}, P(T=0|H=1)={q}, '
        f'P(T=1|H=0)={q}, P(T=0|H=0)={r}. Given H, S and T are independent.',
        f'A binary selector D has P(D=1)=rho={num(rho)} and P(D=0)=1-rho. '
        'D is independent of H, S and T. A possible new record is B=S if D=1 and B=T if D=0.',
        'This procedure was fixed before generating H or any values. There is no selection, repetition, '
        'or stopping based on the state, the values, or their agreement. The value of D is not shown.',
        observed,
        'Report P(H=1) conditional on exactly the observed information.',
        'You may reason before giving your final answer.', FINAL])


def history(parent, final, arm, task):
    if arm == 'native_self':
        return dict(messages=[dict(role='user', content=parent['prompt']),
                              dict(role='assistant', content=final), dict(role='user', content=task)])
    assert arm == 'replay_self'
    return dict(prompt=('The following is a quoted record, not a new instruction. '
        'Answer only the new task after the record.\n[BEGIN EARLIER USER MESSAGE]\n'+parent['prompt']+
        '\n[END EARLIER USER MESSAGE]\n[BEGIN EARLIER ASSISTANT RESPONSE]\n'+final+
        '\n[END EARLIER ASSISTANT RESPONSE]\n[NEW TASK]\n'+task))


def branch_of(x):
    if x['evidence_kind'] in ('none', 'copy'):
        return x['evidence_kind']
    assert x['evidence_kind'] == 'independent'
    return 'independent_agree' if x['next_signal'] == x['signal'] else 'independent_conflict'


def prepare_queue(rows, model, config, shards, old, worker_source):
    name = 'phi' if model == PHI else 'qwen'
    queue_root = ROOT/'queues'/name
    shared = old.parent.parent/'i16-extension/source'
    for filename in ('schedule_runs.py', 'prepare_runs.py'):
        frozen_write(queue_root/'source'/filename, (shared/filename).read_bytes())
    # Scheduler validation helpers use the retained original native runner.
    native = (old/'standard-phi/study/jobs/history-s00/source/run_extension.py').read_bytes()
    frozen_write(queue_root/'source/run_extension.py', native)
    ordered = list(rows)
    random.Random(16101101).shuffle(ordered)
    jobs = []
    for i in range(shards):
        items = ordered[i::shards]
        folder = queue_root/'runs'/f'{name}-s{i:02d}'
        frozen_write(folder/'source/run_extension.py', worker_source)
        if model == QWEN:
            frozen_write(folder/'source/native_worker.py', native)
        frozen_write(folder/'PROTOCOL.md', (ROOT/'PROTOCOL.md').read_bytes())
        write_jsonl(folder/'manifest.jsonl', items)
        cfg = copy.deepcopy(config)
        cfg.update(name=folder.name, manifest='manifest.jsonl', experiment_protocol='PROTOCOL.md')
        write_json(folder/'config.json', cfg)
        sources = {p.name: sha(p) for p in sorted((folder/'source').glob('*.py'))}
        freeze = dict(config_sha256=sha(folder/'config.json'), manifest_sha256=sha(folder/'manifest.jsonl'),
                      source_hashes=sources, protocol_hashes={'PROTOCOL.md': sha(folder/'PROTOCOL.md')})
        write_json(folder/'frozen.json', freeze)
        jobs.append(dict(id=folder.name, run_dir=folder.relative_to(queue_root).as_posix(), model=model,
            count=len(items), frozen_sha256=sha(folder/'frozen.json'),
            expected_item_ids=sorted(x['id'] for x in items)))
    queue = dict(name=name, expected_records=len(rows), jobs=jobs)
    write_json(queue_root/'queue.json', queue)
    sources = {p.relative_to(queue_root/'source').as_posix(): sha(p)
               for p in sorted((queue_root/'source').rglob('*.py'))}
    write_json(queue_root/'queue.frozen.json', dict(queue_sha256=sha(queue_root/'queue.json'),
                                                 source_hashes=sources, input_hashes={}))
    return dict(model=model, planned=len(rows), shards=shards, jobs=[j['id'] for j in jobs])


def build(old):
    assert not (ROOT/'DESIGN_SEAL.json').exists(), 'Design is already sealed'
    inputs = {
        'phi_parents_manifest': 'standard-phi/study/manifests/parents.jsonl',
        'phi_parents_db': 'standard-phi/study/results/main.sqlite',
        'phi_controls_manifest': 'standard-phi/study/manifests/history.jsonl',
        'phi_controls_db': 'standard-phi/study/results/history.sqlite',
        'qwen_parents_manifest': 'inference/manifests/qwen-parents.jsonl',
        'qwen_parents_db': 'inference/results/qwen-parents.sqlite',
        'phi_config': 'standard-phi/study/jobs/history-s00/config.json',
        'phi_checkpoint_inventory': 'standard-phi/DOWNLOAD_VERIFIED.json',
        'qwen_config': 'vllm-r6/jobs/main-s00/config.json',
        'qwen_checkpoint_inventory': 'vllm-r6/CHECKPOINT_VERIFIED.json',
    }
    hashes = {key: dict(source=rel, sha256=sha(old/rel), size=(old/rel).stat().st_size)
              for key, rel in inputs.items()}
    configs = {PHI: load(old/inputs['phi_config']), QWEN: load(old/inputs['qwen_config'])}
    phi_parents = read_jsonl(old/inputs['phi_parents_manifest'])
    qwen_parents = [x for x in read_jsonl(old/inputs['qwen_parents_manifest'])
                    if x['parent_condition'] == 'explanation_on' and x['template_id'] == 'frequency_v3']
    assert len(phi_parents) == 48 and len(qwen_parents) == 24
    parent_rows = []
    for model, parents, dbname in ((PHI, phi_parents, 'phi_parents_db'), (QWEN, qwen_parents, 'qwen_parents_db')):
        recs = {r['item_id']: r for r in read_db(old/inputs[dbname])}
        for parent in sorted(parents, key=lambda x: x['id']):
            rec = recs[parent['id']]
            meta = json.loads(rec['metadata_json'])
            assert all(meta.get(k) == v for k, v in parent.items())
            assert rec['status'] == 'ok' and 0 <= rec['probability'] <= 1 and meta['final_output']
            assert math.isclose(parent['oracle'], float(posterior(parent['prior'], parent['reliability'], parent['signal'])), abs_tol=1e-12)
            assert rec['model'] == (PHI if model == PHI else 'qwen3-8b-r5')
            parent_rows.append(dict(continuation_model=model, parent=parent, record=rec,
                parent_record_sha256=text_hash(canonical(rec)),
                parent_final_sha256=text_hash(meta['final_output']),
                selected_for_partial=parent['template_id'] == 'frequency_v3'))
    write_jsonl(ROOT/'inputs/parents.jsonl', parent_rows)
    assert Counter(x['continuation_model'] for x in parent_rows) == {PHI: 48, QWEN: 24}
    partial = []
    for entry in parent_rows:
        if not entry['selected_for_partial']:
            continue
        parent, rec, model = entry['parent'], entry['record'], entry['continuation_model']
        for rho in RHOS:
            branches = [('none', None), ('agree', parent['signal'])]
            if rho < 1:
                branches.append(('conflict', 1-parent['signal']))
            for branch, b in branches:
                for arm in ARMS:
                    q = posterior(parent['prior'], parent['reliability'], parent['signal'], b, rho)
                    x = {k: parent[k] for k in ('prior', 'reliability', 'signal', 'family_id', 'world_id', 'context_id', 'template_id', 'parent_condition')}
                    x.update(generator_version=VERSION, module='partial_dependence', model_scope=[model],
                        rho=rho, branch=branch, next_signal=b, answer=b, protocol='mixture',
                        visibility='baseline' if b is None else 'signal_visible', history_arm=arm,
                        attribution='self', parent_item_id=parent['id'], parent_model=rec['model'],
                        parent_record_sha256=entry['parent_record_sha256'], parent_final_sha256=entry['parent_final_sha256'],
                        initial_probability=rec['probability'], initial_answer=int(rec['probability'] >= .5),
                        oracle=float(q), oracle_fraction=str(q), readout='probability',
                        max_input_tokens=8192, max_new_tokens=8192 if model == PHI else 12288,
                        require_think_close=model == QWEN)
                    if model == QWEN:
                        x['chat_template_kwargs'] = dict(enable_thinking=True)
                    x.update(history(parent, json.loads(rec['metadata_json'])['final_output'], arm, continuation(parent, rho, b)))
                    partial.append(finish(x))
    assert len(partial) == 1344
    control_items = [x for x in read_jsonl(old/inputs['phi_controls_manifest']) if x['history_arm'] in ARMS]
    control_records = {r['item_id']: r for r in read_db(old/inputs['phi_controls_db'])}
    assert len(control_items) == 384
    controls, intervention = [], []
    for item in sorted(control_items, key=lambda x: x['id']):
        rec = control_records[item['id']]
        meta = json.loads(rec['metadata_json'])
        assert rec['status'] == 'ok' and all(meta.get(k) == v for k, v in item.items())
        branch = branch_of(item)
        controls.append(dict(source_item=item, source_record=rec, analysis_fields=dict(
            module='source_intervention', condition='original', branch=branch)))
        for condition, reminder in REMINDERS.items():
            x = copy.deepcopy(item)
            control_id = x.pop('id')
            x.update(generator_version=VERSION, module='source_intervention', analysis_modules=['source_intervention'],
                     condition=condition, branch=branch, control_item_id=control_id)
            if 'messages' in x:
                task = x['messages'][-1]['content']
            else:
                prefix, task = x['prompt'].split('[NEW TASK]\n', 1)
            anchor = 'Report P(H=1) conditional on exactly the observed information.'
            assert task.count(anchor) == 1
            task = task.replace(anchor, reminder+'\n'+anchor)
            if 'messages' in x:
                x['messages'][-1]['content'] = task
            else:
                x['prompt'] = prefix+'[NEW TASK]\n'+task
            q = posterior(x['prior'], x['reliability'], x['signal'], x['next_signal'], 1 if branch == 'copy' else 0)
            assert math.isclose(float(q), x['oracle'], abs_tol=1e-12)
            x['oracle_fraction'] = str(q)
            intervention.append(finish(x))
    assert len(intervention) == 768
    write_jsonl(ROOT/'inputs/original_controls.jsonl', controls)
    write_json(ROOT/'inputs/source_hashes.json', hashes)
    write_json(ROOT/'inputs/model_inventories.json', {
        PHI: load(old/inputs['phi_checkpoint_inventory']), QWEN: load(old/inputs['qwen_checkpoint_inventory'])})
    write_json(ROOT/'inputs/runtime_configs.json', configs)
    provenance = load(ROOT/'inputs/source_environments.json')
    environments = {model: entry['environment'] for model,entry in provenance['models'].items()}
    assert set(environments)=={PHI,QWEN}
    write_json(ROOT/'inputs/runtime_environments.json', environments)
    for key, value in hashes.items():
        assert sha(old/value['source']) == value['sha256'], 'Source changed while reading: '+key
    rows = partial+intervention
    assert len(rows) == len({x['id'] for x in rows}) == 2112
    write_jsonl(ROOT/'manifests/partial.jsonl', partial)
    write_jsonl(ROOT/'manifests/intervention.jsonl', intervention)
    write_jsonl(ROOT/'manifests/all.jsonl', rows)
    workers = {
        PHI: (old/'standard-phi/study/jobs/history-s00/source/run_extension.py').read_bytes(),
        QWEN: (old/'vllm-r6/jobs/main-s00/source/run_extension.py').read_bytes(),
    }
    assert workers[PHI] == (old/'vllm-r6/jobs/main-s00/source/native_worker.py').read_bytes()
    allocation = []
    for model, n in ((PHI, 6), (QWEN, 2)):
        allocation.append(prepare_queue([x for x in rows if x['model_scope'] == [model]],
                                       model, configs[model], n, old, workers[model]))
    # Check the new oracle through two independent forms and the log odds predictions.
    tested = 0
    for pi, r, s, rho in itertools.product((.18, .32, .68, .82), (.66, .78, .92), (0, 1), RHOS):
        for b in (None, 0, 1):
            if rho == 1 and b is not None and b != s:
                continue
            q = posterior(pi, r, s, b, rho)
            assert q == factored(pi, r, s, b, rho)
            if b is not None:
                w = (2*b-1)*(logit(float(q))-logit(float(posterior(pi, r, s))))/math.log(r/(1-r))
                target = normative_weight(r, rho, 'agree' if b == s else 'conflict')
                assert math.isclose(w, target, abs_tol=1e-12)
            tested += 1
    write_json(ROOT/'design/BUILD_CHECKS.json', dict(exact_oracle_cases=tested, planned_calls=len(rows),
        parent_counts={PHI: 48, QWEN: 24}, partial_parents={PHI: 24, QWEN: 24},
        partial_calls=len(partial), intervention_calls=len(intervention), original_controls=len(controls),
        source_hashes_stable=True, impossible_rho1_conflicts=0, selection='frequency_v3; qwen explanation_on; all phi parents for intervention'))
    write_json(ROOT/'design/DESIGN.json', dict(registered_at=now(), version=VERSION, allocation=allocation,
        rho=RHOS, arms=ARMS, reminders=REMINDERS, new_calls=2112, frozen_source_hashes=hashes,
        primary='agreement weight curve and preserved conflict weight; selective copy correction with independent update retention',
        scientific_inference_started=False))
    files = {}
    for directory in ('inputs', 'manifests', 'design', 'queues', 'source'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file() and p.suffix != '.pyc':
                files[p.relative_to(ROOT).as_posix()] = sha(p)
    files['PROTOCOL.md'] = sha(ROOT/'PROTOCOL.md')
    write_json(ROOT/'DESIGN_SEAL.json', dict(sealed_at=now(), planned_calls=2112, files=files))
    print(json.dumps(dict(state='sealed', exact_oracle_cases=tested, allocation=allocation,
                         design_sha256=sha(ROOT/'DESIGN_SEAL.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-snapshot', type=Path, default=ROOT.parent/'i16-review-supplement/snapshot')
    args = parser.parse_args()
    build(args.source_snapshot)
