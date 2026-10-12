"""Freeze both scientific stages and construct descendants from complete parents."""
import argparse
import copy
import itertools
import math
from pathlib import Path
import random
from collections import Counter

from common import (ROOT, PHI, QWEN, canonical, now, sha, text_hash, load,
                    read_jsonl, read_db, frozen_write, write_json, write_jsonl, validate_seal)
from oracle import posterior, factored, normative_weight, logit

VERSION = 'i16-mechanism-followup-20261012'
RHOS = (0, .5, 1)
PRIORS = (.18, .32, .68, .82)
NEW_RELIABILITIES = (.72, .86)
CONTEXTS = ('sensor', 'report')
RULES = ('direct', 'selector')
QUESTIONS = ('general', 'explicit')
CONDITIONS = ('ordinary', 'general_bayes', 'explicit', 'full_steps')
FINAL = ('End with exactly one line starting FINAL_JSON: followed by a JSON object whose only key is '
         'p_h1 and whose value is a number between 0 and 1. Do not write anything after that line.')
BAYES = "Use Bayes' theorem to combine the observations according to the stated probabilities before reporting the probability of H=1."
FULL_STEPS = (
    'Before calculating the final probability, perform the following steps.\n'
    '1. List every displayed observed value by its variable name. Keep earlier observations in the list. '
    'Do not treat the earlier assistant answer or explanation as another observation.\n'
    '2. State how each displayed record was generated and which records share an observation source. '
    'Preserve the stated probability of any unrevealed source choice. Equal values from independent '
    'observations still count as two observations. A quotation adds documentation of its source, not '
    'another independent observation.\n'
    '3. For H=0 and H=1 separately, compute the joint likelihood of the complete displayed observations. '
    'Sum over source choices and potential observations whose realized values are not displayed.\n'
    '4. Combine those likelihoods with the stated prior and report the conditional probability.\n'
    'Before the final answer, write one line starting WORK_JSON: followed by a JSON object with keys '
    'observed (an object mapping each observed variable to its displayed value), sources (an object '
    'mapping each displayed record name to a short source description), not_revealed_source_variables '
    '(an array of source variable names whose realized values were not displayed), likelihood_h0 '
    '(a number), and likelihood_h1 (a number). These likelihoods must cover the complete displayed '
    'observations, not include the prior. Then give the required FINAL_JSON line.')


def num(x):
    return format(x, '.12g')


def finish(x):
    x = copy.deepcopy(x)
    x.pop('id', None)
    x['id'] = text_hash(canonical(x))[:24]
    return x


def legal_branches(signal):
    for rho in RHOS:
        yield rho, 'none', None
        yield rho, 'agree', signal
        if rho < 1:
            yield rho, 'conflict', 1-signal


def profile(model):
    x = dict(model_scope=[model], readout='probability', max_input_tokens=8192,
             max_new_tokens=8192 if model == PHI else 12288, require_think_close=model == QWEN)
    if model == QWEN:
        x['chat_template_kwargs'] = dict(enable_thinking=True)
    return x


def rule_text(parent, rho, rule, context):
    r, q = num(parent['reliability']), num(round(1-parent['reliability'], 12))
    kind = 'sensor value' if context == 'sensor' else 'physical observation underlying a report'
    text = [f'A second potential {kind} T has P(T=1|H=1)={r}, P(T=0|H=1)={q}, '
            f'P(T=1|H=0)={q}, P(T=0|H=0)={r}. Given H, S and T are independent.']
    if rule == 'selector':
        text += [f'A binary selector D has P(D=1)=rho={num(rho)} and P(D=0)=1-rho. '
                 'D is independent of H, S and T. A possible new record is B=S if D=1 and B=T if D=0.',
                 'The value of D is not shown.']
    elif rule == 'direct':
        if rho == 0:
            text += ['The new record B always reports T, a second physical observation independent '
                     'of S conditional on H. B is never copied from S. The copying probability is rho=0.']
        elif rho == 1:
            text += ['The new record B always copies the already observed value S. No second '
                     'physical observation is reported by B. The copying probability is rho=1.']
        else:
            text += [f'With probability rho={num(rho)}, the new record B copies the already observed S. '
                     f'With probability {num(1-rho)}, B reports T from a second physical observation '
                     'independent of S conditional on H. This random origin choice is independent of '
                     'H, S and T. Its realized outcome is not shown.']
    else:
        raise ValueError(rule)
    text += ['This procedure was fixed before generating H or any values. There is no selection, '
             'repetition, or stopping based on the state, the values, or their agreement.']
    if context == 'report':
        text += ['S names the first report of its physical observation. B names the second report. '
                 'Whenever B is displayed, an additional record C quotes B exactly without error. '
                 'C has no observation source beyond B.']
    return '\n'.join(text)


def continuation(parent, rho, b, rule, condition):
    context = parent['context_id']
    observed = ('No value of B or T is revealed. No new sensor reading or record is provided.'
                if b is None else f'Observed new record: B={b}. Neither D nor T is separately revealed.')
    if context == 'report':
        observed = ('No value of B, T or C is revealed. No second report or quotation is provided.'
                    if b is None else f'Observed second report: B={b}. Observed quotation: C={b}, '
                    'which quotes B exactly. Neither D nor T is separately revealed.')
    if rule == 'direct' and b is not None:
        observed = observed.replace('Neither D nor T is separately revealed.',
                                    'The origin choice and T are not separately revealed.')
    target = ('P(H=1|S)' if b is None else
              ('P(H=1|S,B,C)' if context == 'report' else 'P(H=1|S,B)'))
    question = 'Report P(H=1) conditional on exactly the observed information.'
    if condition == 'general_bayes':
        question = BAYES+'\n'+question
    elif condition == 'explicit':
        question = f'Report {target}, conditional on all and exactly the displayed observed variables.'
    elif condition == 'full_steps':
        question = FULL_STEPS+'\n'+f'Report {target}, conditional on all and exactly the displayed observed variables.'
    else:
        assert condition in ('general', 'ordinary')
    return '\n'.join([
        'Now report your probability using all observed information.',
        'The earlier response was produced using only the displayed first observation S and the '
        'specified probability rules. No additional observation of H or any other physical observation '
        'was available to its author.',
        'Attributed author of the earlier response: you.',
        'Only the author label provides no additional information about H.',
        f'The first observed value remains S={parent["signal"]}.',
        rule_text(parent, rho, rule, context), observed, question,
        'You may reason before giving your final answer.', FINAL])


def child(entry, stage, rho, branch, b, rule, condition):
    parent, rec, model = entry['parent'], entry['record'], entry['continuation_model']
    q = posterior(parent['prior'], parent['reliability'], parent['signal'], b, rho)
    x = {k: parent[k] for k in ('prior', 'reliability', 'signal', 'family_id', 'context_id')}
    x.update(profile(model), generator_version=VERSION, module=stage, stage=stage,
             rule_expression=rule, condition=condition, rho=rho, branch=branch, next_signal=b,
             answer=b, protocol='mixture', attribution='self', history_arm='native_self',
             parent_item_id=parent['id'], parent_model=rec['model'],
             parent_record_sha256=entry['parent_record_sha256'],
             parent_final_sha256=entry['parent_final_sha256'], initial_probability=rec['probability'],
             oracle=float(q), oracle_fraction=str(q),
             messages=[dict(role='user', content=parent['prompt']),
                       dict(role='assistant', content=__import__('json').loads(rec['metadata_json'])['final_output']),
                       dict(role='user', content=continuation(parent, rho, b, rule, condition))])
    return finish(x)


def new_parent(model, context, r, pi, s):
    first = ('A sensor produces the first physical observation S.' if context == 'sensor' else
             'A first report, named S, gives the value of a physical observation without reporting error.')
    prompt = '\n'.join([
        f'A binary state is named H. Before observations, its probability of being 1 is {num(pi)}. '+first,
        f'Conditional on H=1, S=1 occurs in {num(100*r)} out of 100 trials and S=0 in '
        f'{num(round(100*(1-r), 8))} out of 100. Conditional on H=0, S=1 occurs in '
        f'{num(round(100*(1-r), 8))} out of 100 trials and S=0 in {num(100*r)} out of 100. '
        'These frequencies specify exact conditional probabilities, not additional observed trials.',
        'No second physical observation, second report or information about the hidden state is available.',
        f'Observed: S={s}.', 'Report P(H=1) conditional on exactly the observed information.',
        'You may reason before giving your final answer.', FINAL])
    family = text_hash(canonical(dict(r=r, complementary_prior_pair=sorted((round(pi, 2), round(1-pi, 2))))))[:24]
    q = posterior(pi, r, s)
    return finish(dict(profile(model), generator_version=VERSION, module='transfer_parent',
        stage='transfer_parent', context_id=context, prior=pi, reliability=r, signal=s,
        family_id=family, template_id='frequency_followup', branch='parent', next_signal=None,
        oracle=float(q), oracle_fraction=str(q), prompt=prompt))


def prepare_queue(rows, model, label, shards):
    if not rows:
        return None
    qroot = ROOT/'queues'/label
    for p in sorted((ROOT/'worker_sources/scheduler').glob('*.py')):
        frozen_write(qroot/'source'/p.name, p.read_bytes())
    ordered = list(rows)
    random.Random(16101201).shuffle(ordered)
    jobs = []
    cfg0 = load(ROOT/'inputs/runtime_configs.json')[model]
    worker_label = 'phi' if model == PHI else 'qwen'
    for i in range(min(shards, len(rows))):
        items = ordered[i::shards]
        folder = qroot/'runs'/f'{label}-s{i:02d}'
        for p in sorted((ROOT/'worker_sources'/worker_label).glob('*.py')):
            frozen_write(folder/'source'/p.name, p.read_bytes())
        frozen_write(folder/'PROTOCOL.md', (ROOT/'PROTOCOL.md').read_bytes())
        write_jsonl(folder/'manifest.jsonl', items)
        cfg = copy.deepcopy(cfg0)
        cfg.update(name=folder.name, manifest='manifest.jsonl', experiment_protocol='PROTOCOL.md')
        write_json(folder/'config.json', cfg)
        write_json(folder/'frozen.json', dict(config_sha256=sha(folder/'config.json'),
            manifest_sha256=sha(folder/'manifest.jsonl'),
            source_hashes={p.name:sha(p) for p in sorted((folder/'source').glob('*.py'))},
            protocol_hashes={'PROTOCOL.md':sha(folder/'PROTOCOL.md')}))
        jobs.append(dict(id=folder.name, run_dir=folder.relative_to(qroot).as_posix(), model=model,
            count=len(items), expected_item_ids=sorted(x['id'] for x in items),
            frozen_sha256=sha(folder/'frozen.json')))
    write_json(qroot/'queue.json', dict(name=label, expected_records=len(rows), jobs=jobs))
    write_json(qroot/'queue.frozen.json', dict(queue_sha256=sha(qroot/'queue.json'),
        source_hashes={p.name:sha(p) for p in sorted((qroot/'source').glob('*.py'))}, input_hashes={}))
    return dict(label=label, model=model, calls=len(rows), shards=len(jobs))


def build():
    assert not (ROOT/'DESIGN_SEAL.json').exists(), 'Already frozen'
    old = ROOT.parent/'i16-dependence-enhancement'
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('old_common', old/'source/common.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_seal(old)
    selected = [x for x in read_jsonl(ROOT/'inputs/parents.jsonl')
                if x['continuation_model'] == PHI and x['selected_for_partial']]
    assert len(selected) == 24
    first = [child(e, 'expression', rho, branch, b, rule, question)
             for e in selected for rho,branch,b in legal_branches(e['parent']['signal'])
             for rule in RULES for question in QUESTIONS]
    parents = [new_parent(model, context, r, pi, s)
               for model,context,r,pi,s in itertools.product((PHI,QWEN), CONTEXTS,
                                                            NEW_RELIABILITIES, PRIORS, (0,1))]
    assert len(first) == 768 and len(parents) == 64
    assert len({x['id'] for x in first+parents}) == 832
    write_jsonl(ROOT/'manifests/expression.jsonl', first)
    write_jsonl(ROOT/'manifests/transfer_parents.jsonl', parents)
    write_jsonl(ROOT/'manifests/preflight_static.jsonl', first+parents)
    allocations = [prepare_queue([x for x in parents if x['model_scope']==[m]],m,'parents_'+label,n)
                   for m,label,n in ((PHI,'phi',6),(QWEN,'qwen',2))]
    cases = 0
    for pi,r,s in itertools.product(PRIORS, (.66,.78,.92)+NEW_RELIABILITIES, (0,1)):
        for rho,branch,b in legal_branches(s):
            q = posterior(pi,r,s,b,rho)
            assert q == factored(pi,r,s,b,rho)
            if b is not None:
                w = (2*b-1)*(logit(float(q))-logit(float(posterior(pi,r,s))))/math.log(r/(1-r))
                assert math.isclose(w,normative_weight(r,rho,branch),abs_tol=1e-12)
            cases += 1
    write_json(ROOT/'design/DESIGN.json', dict(version=VERSION, frozen_at=now(),
        stage1_calls=768, stage2_parents=64, stage2_children_max=2048, max_calls=2880,
        rhos=RHOS, rules=RULES, questions=QUESTIONS, conditions=CONDITIONS,
        priors=PRIORS, transfer_reliabilities=NEW_RELIABILITIES, contexts=CONTEXTS,
        old_design_sha256=sha(old/'DESIGN_SEAL.json'), exact_oracle_cases=cases,
        allocations=allocations, complete_children_per_valid_parent=32,
        selection='all status=ok parents with finite p in [0,1], irrespective of accuracy',
        stage2_adaptation=False, final_instruction=FINAL, full_steps=FULL_STEPS, general_bayes=BAYES))
    from validate_design import validate
    validate(save=False)
    # Queues and dynamic construction use these exact sources. Administrative
    # preflight records and generated children are separately sealed at runtime.
    files = [ROOT/'PROTOCOL.md', ROOT/'design/DESIGN.json']
    for folder in ('inputs','source','worker_sources','manifests','queues'):
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files = sorted(set(files))
    write_json(ROOT/'DESIGN_SEAL.json', dict(frozen_at=now(), version=VERSION,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print(canonical(dict(state='frozen',max_calls=2880,seal_sha256=sha(ROOT/'DESIGN_SEAL.json'))))


def construct(label):
    validate_seal()
    model = PHI if label=='phi' else QWEN
    parent_items = [x for x in read_jsonl(ROOT/'manifests/transfer_parents.jsonl') if x['model_scope']==[model]]
    database = ROOT/'queues'/('parents_'+label)/'merged.sqlite'
    records = {r['item_id']:r for r in read_db(database)}
    assert set(records)=={x['id'] for x in parent_items} and len(records)==32
    entries,coverage,children = [],[],[]
    import json
    for p in parent_items:
        rec = records[p['id']]
        meta = json.loads(rec['metadata_json'])
        assert rec['model']==model and all(meta.get(k)==v for k,v in p.items())
        eligible = (rec['status']=='ok' and type(rec['probability']) in (int,float)
                    and math.isfinite(rec['probability']) and 0<=rec['probability']<=1)
        assert (rec['status']=='ok') == eligible
        coverage.append(dict(parent_item_id=p['id'], model=model,context=p['context_id'],
            reliability=p['reliability'],prior=p['prior'],signal=p['signal'],family_id=p['family_id'],
            status=rec['status'],eligible=eligible,children_constructed=32 if eligible else 0))
        entry = dict(parent=p,record=rec,continuation_model=model,
            parent_record_sha256=text_hash(canonical(rec)),parent_final_sha256=text_hash(meta['final_output']))
        entries.append(entry)
        if eligible:
            assert meta['final_output']
            children += [child(entry,'transfer',rho,branch,b,'selector',condition)
                         for rho,branch,b in legal_branches(p['signal']) for condition in CONDITIONS]
    cohort = ROOT/'cohorts'/label
    write_jsonl(cohort/'parents.jsonl',entries)
    write_jsonl(cohort/'coverage.jsonl',coverage)
    write_jsonl(cohort/'children.jsonl',children)
    followups = children+(read_jsonl(ROOT/'manifests/expression.jsonl') if label=='phi' else [])
    allocation = prepare_queue(followups,model,'followups_'+label,6 if label=='phi' else 2)
    paths = [p for p in cohort.glob('*') if p.is_file()]
    if allocation:
        paths += [p for p in (ROOT/'queues'/('followups_'+label)).rglob('*') if p.is_file()]
    write_json(cohort/'CONSTRUCTION.json',dict(constructed_at=now(),model=model,
        parent_queue_sha256=sha(database),parent_status_counts=dict(Counter(r['status'] for r in records.values())),
        planned_parents=32,valid_parents=sum(x['eligible'] for x in coverage),
        constructed_children=len(children),stage1_calls=768 if label=='phi' else 0,allocation=allocation,
        design_seal_sha256=sha(ROOT/'DESIGN_SEAL.json'),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
    print(canonical(dict(label=label,valid_parents=sum(x['eligible'] for x in coverage),
                         children=len(children),continuation_calls=len(followups))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('build','construct'))
    parser.add_argument('--model',choices=('phi','qwen'))
    args=parser.parse_args()
    if args.action=='build':
        build()
    else:
        assert args.model
        construct(args.model)
