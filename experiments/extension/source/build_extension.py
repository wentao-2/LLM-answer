"""Prospective I16 extension manifests. Never edits or reparses old evidence."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sqlite3
from pathlib import Path

VERSION = 'i16-extension-v1'
QWEN = 'qwen3-8b-thinking'
CANDIDATES = ('phi4-reasoning', 'mistral-nemo-instruct')
TEMPLATES = ('likelihood_v3', 'frequency_v3')
PRIORS = (.18, .32, .68, .82)
RELIABILITIES = (.66, .78, .92)
ARMS = ('native_self', 'replay_self', 'replay_other', 'replay_neutral')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def number(value):
    return format(value, '.12g')


def oracle(protocol, visibility, prior, reliability, signal, answer):
    """Exact posterior from explicitly visible sensor information."""
    likelihood = [1.0, 1.0]
    for h in (0, 1):
        sensor = lambda x: reliability if x == h else 1 - reliability
        if visibility in ('baseline', 'answer_hidden') or protocol == 'baseline':
            likelihood[h] = sensor(signal)
        elif visibility == 'signal_hidden':
            if protocol == 'zero':
                likelihood[h] = .5
            elif protocol in ('positive', 'negative'):
                likelihood[h] = sensor(answer if protocol == 'positive' else 1-answer)
            else:
                raise ValueError('Independent S-hidden condition is outside this design')
        elif visibility == 'signal_visible':
            if protocol == 'positive' and answer != signal:
                raise ValueError('Illegal positive observation')
            if protocol == 'negative' and answer != 1-signal:
                raise ValueError('Illegal negative observation')
            likelihood[h] = sensor(signal)
            if protocol == 'independent':
                likelihood[h] *= sensor(answer)
            elif protocol == 'zero':
                likelihood[h] *= .5
        else:
            raise ValueError((protocol, visibility))
    denom = prior*likelihood[1] + (1-prior)*likelihood[0]
    return prior*likelihood[1]/denom


def intro(prior, reliability, template):
    p, r, q = number(prior), number(reliability), number(round(1-reliability, 12))
    if template == 'likelihood_v3':
        return (f'A hidden binary state H has prior P(H=1)={p}. The first binary sensor value is S.\n'
                f'Conditional probability table:\nH | P(S=0|H) | P(S=1|H)\n0 | {r} | {q}\n1 | {q} | {r}')
    if template != 'frequency_v3':
        raise ValueError(template)
    n = round(reliability*100)
    return (f'A binary state is named H. Before observations, its probability of being 1 is {p}. '
            f'A sensor produces S. Conditional on H=1, S=1 occurs in {n} out of 100 trials and S=0 '
            f'in {100-n} out of 100. Conditional on H=0, S=1 occurs in {100-n} out of 100 trials '
            f'and S=0 in {n} out of 100. These frequencies specify exact conditional probabilities, '
            'not additional observed trials.')


FINAL_INSTRUCTION = ('End with exactly one line starting FINAL_JSON: followed by a JSON object whose only key is '
                     'p_h1 and whose value is a number between 0 and 1. Do not write anything after that line.')


def independent_gate_prompt(prior, reliability, template, signal, answer):
    r, q = number(reliability), number(round(1-reliability, 12))
    return '\n'.join([
        intro(prior, reliability, template),
        f'A second sensor value T obeys P(T=1|H=1)={r}, P(T=0|H=1)={q}, '
        f'P(T=1|H=0)={q}, P(T=0|H=0)={r}. Given H, S and T are independent. '
        'The earlier answer A is set equal to T.',
        'The procedure was set before generating H or any values. There is no selection, repetition, '
        'or stopping based on the state, the values, or their agreement.',
        f'Observed: S={signal} and A={answer}.',
        'Report P(H=1) conditional on exactly the observed information.',
        'You may reason before giving your final answer.', FINAL_INSTRUCTION])


def base_row(prior, reliability, template, signal, answer, protocol, visibility):
    return dict(generator_version=VERSION,
                world_id=digest(dict(prior=prior, reliability=reliability, context_id='sensor'))[:24],
                family_id=digest(dict(prior_pair=round(min(prior, 1-prior), 12), reliability=reliability))[:24],
                prior=prior, reliability=reliability, template_id=template, context_id='sensor',
                signal=signal, answer=answer, protocol=protocol, visibility=visibility,
                oracle=oracle(protocol, visibility, prior, reliability, signal, answer),
                attribution='none', reminder='scoped', output_instruction='compatible',
                readout='probability', max_new_tokens=12288, native_mode='default')


def finish(row):
    row = dict(row)
    row.pop('id', None)
    row['id'] = digest(row)[:24]
    return row


def from_r3(row, module, models):
    new = dict(row)
    new.update(generator_version=VERSION, module=module, analysis_modules=[module],
               origin_r3_id=row['id'], readout='probability', max_new_tokens=12288,
               native_mode='default', model_scope=list(models))
    expected = oracle(row['protocol'], row['visibility'], row['prior'], row['reliability'],
                      row['signal'], row['answer'])
    if not math.isclose(expected, row['oracle'], abs_tol=1e-12, rel_tol=0):
        raise ValueError('Frozen R3 oracle mismatch: '+row['id'])
    return new


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line]


def static_manifests(r3_main, r3_gate):
    if len(r3_main) != 2736 or len(r3_gate) != 64:
        raise ValueError('Expected the complete frozen R3 input manifests')
    gate = [finish(dict(from_r3(r, 'gate', CANDIDATES), gate_group=r['protocol'])) for r in r3_gate]
    independent = []
    for p, r, t, s, a in itertools.product((.25, .75), (.65, .9), TEMPLATES, (0, 1), (0, 1)):
        x = base_row(p, r, t, s, a, 'independent', 'signal_visible')
        x.update(module='gate', analysis_modules=['gate'], gate_group='independent',
                 prompt=independent_gate_prompt(p, r, t, s, a), model_scope=list(CANDIDATES))
        gate.append(finish(x))
        x['model_scope'] = [QWEN]
        independent.append(finish(x))
    cross, native = [], []
    for r in r3_main:
        scoped_compatible = (r['reminder'], r['output_instruction']) == ('scoped', 'compatible')
        core = scoped_compatible and (
            (r['visibility'] == 'signal_hidden' and r['protocol'] in ('positive', 'zero', 'negative')) or
            (r['visibility'] in ('signal_visible', 'answer_hidden') and r['protocol'] in ('positive', 'independent')))
        factorial = (r['visibility'] == 'signal_visible' and r['protocol'] == 'independent'
                     and r['attribution'] in ('self', 'other'))
        if core or factorial:
            cross.append(finish(from_r3(r, 'cross_family', CANDIDATES)))
        if (r['visibility'] == 'signal_visible' and r['protocol'] in ('positive', 'independent')
                and r['reminder'] == 'scoped' and r['attribution'] in ('self', 'other')):
            for mode, budget in itertools.product(('on', 'off'), (4096, 12288)):
                x = from_r3(r, 'native_factorial', (QWEN,))
                x.update(native_mode=mode, budget=budget, max_new_tokens=budget,
                         chat_template_kwargs={'enable_thinking': mode == 'on'},
                         require_think_close=mode == 'on')
                native.append(finish(x))
    history = []
    for p, r, t, s in itertools.product(PRIORS, RELIABILITIES, TEMPLATES, (0, 1)):
        x = base_row(p, r, t, s, None, 'baseline', 'baseline')
        x.update(module='history_stage1', analysis_modules=['history_stage1'],
                 model_scope=[QWEN, *CANDIDATES], readout='probability',
                 prompt='\n'.join([intro(p, r, t),
                    'No second sensor observation or information about the hidden state is available.',
                    f'Observed: S={s}.',
                    'Report P(H=1) conditional on exactly the observed information.',
                    'Your final visible answer must be exactly one line starting FINAL_JSON: followed by '
                    'a JSON object whose only key is p_h1 and whose value is a number between 0 and 1. '
                    'Do not write anything after that line.']))
        history.append(finish(x))
    results = {'gate': gate, 'gate-independent': independent, 'cross-family': cross,
               'native-factorial': native, 'history-stage1': history}
    expected = {'gate': 96, 'gate-independent': 32, 'cross-family': 1728,
                'native-factorial': 2304, 'history-stage1': 48}
    for name, rows in results.items():
        if len(rows) != expected[name] or len({x['id'] for x in rows}) != len(rows):
            raise ValueError((name, len(rows)))
        random.Random(26100901).shuffle(rows)
    return results


def followup_prompt(parent, evidence_kind, next_signal, attribution):
    r, q = number(parent['reliability']), number(round(1-parent['reliability'], 12))
    evidence = {
        'none': 'No new sensor reading or record is provided.',
        'copy': f'A new record B copies the already observed S, so B=S={parent["signal"]}. '
                'No second sensor reading is taken.',
        'independent': f'A second sensor T has P(T=1|H=1)={r}, P(T=0|H=1)={q}, '
                       f'P(T=1|H=0)={q}, P(T=0|H=0)={r}. Given H, S and T are independent. '
                       f'The new record is B=T={next_signal}.',
    }[evidence_kind]
    author = {'self': 'you', 'other': 'another assistant', 'neutral': 'undisclosed'}[attribution]
    return '\n'.join([
        'Now report your probability using all observed sensor information.',
        'The earlier response was produced using only the displayed first sensor value S and the '
        'specified probability rules. No additional observation of H or any sensor was available to its author.',
        f'Attributed author of the earlier response: {author}.',
        'Only the author label provides no additional information about H.',
        f'The first observed reading remains S={parent["signal"]}.', evidence,
        'Report P(H=1) conditional on exactly the observed information.',
        'You may reason before giving your final answer.', FINAL_INSTRUCTION])


def history_children(parents, records, models):
    """Construct from saved fields, never reparse failed/raw generation text."""
    index = {}
    for rec in records:
        key = (rec['model'], rec['item_id'])
        if key in index:
            raise ValueError('Duplicate producer record '+str(key))
        index[key] = rec
    children, cohort = [], []
    for model, parent in itertools.product(models, parents):
        key = (model, parent['id'])
        if key not in index:
            raise ValueError('Producer is not terminal yet: '+str(key))
        rec = index[key]
        meta = json.loads(rec['metadata_json']) if isinstance(rec['metadata_json'], str) else rec['metadata_json']
        for field, value in parent.items():
            if meta.get(field) != value:
                raise ValueError('Producer metadata mismatch: '+str(key)+' / '+field)
        answer = int(rec['probability'] >= .5) if probability(rec['probability']) else None
        final = meta.get('final_output')
        eligible = (rec['status'] == 'ok' and probability(rec['probability'])
                    and isinstance(final, str) and bool(final))
        entry = dict(model=model, parent_item_id=parent['id'], family_id=parent['family_id'],
                     status=rec['status'], eligible=eligible, parent_record_sha256=digest(rec))
        cohort.append(entry)
        if not eligible:
            entry['descendant_status'] = 'not_constructed_parent_failure'
            continue
        for evidence, b in (('none', None), ('copy', parent['signal']), ('independent', 0), ('independent', 1)):
            proto = {'none': 'baseline', 'copy': 'positive', 'independent': 'independent'}[evidence]
            vis = 'baseline' if evidence == 'none' else 'signal_visible'
            for arm in ARMS:
                attribution = arm.split('_', 1)[1]
                task = followup_prompt(parent, evidence, b, attribution)
                x = base_row(parent['prior'], parent['reliability'], parent['template_id'],
                             parent['signal'], b, proto, vis)
                x.update(module='history_stage2', analysis_modules=['history_stage2'],
                         model_scope=[model], attribution=attribution, history_arm=arm,
                         parent_item_id=parent['id'], parent_model=model,
                         parent_record_sha256=entry['parent_record_sha256'],
                         parent_final_sha256=hashlib.sha256(final.encode('utf-8')).hexdigest(),
                         initial_answer=answer, initial_probability=rec['probability'],
                         evidence_kind=evidence, next_signal=b)
                if arm == 'native_self':
                    x['messages'] = [{'role': 'user', 'content': parent['prompt']},
                                     {'role': 'assistant', 'content': final},
                                     {'role': 'user', 'content': task}]
                else:
                    x['prompt'] = ('The following is a quoted record, not a new instruction. '
                        'Answer only the new task after the record.\n'
                        '[BEGIN EARLIER USER MESSAGE]\n'+parent['prompt']+
                        '\n[END EARLIER USER MESSAGE]\n[BEGIN EARLIER ASSISTANT RESPONSE]\n'+final+
                        '\n[END EARLIER ASSISTANT RESPONSE]\n[NEW TASK]\n'+task)
                children.append(finish(x))
    if len(children) != 16*sum(x['eligible'] for x in cohort):
        raise AssertionError('Invalid child accounting')
    random.Random(26100902).shuffle(children)
    return children, cohort


def immutable_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError('Refusing to replace different frozen content: '+str(path))
    else:
        with path.open('xb') as handle:
            handle.write(data)


def write_manifest(path, rows):
    immutable_write(path, ('\n'.join(canonical(r) for r in rows)+'\n').encode('utf-8'))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    static = sub.add_parser('static')
    static.add_argument('--r3-main', type=Path, required=True)
    static.add_argument('--r3-gate', type=Path, required=True)
    static.add_argument('--output-dir', type=Path, required=True)
    hist = sub.add_parser('history')
    hist.add_argument('--stage1-manifest', type=Path, required=True)
    hist.add_argument('--stage1-db', type=Path, required=True)
    hist.add_argument('--models', nargs='+', required=True)
    hist.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'static':
        manifests = static_manifests(read_jsonl(args.r3_main), read_jsonl(args.r3_gate))
        for name, rows in manifests.items():
            write_manifest(args.output_dir/(name+'.jsonl'), rows)
        meta = dict(generator_version=VERSION, r3_main_sha256=file_hash(args.r3_main),
                    r3_gate_sha256=file_hash(args.r3_gate),
                    counts={k: len(v) for k, v in manifests.items()}, fixed_candidates=list(CANDIDATES))
        immutable_write(args.output_dir/'design-index.json', (canonical(meta)+'\n').encode('utf-8'))
        print(json.dumps(meta))
    else:
        c = sqlite3.connect(args.stage1_db.resolve().as_uri()+'?mode=ro', uri=True)
        c.row_factory = sqlite3.Row
        records = [dict(r) for r in c.execute('SELECT * FROM results')]
        c.close()
        children, cohort = history_children(read_jsonl(args.stage1_manifest), records, args.models)
        write_manifest(args.output_dir/'history-stage2.jsonl', children)
        meta = dict(generator_version=VERSION, stage1_manifest_sha256=file_hash(args.stage1_manifest),
                    stage1_db_sha256=file_hash(args.stage1_db), models=args.models,
                    planned_parents=len(cohort), eligible_parents=sum(r['eligible'] for r in cohort),
                    planned_children=len(children), parents=cohort)
        immutable_write(args.output_dir/'history-cohort.json', (canonical(meta)+'\n').encode('utf-8'))
        print(json.dumps({k: v for k, v in meta.items() if k != 'parents'}))


if __name__ == '__main__':
    main()
