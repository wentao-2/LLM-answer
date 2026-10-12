"""Small mathematical and integration checks. Fixtures are not scientific results."""
import copy
import itertools
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from build_extension import (ARMS, CANDIDATES, QWEN, canonical, history_children,
                             oracle, read_jsonl, static_manifests)
from analyze_extension import (analyze, analyze_gate, attribution_analyses, full_bounds,
                               native_analyses, summarize_units, unit)

ROOT = Path(__file__).resolve().parents[4]
R3 = ROOT/'i16-self-evidence/results/validation-8gpu-recovery/design_snapshot'
SCHEMA = '''CREATE TABLE results (id TEXT PRIMARY KEY, model TEXT, item_id TEXT,
status TEXT, probability REAL, raw_output TEXT, input_tokens INTEGER, output_tokens INTEGER,
duration_sec REAL, created_at TEXT, metadata_json TEXT)'''


def database(path, rows, model, missing=(), invalid=()):
    c = sqlite3.connect(path); c.execute(SCHEMA)
    for i, x in enumerate(rows):
        if i in missing:
            continue
        status = 'invalid' if i in invalid else 'ok'
        c.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (model+':'+x['id'], model, x['id'], status, x['oracle'] if status == 'ok' else None,
             'unit-test fixture', 1, 1, 0, 'fixture', canonical(x)))
    c.commit(); c.close()


def rec(rid, p, q=.5, family='f', **item):
    x = dict(id=rid, oracle=q, family_id=family, world_id='w', template_id='template',
             protocol='positive', visibility='signal_visible', signal=1, answer=1,
             reminder='scoped', output_instruction='compatible', attribution='self')
    x.update(item)
    return dict(model='fixture', item=x, status='ok' if p is not None else 'truncated', probability=p, metadata={})


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = read_jsonl(R3/'manifest-main.jsonl')
        cls.gate = read_jsonl(R3/'manifest-gate.jsonl')
        cls.manifests = static_manifests(cls.main, cls.gate)

    def test_matrix_and_frozen_prompt_reuse(self):
        self.assertEqual({k: len(v) for k, v in self.manifests.items()},
            {'gate': 96, 'gate-independent': 32, 'cross-family': 1728,
             'native-factorial': 2304, 'history-stage1': 48})
        old = {x['id']: x for x in self.main}
        for name in ('cross-family', 'native-factorial'):
            for x in self.manifests[name]:
                self.assertEqual(x['prompt'], old[x['origin_r3_id']]['prompt'])
        blocks = {}
        for x in self.manifests['native-factorial']:
            k = tuple(x[f] for f in ('world_id', 'template_id', 'protocol', 'signal', 'answer'))
            blocks.setdefault(k, set()).add((x['native_mode'], x['output_instruction'], x['budget'], x['attribution']))
        self.assertEqual(len(blocks), 144)
        self.assertTrue(all(len(v) == 16 for v in blocks.values()))

    def test_oracles_against_latent_enumeration(self):
        for x in self.manifests['gate']+self.manifests['cross-family']:
            mass = [0., 0.]
            for h, s, t, coin in itertools.product((0, 1), repeat=4):
                r = x['reliability']; prior = x['prior']
                w = (prior if h else 1-prior)*(r if s == h else 1-r)*(r if t == h else 1-r)*.5
                a = {'positive': s, 'negative': 1-s, 'zero': coin, 'independent': t, 'baseline': None}[x['protocol']]
                vis = x['visibility']
                observed = ((vis == 'signal_hidden' and a == x['answer']) or
                            (vis in ('baseline', 'answer_hidden') and s == x['signal']) or
                            (vis == 'signal_visible' and s == x['signal'] and a == x['answer']))
                if observed:
                    mass[h] += w
            self.assertAlmostEqual(x['oracle'], mass[1]/sum(mass), places=12)

    def test_shared_missing_baseline_cancels(self):
        baseline, zero, one = rec('b', None), rec('zero', .2), rec('one', .7)
        units = [unit([(zero, -1), (baseline, 1)], 'f'), unit([(one, 1), (baseline, -1)], 'f')]
        out = summarize_units(units)
        self.assertIsNone(out['point']['estimate'])
        self.assertAlmostEqual(out['bounds']['lower'], .25)
        self.assertAlmostEqual(out['bounds']['upper'], .25)
        self.assertEqual(out['bounds']['n_missing_records'], 1)
        self.assertEqual(out['bounds']['n_missing_nonzero_coefficients'], 0)

    def test_full_bounds_fixed_family_weights(self):
        records = [rec('a', .1, family='f1'), rec('b', .3, family='f1'), rec('c', .9, family='f2')]
        units = [unit([(r, 1)], r['item']['family_id']) for r in records]
        out = full_bounds(units)
        self.assertAlmostEqual(out['lower'], .55)
        self.assertEqual(out['lower'], out['upper'])

    def test_sixteen_cell_effects(self):
        records = []
        for m, w, b, a in itertools.product(('on', 'off'), ('compatible', 'restrictive'), (4096, 12288), ('self', 'other')):
            p = .5 + (.02 if m == 'on' else 0) + (.01 if w == 'compatible' else 0) + (.005 if b == 12288 else 0)
            records.append(rec(str((m,w,b,a)), p, native_mode=m, output_instruction=w, budget=b, attribution=a))
        result = native_analyses(records)[0]['contrasts']
        self.assertAlmostEqual(result['native_mode_mae']['point']['estimate'], .02)
        self.assertAlmostEqual(result['wording_mae']['point']['estimate'], .01)
        self.assertAlmostEqual(result['budget_mae']['point']['estimate'], .005)
        self.assertAlmostEqual(result['native_mode_attribution']['point']['estimate'], 0.)
        self.assertAlmostEqual(result['mode_wording_budget_mae']['point']['estimate'], 0.)
        self.assertTrue(all(x['complete_valid_sets'] == 1 for x in result.values()))

    def test_decomposition_uses_identical_complete_pairs(self):
        rows = [rec('s0', .9, world_id='w0', attribution='self'),
                rec('o0', None, world_id='w0', attribution='other'),
                rec('s1', .4, world_id='w1', attribution='self'),
                rec('o1', .3, world_id='w1', attribution='other')]
        result = attribution_analyses(rows)[0]
        self.assertAlmostEqual(result['self_directional_error']['point']['estimate'], -.1)
        self.assertAlmostEqual(result['other_directional_error']['point']['estimate'], -.2)
        self.assertAlmostEqual(result['gap']['point']['estimate'], .1)
        self.assertEqual(result['pair_sign_counts'], {'both_negative': 1})
        self.assertEqual(result['self_directional_error']['complete_valid_sets'], 1)

    def test_history_provenance_oracle_and_failed_parent(self):
        parent = self.manifests['history-stage1'][0]
        final = 'Visible explanation kept verbatim.\nFINAL_JSON: {"p_h1": 0.5}'
        record = dict(model=QWEN, item_id=parent['id'], status='ok', probability=.5,
                      metadata_json=canonical(dict(parent, final_output=final, parsed_answer=None)))
        children, cohort = history_children([parent], [record], [QWEN])
        self.assertEqual(len(children), 16)
        self.assertTrue(all(x['initial_answer'] == 1 for x in children))
        for x in children:
            if x['history_arm'] == 'native_self':
                self.assertEqual(x['messages'][1]['content'], final)
            else:
                self.assertIn(final, x['prompt'])
            if x['evidence_kind'] in ('none', 'copy'):
                self.assertAlmostEqual(x['oracle'], parent['oracle'])
        for evidence in ('none', 'copy', 'independent'):
            for b in {x['next_signal'] for x in children if x['evidence_kind'] == evidence}:
                self.assertEqual(len({x['oracle'] for x in children if x['evidence_kind'] == evidence and x['next_signal'] == b}), 1)
        failed = dict(record, status='truncated', probability=None)
        empty, cohort = history_children([parent], [failed], [QWEN])
        self.assertEqual(empty, [])
        self.assertEqual(cohort[0]['descendant_status'], 'not_constructed_parent_failure')

    def test_gate_requires_all_terminal(self):
        rows = self.manifests['gate-independent']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.sqlite'
            database(path, rows, QWEN, missing=(31,))
            result = analyze_gate(rows, path, QWEN)
            self.assertFalse(result['passes'])
            self.assertEqual(result['groups'][0]['n_valid'], 31)
            c = sqlite3.connect(path)
            x = rows[31]
            c.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (QWEN+':'+x['id'],QWEN,x['id'],'truncated',None,'fixture',1,1,0,'fixture',canonical(x)))
            c.commit(); c.close()
            self.assertTrue(analyze_gate(rows, path, QWEN)['passes'])

    def test_static_analysis_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, model in (('gate', CANDIDATES[0]), ('cross-family', CANDIDATES[0]), ('native-factorial', QWEN)):
                rows = self.manifests[name]
                manifest = Path(directory)/(name+'.jsonl')
                manifest.write_text('\n'.join(canonical(x) for x in rows)+'\n', encoding='utf-8')
                db = Path(directory)/(name+'.sqlite')
                database(db, rows, model)
                result = analyze(manifest, db, [model])['models'][0]
                self.assertAlmostEqual(result['overall']['mae']['estimate'], 0.)
                self.assertEqual(result['overall']['n_valid'], len(rows))

    def test_history_analysis_integration(self):
        parent = self.manifests['history-stage1'][0]
        record = dict(model=QWEN, item_id=parent['id'], status='ok', probability=.4,
                      metadata_json=canonical(dict(parent, final_output='FINAL_JSON: {"p_h1":0.4}')))
        children, _ = history_children([parent], [record], [QWEN])
        with tempfile.TemporaryDirectory() as directory:
            manifest, db = Path(directory)/'history.jsonl', Path(directory)/'history.sqlite'
            manifest.write_text('\n'.join(canonical(x) for x in children)+'\n', encoding='utf-8')
            database(db, children, QWEN)
            result = analyze(manifest, db, [QWEN])['models'][0]
            self.assertEqual(result['overall']['n_valid'], 16)
            for entry in result['history']['presentation']:
                for contrast in entry['contrasts'].values():
                    self.assertAlmostEqual(contrast['point']['estimate'], 0.)
            for entry in result['history']['information']:
                self.assertAlmostEqual(entry['update']['residual']['estimate'], 0.)


if __name__ == '__main__':
    unittest.main()
