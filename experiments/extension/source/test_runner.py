"""Small CPU-only checks of the extension's evidence-preservation contracts."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

import run_extension as runner
import prepare_runs
import schedule_runs


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(("" if skip_special_tokens else "<eos>") if value == 0 else chr(value)
                       for value in token_ids)


class ReadoutTests(unittest.TestCase):
    def test_exact_terminal_readouts(self):
        self.assertEqual(runner.parse_final('Reasoning is visible.\nFINAL_JSON: {"p_h1":0.4}'),
                         {"probability": 0.4, "parsed_answer": None})
        self.assertEqual(runner.parse_final('FINAL_JSON: {"answer":1,"p_h1":0.7}', "answer_probability"),
                         {"probability": 0.7, "parsed_answer": 1})
        for text in (
            'FINAL_JSON: {"p_h1":0.4,"p_h1":0.7}',
            'FINAL_JSON: {"p_h1":true}', 'FINAL_JSON: {"p_h1":NaN}',
            'FINAL_JSON: {"p_h1":Infinity}', 'FINAL_JSON: {"p_h1":1.1}',
            'FINAL_JSON: {"p_h1":-0.1}', 'FINAL_JSON: {"p_h1":"0.5"}',
            'FINAL_JSON: {"p_h1":0.4,"answer":1}',
            'FINAL_JSON: {"p_h1":0.4}\ntrailing explanation',
            'FINAL_JSON: {"p_h1":0.4}\nFINAL_JSON: {"p_h1":0.4}',
            'embedded FINAL_JSON: {"p_h1":0.4}',
            '```json\nFINAL_JSON: {"p_h1":0.4}\n```',
            'FINAL_JSON: {"p_h1":' + '9' * 400 + '}',
        ):
            with self.subTest(text=text):
                self.assertIsNone(runner.parse_final(text))
        for answer in ("true", "1.0", "2", '"1"', "null"):
            self.assertIsNone(runner.parse_final('FINAL_JSON: {"answer":' + answer + ',"p_h1":0.7}',
                                                 "answer_probability"))

    def test_no_parse_before_eos_or_think_close(self):
        tokenizer = CharacterTokenizer()
        on = runner.GenerationProfile(100, 100, '{}', True, 'probability')
        off = runner.GenerationProfile(100, 100, '{}', False, 'probability')
        final = 'FINAL_JSON: {"p_h1":0.75}'
        closed = tokenizer.encode('FINAL_JSON: {"p_h1":0.1}\n</think>\n' + final)
        status, value, raw, tokens, metadata = runner.classify_output(closed + [0, 0], tokenizer, {0}, on)
        self.assertEqual((status, value), ('ok', 0.75))
        self.assertEqual(metadata['final_output'], final)
        self.assertTrue(raw.endswith('<eos>'))
        self.assertEqual(tokens[-1], 0)
        self.assertEqual(len(tokens), len(closed) + 1)
        self.assertEqual(runner.classify_output(closed, tokenizer, {0}, on)[:2], ('truncated', None))
        self.assertEqual(runner.classify_output(tokenizer.encode(final) + [0], tokenizer, {0}, on)[:2],
                         ('invalid', None))
        self.assertEqual(runner.classify_output(tokenizer.encode(final) + [0], tokenizer, {0}, off)[:2],
                         ('ok', 0.75))


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = {'max_new_tokens': 12288, 'max_input_tokens': 1600, 'do_sample': False, 'num_beams': 1}
        self.qwen = {'chat_template_kwargs': {'enable_thinking': True}, 'require_think_close': True}

    def test_item_overrides_control_batch_and_parse_together(self):
        items = [
            {'id': 'on'},
            {'id': 'off', 'chat_template_kwargs': {'enable_thinking': False}, 'max_new_tokens': 4096},
            {'id': 'on2'},
            {'id': 'on_small', 'max_new_tokens': 4096},
            {'id': 'stage1', 'readout': 'answer_probability'},
        ]
        groups = runner.group_profiles(items, self.config, self.qwen)
        self.assertEqual([[item['id'] for item in rows] for _, rows in groups],
                         [['on', 'on2'], ['off'], ['on_small'], ['stage1']])
        self.assertFalse(groups[1][0].require_think_close)
        self.assertEqual(groups[1][0].max_new_tokens, 4096)
        phi = runner.effective_profile(self.config, {'require_think_close': True}, {})
        self.assertEqual(phi.chat_template_kwargs, {})
        self.assertTrue(phi.require_think_close)
        for change in ({'do_sample': True}, {'num_beams': 2}, {'max_new_tokens': True},
                       {'chat_template_kwargs': {'enable_thinking': False}, 'require_think_close': True},
                       {'chat_template_kwargs': {'chat_template': 'replacement'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.effective_profile(self.config, self.qwen, change)

    def test_history_is_forwarded_without_added_system(self):
        messages = [{'role': 'user', 'content': 'First question'},
                    {'role': 'assistant', 'content': 'Explanation.\nFINAL_JSON: {"answer":1,"p_h1":0.7}'},
                    {'role': 'user', 'content': 'New evidence'}]
        self.assertIs(runner.item_messages({'messages': messages, 'prompt': 'ignored'}), messages)
        self.assertEqual(runner.item_messages({'prompt': 'Question'}), [{'role': 'user', 'content': 'Question'}])
        self.assertEqual(runner.select_rows([{'id': 'a', 'model_scope': ['q']}, {'id': 'a', 'model_scope': ['p']}], 'q'),
                         [{'id': 'a', 'model_scope': ['q']}])
        with self.assertRaises(ValueError):
            runner.select_rows([{'id': 'a'}, {'id': 'a', 'model_scope': ['q']}], 'q')

    def test_context_budget(self):
        runner.validate_context_budget(100, 400, 500)
        with self.assertRaises(RuntimeError):
            runner.validate_context_budget(101, 400, 500)
        with self.assertRaises(ValueError):
            runner.validate_context_budget(100, 400, None)


class PersistenceTests(unittest.TestCase):
    def test_saved_failures_are_skipped_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            db = runner.connect_db(Path(directory) / 'results.sqlite')
            try:
                for item_id, status in (('a', 'ok'), ('b', 'invalid'), ('c', 'truncated')):
                    values = ('q:' + item_id, 'q', item_id, status, None, 'raw', 10, 2, 0.1, 'timestamp', '{}')
                    runner.insert_result(db, values)
                pending, done = runner.remaining_rows(db, 'q', [{'id': char} for char in 'abcd'])
                self.assertEqual(pending, [{'id': 'd'}])
                self.assertEqual(done, {'a', 'b', 'c'})
                with self.assertRaises(sqlite3.IntegrityError):
                    runner.insert_result(db, values)
                runner.require_metadata(db, 'frozen', {'value': 1})
                runner.require_metadata(db, 'frozen', {'value': 1})
                with self.assertRaises(RuntimeError):
                    runner.require_metadata(db, 'frozen', {'value': 2})
            finally:
                db.close()

    def test_all_frozen_artifacts_and_running_copy_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            config = {'manifest': 'manifest.jsonl', 'experiment_protocol': 'PROTOCOL.md'}
            (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            (root / 'manifest.jsonl').write_text('{"id":"a"}\n', encoding='utf-8')
            (root / 'PROTOCOL.md').write_text('Frozen protocol', encoding='utf-8')
            (source / 'run_extension.py').write_text('# fixture source\n', encoding='utf-8')
            frozen = {'config_sha256': runner.digest(root / 'config.json'),
                      'manifest_sha256': runner.digest(root / 'manifest.jsonl'),
                      'source_hashes': {'run_extension.py': runner.digest(source / 'run_extension.py')},
                      'protocol_hashes': {'PROTOCOL.md': runner.digest(root / 'PROTOCOL.md')}}
            (root / 'frozen.json').write_text(json.dumps(frozen), encoding='utf-8')
            runner.validate_frozen(root / 'config.json', source / 'run_extension.py')
            for relative in ('config.json', 'manifest.jsonl', 'source/run_extension.py', 'PROTOCOL.md'):
                path = root / relative
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                with self.subTest(relative=relative), self.assertRaises(RuntimeError):
                    runner.validate_frozen(root / 'config.json', source / 'run_extension.py')
                path.write_bytes(original)
            with self.assertRaises(RuntimeError):
                runner.validate_frozen(root / 'config.json', root / 'different_runner.py')


class QueueTests(unittest.TestCase):
    def make_queue(self, directory):
        root = Path(directory)
        config = {'seed': 1601, 'batch_size': 4, 'max_input_tokens': 1000, 'max_new_tokens': 100,
                  'models': [{'name': 'q', 'path': '/unused/q', 'chat_template_kwargs': {'enable_thinking': True}},
                             {'name': 'p', 'path': '/unused/p', 'require_think_close': True}]}
        (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        (root / 'protocol.md').write_text('Frozen CPU test protocol', encoding='utf-8')
        items = [{'id': str(index), 'prompt': 'Question', 'model_scope': ['q', 'p']} for index in range(6)]
        items.extend({'id': 'off' + str(index), 'prompt': 'Question', 'model_scope': ['q'],
                      'chat_template_kwargs': {'enable_thinking': False}, 'max_new_tokens': 50}
                     for index in range(6))
        prepare_runs.write_manifest(root / 'manifest.jsonl', items)
        queue = prepare_runs.prepare(root / 'config.json', [root / 'manifest.jsonl'], root / 'protocol.md',
                                     root / 'queue', shards_per_model=2)
        return root / 'queue', queue

    def test_prepare_preserves_ids_balances_profiles_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root, queue = self.make_queue(directory)
            _, verified, _ = schedule_runs.validate_queue(root)
            self.assertEqual(verified['expected_records'], 18)
            self.assertEqual(len(queue['jobs']), 4)
            for job in queue['jobs']:
                run_dir = root / job['run_dir']
                items = [json.loads(line) for line in (run_dir / 'manifest.jsonl').read_text().splitlines()]
                self.assertEqual(len(items), 6 if job['model'] == 'q' else 3)
                if job['model'] == 'q':
                    self.assertEqual(sum('off' in item['id'] for item in items), 3)
            with self.assertRaises(FileExistsError):
                prepare_runs.prepare(Path(directory) / 'config.json', [Path(directory) / 'manifest.jsonl'],
                                     Path(directory) / 'protocol.md', root, shards_per_model=2)

    def test_merge_uses_exact_ids_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root, queue = self.make_queue(directory)
            _, _, frozen = schedule_runs.validate_queue(root)
            with self.assertRaises(RuntimeError):
                schedule_runs.merge_queue(root, queue, frozen)
            for job in queue['jobs']:
                run_dir = root / job['run_dir']
                db = runner.connect_db(run_dir / 'results.sqlite')
                worker_freeze = json.loads((run_dir / 'frozen.json').read_text())
                try:
                    for key in ('config_sha256', 'manifest_sha256', 'source_hashes', 'protocol_hashes'):
                        runner.require_metadata(db, key, worker_freeze[key])
                    for item_id in job['expected_item_ids']:
                        runner.insert_result(db, (job['model'] + ':' + item_id, job['model'], item_id, 'invalid', None,
                                                  'raw', 10, 1, 0.1, 'timestamp', '{}'))
                finally:
                    db.close()
                runner.atomic_json(run_dir / 'terminal.json', {'returncode': 0})
            merged_path = schedule_runs.merge_queue(root, queue, frozen)
            first_hash = runner.digest(merged_path)
            self.assertEqual(schedule_runs.merge_queue(root, queue, frozen), merged_path)
            self.assertEqual(runner.digest(merged_path), first_hash)
            db = sqlite3.connect(merged_path)
            try:
                self.assertEqual(db.execute('SELECT count(*) FROM results').fetchone()[0], 18)
                with db:
                    db.execute("UPDATE results SET raw_output='tampered' WHERE id=(SELECT min(id) FROM results)")
            finally:
                db.close()
            changed_hash = runner.digest(merged_path)
            with self.assertRaises(RuntimeError):
                schedule_runs.merge_queue(root, queue, frozen)
            self.assertEqual(runner.digest(merged_path), changed_hash)


@unittest.skipUnless(os.name == 'posix', 'Requires POSIX inherited file locks')
class PosixQueueTests(unittest.TestCase):
    def fixture(self, directory, fail=False):
        root = Path(directory)
        sources = root / 'fixture-source'
        sources.mkdir()
        for name in ('run_extension.py', 'prepare_runs.py', 'schedule_runs.py'):
            shutil.copyfile(Path(__file__).parent / name, sources / name)
        runner_file = sources / 'run_extension.py'
        content = runner_file.read_text(encoding='utf-8')
        fake_main = r'''
if __name__ == "__main__":
    # CPU-only synthetic fixture. No torch import or model access.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    config, frozen = validate_frozen(args.config)
    run_dir = Path(args.config).parent
    with worker_lock(run_dir / 'worker.lock'):
        launches = run_dir / 'fixture_launches.txt'
        launches.write_text(str(int(launches.read_text()) + 1 if launches.exists() else 1))
        db = connect_db(run_dir / 'results.sqlite')
        for key in ('config_sha256', 'manifest_sha256', 'source_hashes', 'protocol_hashes'):
            require_metadata(db, key, frozen[key])
        def fixture_event(kind):
            with open(config['fixture_events'], 'a') as stream:
                stream.write(json.dumps({'event': kind, 'model': args.model}) + '\n')
                stream.flush()
        fixture_event('start')
        (run_dir / 'fixture_started').write_text('started')
        time.sleep(2)
        if args.model in config.get('fixture_fail_models', []):
            fixture_event('end')
            db.close()
            sys.exit(7)
        rows = [json.loads(line) for line in (run_dir / config['manifest']).read_text().splitlines()]
        for item in remaining_rows(db, args.model, rows)[0]:
            insert_result(db, (args.model + ':' + item['id'], args.model, item['id'], 'ok', .5,
                               'SYNTHETIC CPU FIXTURE', 1, 1, 0., now(), '{}'))
        db.close()
        fixture_event('end')
'''
        old_main = 'if __name__ == "__main__":\n    main()'
        self.assertIn(old_main, content)
        runner_file.write_text(content.replace(old_main, fake_main), encoding='utf-8')
        models = [{'name': 'a', 'path': '/unused/a'}]
        if not fail:
            models.append({'name': 'b', 'path': '/unused/b'})
        config = {'seed': 1, 'batch_size': 1, 'max_input_tokens': 100, 'max_new_tokens': 100, 'models': models,
                  'fixture_events': str(root / 'events.jsonl'), 'fixture_fail_models': ['a'] if fail else []}
        (root / 'base.json').write_text(json.dumps(config), encoding='utf-8')
        (root / 'manifest.jsonl').write_text('{"id":"one","prompt":"CPU fixture"}\n', encoding='utf-8')
        (root / 'protocol.md').write_text('Synthetic CPU fixture only.', encoding='utf-8')
        queue = prepare_runs.prepare(root / 'base.json', [root / 'manifest.jsonl'], root / 'protocol.md',
                                     root / 'queue', 1, source_dir=sources)
        command = [sys.executable, str(root / 'queue/source/schedule_runs.py'), '--queue', str(root / 'queue'),
                   '--gpus', '991', '--gpu-lock-dir', str(root / 'locks'), '--poll-seconds', '.1']
        return root, queue, command

    def test_scheduler_restart_keeps_gpu_lock_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root, queue, command = self.fixture(directory)
            with (root / 'scheduler-test.log').open('w') as log:
                first = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 10
                while not list((root / 'queue/runs').glob('*/fixture_started')):
                    if first.poll() is not None or time.monotonic() > deadline:
                        self.fail('Synthetic worker failed to start: ' + (root / 'scheduler-test.log').read_text())
                    time.sleep(.05)
                first.terminate()
                first.wait(timeout=5)
                inherited = schedule_runs.try_lock(root / 'locks/gpu-991.lock')
                if inherited is not None:
                    inherited.close()
                self.assertIsNone(inherited, 'Live child must retain the GPU lock after parent termination')
                resumed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=20)
            self.assertEqual(resumed.returncode, 0, (root / 'scheduler-test.log').read_text())
            self.assertTrue((root / 'queue/merged.sqlite').exists())
            for job in queue['jobs']:
                self.assertEqual((root / 'queue' / job['run_dir'] / 'fixture_launches.txt').read_text(), '1')
            active = 0
            for line in (root / 'events.jsonl').read_text().splitlines():
                active += 1 if json.loads(line)['event'] == 'start' else -1
                self.assertIn(active, (0, 1), 'Two fake workers shared one GPU lock')
            self.assertEqual(active, 0)

    def test_nonzero_worker_remains_failed_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root, queue, command = self.fixture(directory, fail=True)
            with (root / 'scheduler-test.log').open('w') as log:
                first = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=20)
                second = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=20)
            self.assertEqual((first.returncode, second.returncode), (1, 1), (root / 'scheduler-test.log').read_text())
            run_dir = root / 'queue' / queue['jobs'][0]['run_dir']
            self.assertEqual((run_dir / 'fixture_launches.txt').read_text(), '1')
            self.assertEqual(json.loads((run_dir / 'terminal.json').read_text())['returncode'], 7)
            self.assertFalse((root / 'queue/merged.sqlite').exists())


if __name__ == '__main__':
    unittest.main()
