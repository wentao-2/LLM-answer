import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from common import connect_db, parse_probability, require_metadata, after_reasoning, validate_frozen, digest, atomic_json


class DurableStorageTests(unittest.TestCase):
    def test_freeze_detects_changes_before_first_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'source').mkdir()
            (root/'source'/'run.py').write_text('print(1)', encoding='utf-8')
            (root/'manifest.jsonl').write_text('{}\n', encoding='utf-8')
            (root/'PROTOCOL.md').write_text('fixed', encoding='utf-8')
            atomic_json(root/'config.json', {'manifest':'manifest.jsonl'})
            frozen = {'config_sha256':digest(root/'config.json'),
                'manifest_sha256':digest(root/'manifest.jsonl'),
                'source_hashes':{'run.py':digest(root/'source'/'run.py')},
                'protocol_hashes':{'PROTOCOL.md':digest(root/'PROTOCOL.md')}}
            atomic_json(root/'frozen.json', frozen)
            validate_frozen(root)
            for filename in ('manifest.jsonl', 'config.json', 'source/run.py', 'PROTOCOL.md'):
                path = root/filename
                original = path.read_bytes()
                path.write_bytes(original+b' ')
                with self.assertRaises(RuntimeError):
                    validate_frozen(root)
                path.write_bytes(original)
            validate_frozen(root)
    def test_reasoning_must_finish_before_final_answer(self):
        self.assertIsNone(after_reasoning([1,2,8], [8,9]))
        self.assertEqual(after_reasoning([1,2,8,9,4,5], [8,9]), [4,5])
        self.assertIsNone(after_reasoning([1,2,4,5], [8,9]))
    def test_strict_output(self):
        self.assertEqual(parse_probability('{"p_h1":0.8}'), .8)
        for value in ('{"p_h1":80}', '{"p_h1":true}', '{"p_h1":NaN}',
                      'reasoning {"p_h1":0.8}', '{"p_h1":"0.8"}', '{"p_h1":0.8,"other":1}'):
            self.assertIsNone(parse_probability(value))

    def test_resume_preserves_saved_and_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.sqlite'
            db = connect_db(path)
            require_metadata(db, 'config', {'a': 1})
            db.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       ('m:a','m','a','ok',.8,'{"p_h1":0.8}',10,5,.5,'now','{}'))
            db.commit()
            db.close()
            db = connect_db(path)
            self.assertEqual(db.execute('SELECT item_id FROM results').fetchall(), [('a',)])
            require_metadata(db, 'config', {'a': 1})
            with self.assertRaises(RuntimeError):
                require_metadata(db, 'config', {'a': 2})
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('INSERT INTO results SELECT * FROM results')
            db.close()


if __name__ == '__main__':
    unittest.main()
