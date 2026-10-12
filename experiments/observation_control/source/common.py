from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
PHI = 'phi4-standard-r6'
QWEN = 'qwen3-8b-r6'


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def text_hash(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s]


def frozen_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode('utf-8')
    if path.exists():
        assert path.read_bytes() == data, 'Refusing to replace frozen content: '+str(path)
        return
    with path.open('xb') as stream:
        stream.write(data)


def write_json(path, data):
    frozen_write(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_jsonl(path, rows):
    frozen_write(path, ''.join(canonical(r)+'\n' for r in rows))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def read_db(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=60)
    db.row_factory = sqlite3.Row
    try:
        return [dict(x) for x in db.execute('SELECT * FROM results ORDER BY id')]
    finally:
        db.close()


def validate_seal(root=ROOT):
    seal = load(root/'DESIGN_SEAL.json')
    for rel, expected in seal['files'].items():
        assert sha(root/rel) == expected, 'Design SHA mismatch: '+rel
    return seal
