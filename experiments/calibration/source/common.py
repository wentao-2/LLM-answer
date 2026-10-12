"""Durable, immutable run storage. Standard library only."""
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def connect_db(path):
    db = sqlite3.connect(str(path), timeout=60)
    # DELETE journaling is intentional: /workspace is shared network storage.
    # WAL's shared-memory assumptions are inappropriate across network hosts.
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('''CREATE TABLE IF NOT EXISTS results (
      id TEXT PRIMARY KEY, model TEXT NOT NULL, item_id TEXT NOT NULL,
      status TEXT NOT NULL, probability REAL, raw_output TEXT NOT NULL,
      input_tokens INTEGER, output_tokens INTEGER, duration_sec REAL,
      created_at TEXT NOT NULL, metadata_json TEXT NOT NULL)''')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS model_item ON results(model,item_id)')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.commit()
    return db


def require_metadata(db, key, value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
    old = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
    if old and old[0] != encoded:
        raise RuntimeError(f'Immutable run metadata changed: {key}; create a new run.')
    if not old:
        db.execute('INSERT INTO metadata VALUES (?,?)', (key, encoded))
        db.commit()


def parse_probability(text):
    import math
    import re
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.S)
        if match:
            text = match.group(1)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {'p_h1'}:
        return None
    probability = value['p_h1']
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        return None
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        return None
    return float(probability)


def after_reasoning(token_ids, close_ids):
    """Return final tokens only after a complete reasoning-close marker."""
    if not close_ids:
        raise ValueError('Empty reasoning-close marker')
    ends = [i + len(close_ids) for i in range(len(token_ids) - len(close_ids) + 1)
            if token_ids[i:i + len(close_ids)] == close_ids]
    return token_ids[ends[-1]:] if ends else None


def validate_frozen(run_dir):
    """Check the recorded freeze before the first inference or any resume."""
    run_dir = Path(run_dir)
    frozen = json.loads((run_dir / 'frozen.json').read_text(encoding='utf-8'))
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    actual = {
        'config_sha256': digest(run_dir / 'config.json'),
        'manifest_sha256': digest(run_dir / config['manifest']),
        'source_hashes': {p.name: digest(p) for p in sorted((run_dir / 'source').glob('*.py'))},
    }
    for key, value in actual.items():
        if value != frozen.get(key):
            raise RuntimeError(f'Frozen artifact mismatch: {key}; refusing inference.')
    for name, expected in frozen.get('protocol_hashes', {}).items():
        if digest(run_dir / name) != expected:
            raise RuntimeError(f'Frozen protocol mismatch: {name}; refusing inference.')
    return config


def parse_final_probability(text):
    """Predeclared R3 readout: one terminal marker, EOS checked by caller.

    Rationale may precede the final line, but multiple markers, malformed JSON,
    additional final prose, duplicate keys and out-of-range values are invalid.
    No probabilities are recovered from reasoning or truncated generations.
    """
    import re
    if not isinstance(text, str) or text.count('FINAL_JSON:') != 1:
        return None
    match = re.search(r'(?:^|\n)[ \t]*FINAL_JSON:[ \t]*(\{[^\r\n]*\})[ \t]*$', text.strip())
    if not match:
        return None
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(match.group(1), object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    return parse_probability(json.dumps(value))
