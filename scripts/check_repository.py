"""Check public source identities, figure inputs and Python syntax."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

from release_utils import ROOT, load_manifest, sha256


def check():
    ledger = json.loads((ROOT / 'provenance/source_files.json').read_text(encoding='utf-8'))
    files = ledger['files']
    for name, item in files.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('Copied source identity differs: ' + name)
    python_files = sorted(p for parent in ['experiments', 'analysis/scripts', 'scripts', 'tests']
                          for p in (ROOT / parent).rglob('*.py') if p.is_file())
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.relative_to(ROOT).as_posix())
    figure_checks = []
    for stem in ['dependence_diagnosis', 'observation_control', 'dependence_full', 'reminder_full']:
        provenance = json.loads((ROOT / 'analysis/figures' / (stem + '.data.json')).read_text(encoding='utf-8'))
        script = 'plot_followup.py' if stem in {'dependence_diagnosis', 'observation_control'} else 'plot_dependence_appendix.py'
        rendering = provenance.get('rendering', provenance)
        source_digest = rendering.get('source_sha256', provenance.get('script_sha256'))
        if sha256(ROOT / 'analysis/scripts' / script) != source_digest:
            raise ValueError('Figure source hash differs: ' + stem)
        if sha256(ROOT / 'analysis/data/followup/evidence.json') != provenance['input_sha256']:
            raise ValueError('Figure input hash differs: ' + stem)
        figure_checks.append({'figure': stem, 'points': len(provenance['points'])})
    archives = load_manifest()
    if set(archives['packages']) != {'extension', 'wording-history', 'dependence-followup'}:
        raise ValueError('Evidence manifest membership differs')
    return {'state': 'passed', 'copied_files': len(files), 'python_files_parsed': len(python_files),
            'figure_sources_and_inputs_match': figure_checks, 'packages': list(archives['packages'])}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
