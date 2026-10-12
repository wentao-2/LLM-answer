"""Render the four current dependence figures in a separate workspace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from release_utils import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.work/plots')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = (ROOT / 'analysis').resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error('Output must be separate from the released analysis files.')
    if output.exists():
        parser.error('Use a new output directory to preserve previous exports.')
    output.mkdir(parents=True)
    (output / 'figures').mkdir()
    (output / 'scripts').mkdir()
    shutil.copytree(source / 'data', output / 'data')
    scripts = ['plot_followup.py', 'plot_dependence_appendix.py']
    for name in scripts:
        shutil.copyfile(source / 'scripts' / name, output / 'scripts' / name)
        subprocess.run([sys.executable, '-B', str(output / 'scripts' / name)], cwd=output, check=True)
    names = ['dependence_diagnosis', 'observation_control', 'dependence_full', 'reminder_full']
    point_checks = []
    for name in names:
        original = json.loads((source / 'figures' / (name + '.data.json')).read_text(encoding='utf-8'))
        regenerated = json.loads((output / 'figures' / (name + '.data.json')).read_text(encoding='utf-8'))
        if original['points'] != regenerated['points']:
            raise ValueError('Regenerated scientific points differ: ' + name)
        if original['input_sha256'] != regenerated['input_sha256']:
            raise ValueError('Regenerated input identity differs: ' + name)
        point_checks.append({'figure': name, 'points': len(original['points']), 'points_identical': True})
    report = {'state': 'passed', 'figures': point_checks,
              'output_files': {p.name: sha256(p) for p in (output / 'figures').iterdir() if p.is_file()}}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'state': 'passed', 'figures': point_checks}, indent=2))


if __name__ == '__main__':
    main()
