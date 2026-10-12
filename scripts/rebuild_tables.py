"""Regenerate scientific LaTeX tables without changing the released exports."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from release_utils import ROOT, sha256


def render_followup_tables(script, data_path):
    """Use the frozen table renderer with the released unrounded exports."""
    spec = importlib.util.spec_from_file_location('frozen_table_export', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = json.loads(data_path.read_text(encoding='utf-8'))
    module.tables(data)
    module.supplementary_tables(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.work/tables')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = (ROOT / 'analysis').resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error('Output must be separate from the released analysis files.')
    if output.exists():
        parser.error('Use a new output directory to preserve previous tables.')
    output.mkdir(parents=True)
    (output / 'sections').mkdir()
    (output / 'build').mkdir()
    shutil.copytree(source / 'data', output / 'data')
    shutil.copytree(source / 'scripts', output / 'scripts')
    subprocess.run([sys.executable, '-B', str(output / 'scripts/build_review_tables.py')], cwd=output, check=True)
    render_followup_tables(output / 'scripts/export_followup.py', output / 'data/followup/evidence.json')
    generated = sorted((output / 'sections').glob('*_generated.tex'))
    if not generated:
        raise ValueError('No tables were generated')
    for path in generated:
        original = source / 'sections' / path.name
        if not original.is_file() or original.read_bytes().replace(b'\r\n', b'\n') != path.read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('Regenerated table differs: ' + path.name)
    report = {'state': 'passed', 'tables': {p.name: sha256(p) for p in generated},
              'identical_after_operating_system_line_endings': True,
              'latest_numerical_input_sha256': sha256(source / 'data/followup/evidence.json')}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
