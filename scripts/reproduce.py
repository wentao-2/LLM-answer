"""Download exact released evidence and rerun its offline scientific verifiers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from release_utils import ROOT, acquire_archive, load_manifest, package_root, sha256

VERIFIERS = {'extension': 'extension_release.py', 'wording-history': 'verify_supplement.py',
             'dependence-followup': 'followup_release.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', *VERIFIERS], default='all')
    parser.add_argument('--archives-dir', type=Path, default=ROOT / '.work/archives')
    parser.add_argument('--work-dir', type=Path, default=ROOT / '.work/offline')
    parser.add_argument('--offline', action='store_true', help='Require already downloaded archives.')
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Offline package verification requires Python 3.12 or newer.')
    manifest = load_manifest()
    keys = list(VERIFIERS) if args.suite == 'all' else [args.suite]
    reports = []
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.work_dir / 'reports'
    report_dir.mkdir(exist_ok=True)
    for key in keys:
        metadata = manifest['packages'][key]
        archive = acquire_archive(metadata, args.archives_dir, args.offline)
        root = package_root(key, archive, metadata, args.work_dir)
        verifier = root / VERIFIERS[key]
        if not verifier.is_file():
            raise FileNotFoundError('Missing frozen package verifier: ' + verifier.name)
        print('Verifying and recomputing ' + key, flush=True)
        completed = subprocess.run([sys.executable, '-B', str(verifier), '--verify-package', str(root)],
                                   cwd=root, capture_output=True, text=True, encoding='utf-8')
        (report_dir / (key + '.stdout.txt')).write_text(completed.stdout, encoding='utf-8')
        (report_dir / (key + '.stderr.txt')).write_text(completed.stderr, encoding='utf-8')
        if completed.returncode:
            raise RuntimeError(f'{key} verification failed. Inspect {report_dir}.')
        # Verifiers report their scientific checks, with original missing values retained.
        reports.append({'suite': key, 'state': 'passed', 'archive_sha256': metadata['sha256'],
                        'verifier_sha256': sha256(verifier), 'returncode': completed.returncode})
        print(key + ': passed', flush=True)
    report = {'state': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'python_version': sys.version.split()[0], 'new_model_inference': False,
              'suites': reports}
    (report_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
