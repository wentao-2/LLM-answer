"""Download two preselected public model revisions without altering old checkpoints."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
from huggingface_hub import HfApi, snapshot_download

MODELS = [
    ('microsoft/Phi-4-reasoning', '1de18ec97600877ce63dbf60c73b998da99f0195'),
    ('mistralai/Mistral-Nemo-Instruct-2407', '04d8a90549d23fc6bd7f642064003592df51e9b3'),
]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    api = HfApi(token=False)
    for repo, revision in MODELS:
        target = Path('/workspace/models') / repo
        target.mkdir(parents=True, exist_ok=True)
        info = api.model_info(repo, revision=revision, files_metadata=True)
        assert info.sha == revision
        names = [s.rfilename for s in info.siblings if s.rfilename.endswith(('.json', '.model', '.md'))
                 or s.rfilename.startswith('model-') and s.rfilename.endswith('.safetensors')]
        print(json.dumps({'event': 'download', 'repo': repo, 'revision': revision, 'files': names}), flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(target),
                          allow_patterns=names, max_workers=4, token=False)
        records = []
        for s in info.siblings:
            if s.rfilename not in names:
                continue
            file = target / s.rfilename
            assert file.is_file()
            actual = sha(file)
            expected = s.lfs.sha256 if s.lfs else None
            if expected:
                assert actual == expected, s.rfilename
            if s.size is not None:
                assert file.stat().st_size == s.size
            records.append({'file': s.rfilename, 'bytes': file.stat().st_size,
                            'sha256': actual, 'upstream_lfs_sha256': expected, 'git_blob_id': s.blob_id})
        provenance = {'repo_id': repo, 'revision': revision,
                      'source_url': 'https://huggingface.co/' + repo,
                      'downloaded_at': datetime.now(timezone.utc).isoformat(),
                      'files': records, 'weight_hashes_verified': True}
        (target / 'MODEL_PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'event': 'verified', 'repo': repo, 'files': len(records)}), flush=True)

if __name__ == '__main__':
    main()
