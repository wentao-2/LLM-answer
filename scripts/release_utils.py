"""Standard-library helpers for immutable evidence downloads and extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_manifest():
    return json.loads((ROOT / 'reproducibility/archives.json').read_text(encoding='utf-8'))


def verify_archive(path, metadata):
    path = Path(path)
    if path.stat().st_size != metadata['bytes']:
        raise ValueError('Archive byte count differs: ' + path.name)
    if sha256(path) != metadata['sha256']:
        raise ValueError('Archive SHA256 differs: ' + path.name)


def acquire_archive(metadata, directory, offline=False):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    name = metadata['file']
    if PurePosixPath(name).name != name or not name.endswith('.zip'):
        raise ValueError('Invalid archive filename')
    target = directory / name
    if target.exists():
        verify_archive(target, metadata)
        return target
    bundled = ROOT / 'evidence' / name
    if bundled.is_file():
        verify_archive(bundled, metadata)
        partial = target.with_suffix('.zip.part')
        shutil.copyfile(bundled, partial)
        verify_archive(partial, metadata)
        partial.replace(target)
        return target
    if offline:
        raise FileNotFoundError('Offline archive is absent: ' + str(target))
    allowed = 'https://raw.githubusercontent.com/wentao-2/LLM-answer/main/evidence/'
    if metadata['url'] != allowed + name:
        raise ValueError('Archive URL differs from the documented GitHub file path')
    partial = target.with_suffix('.zip.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= metadata['bytes']:
        verify_archive(partial, metadata)
        partial.replace(target)
        return target
    headers = {'User-Agent': 'LLM-answer-reproduction'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(metadata['url'], headers=headers)
    print('Downloading ' + name, flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.status
        if offset and status == 206:
            content_range = response.headers.get('Content-Range', '')
            if not content_range.startswith(f'bytes {offset}-') or not content_range.endswith('/' + str(metadata['bytes'])):
                raise ValueError('Unexpected HTTP range')
            mode = 'ab'
        elif status == 200:
            mode = 'wb'
        else:
            raise ValueError('Unexpected download status: ' + str(status))
        with partial.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    verify_archive(partial, metadata)
    partial.replace(target)
    return target


def safe_member(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or path.is_absolute() or '..' in path.parts
            or any(':' in part for part in path.parts) or path.as_posix() != name):
        raise ValueError('Unsafe ZIP member path')
    return path


def extract_new_archive(archive_path, destination):
    """Extract into a new directory, rejecting escaping paths and links."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Extraction requires a new directory: ' + str(destination))
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        paths = [safe_member(info.filename) for info in members]
        keys = [p.as_posix().casefold() for p in paths]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate ZIP member paths')
        if any(stat.S_ISLNK(info.external_attr >> 16) for info in members):
            raise ValueError('ZIP symbolic links are not accepted')
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
        destination.mkdir(parents=True)
        for info, relative in zip(members, paths):
            output = destination / Path(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    top = {p.parts[0] for p in paths}
    if len(top) == 1:
        candidate = destination / next(iter(top))
        if candidate.is_dir():
            return candidate
    return destination


def package_root(key, archive, metadata, work_dir):
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    receipt = work_dir / (key + '.extraction.json')
    destination = work_dir / key
    if receipt.exists():
        value = json.loads(receipt.read_text(encoding='utf-8'))
        if value['archive_sha256'] != metadata['sha256']:
            raise ValueError('Existing extraction belongs to a different archive')
        relative = safe_member(value['package_root'])
        root = work_dir / Path(*relative.parts)
        if not root.is_dir():
            raise FileNotFoundError('Recorded extracted package is missing')
        return root
    root = extract_new_archive(archive, destination)
    receipt.write_text(json.dumps({'archive_sha256': metadata['sha256'],
        'package_root': root.relative_to(work_dir).as_posix()}, indent=2) + '\n', encoding='utf-8')
    return root
