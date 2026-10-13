"""Check hostile archive paths and evidence tampering at the public boundary."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from release_utils import acquire_archive, extract_new_archive, safe_member, verify_archive


class ReleaseBoundaryTests(unittest.TestCase):
    def test_bundled_evidence_works_offline_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundled = root / 'evidence' / 'saved.zip'
            bundled.parent.mkdir()
            bundled.write_bytes(b'exact saved evidence')
            metadata = {'file': bundled.name, 'bytes': bundled.stat().st_size,
                        'sha256': hashlib.sha256(bundled.read_bytes()).hexdigest()}
            with patch('release_utils.ROOT', root), patch('release_utils.urllib.request.urlopen') as network:
                result = acquire_archive(metadata, root / 'cache', offline=True)
            self.assertEqual(result.read_bytes(), bundled.read_bytes())
            network.assert_not_called()

    def test_corrupt_bundled_evidence_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundled = root / 'evidence' / 'saved.zip'
            bundled.parent.mkdir()
            bundled.write_bytes(b'modified')
            metadata = {'file': bundled.name, 'bytes': 8,
                        'sha256': hashlib.sha256(b'original').hexdigest()}
            with patch('release_utils.ROOT', root), self.assertRaises(ValueError):
                acquire_archive(metadata, root / 'cache', offline=True)
            self.assertFalse((root / 'cache' / bundled.name).exists())

    def test_escaping_paths_are_rejected(self):
        for name in ['../output', '/absolute', 'C:/private', 'folder\\escape', 'a/../../b', 'a//b']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_member(name)

    def test_byte_equal_size_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'evidence.zip'
            path.write_bytes(b'original')
            metadata = {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}
            verify_archive(path, metadata)
            path.write_bytes(b'modified')
            with self.assertRaises(ValueError):
                verify_archive(path, metadata)

    def test_malicious_zip_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / 'unsafe.zip'
            output = Path(temporary) / 'output'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../outside', 'data')
            with self.assertRaises(ValueError):
                extract_new_archive(archive, output)
            self.assertFalse(output.exists())

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / 'safe.zip'
            output = Path(temporary) / 'output'
            output.mkdir()
            retained = output / 'retained'
            retained.write_text('original', encoding='utf-8')
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('root/new', 'data')
            with self.assertRaises(FileExistsError):
                extract_new_archive(archive, output)
            self.assertEqual(retained.read_text(encoding='utf-8'), 'original')


if __name__ == '__main__':
    unittest.main()
