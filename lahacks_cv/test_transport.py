"""Submission transport checks only: no preprocessing or model training."""
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from .transport import transfer, verify_files


def fixture(payload=b'CSI source fixture'):
    name = 'session_03_LAHACKS/staged_fall/trial/measurements.jsonl'
    source = dict(repository='chloe526/blissey-data', revision='a' * 40,
                  files=[dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())])
    return source, name, payload


class TransportTests(unittest.TestCase):
    def test_verified_local_copy_is_rehashed_without_download_or_duplication(self):
        source, name, payload = fixture()
        with tempfile.TemporaryDirectory() as td, patch('urllib.request.urlopen') as network:
            root = Path(td) / 'existing'; p = root / name
            p.parent.mkdir(parents=True); p.write_bytes(payload)
            cache = Path(td) / 'cache'
            result = verify_files(source, cache, [root], False, 1., 1.)
            network.assert_not_called()
            self.assertEqual((result['local_files'], result['downloaded_bytes']), (1, 0))
            self.assertFalse(cache.exists())

    def test_offline_missing_and_corrupt_files_fail_without_network(self):
        source, name, payload = fixture()
        with tempfile.TemporaryDirectory() as td, patch('urllib.request.urlopen') as network:
            with self.assertRaises(FileNotFoundError):
                verify_files(source, td, [], True, 1., 1.)
            p = Path(td) / name; p.parent.mkdir(parents=True); p.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Local SHA-256 mismatch'):
                verify_files(source, td, [], False, 1., 1.)
            network.assert_not_called()

    def test_download_hash_and_partial_cleanup(self):
        source, name, payload = fixture()
        with tempfile.TemporaryDirectory() as td, patch('urllib.request.urlopen', return_value=io.BytesIO(payload)) as network:
            result = verify_files(source, td, [], False, 2., 2.)
            self.assertEqual(result['downloaded_files'], 1)
            self.assertEqual((Path(td) / name).read_bytes(), payload)
            self.assertEqual(network.call_args.kwargs['timeout'], 2.)
        with tempfile.TemporaryDirectory() as td, patch('urllib.request.urlopen', return_value=io.BytesIO(b'bad')):
            with self.assertRaisesRegex(RuntimeError, 'Remote SHA-256 mismatch'):
                verify_files(source, td, [], False, 2., 2.)
            self.assertFalse((Path(td) / name).exists())
            self.assertFalse(list(Path(td).rglob('*.part')))

    def test_hard_deadline_terminates_stuck_subprocess(self):
        source, _, _ = fixture()
        original = subprocess.run
        def stalled_worker(command, **kwargs):
            return original([sys.executable, '-c', 'import time; time.sleep(10)'], **kwargs)
        with patch('lahacks_cv.transport.subprocess.run', side_effect=stalled_worker):
            with self.assertRaisesRegex(TimeoutError, 'no Veri submission attempted'):
                transfer(source, timeout=.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
