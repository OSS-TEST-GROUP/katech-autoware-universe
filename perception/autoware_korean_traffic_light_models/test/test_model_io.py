import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from autoware_korean_traffic_light_models import model_io


class Response:
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def geturl(self): return 'https://release-assets.githubusercontent.com/test'
    def read(self, size):
        data, self.data = self.data[:size], self.data[size:]
        return data


class ModelIO(unittest.TestCase):
    def row(self, data=b'model'):
        return dict(file='test.onnx', bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    url='https://github.com/OSS-TEST-GROUP/katech-autoware-sils/releases/download/test/test.onnx')

    def test_manifest_contract(self):
        rows = model_io.manifest()['files']
        self.assertEqual({(r['kind'], r['batch']) for r in rows},
                         {('detector', 1), ('detector', 6), ('classifier', 1), ('classifier', 6)})
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r['bytes'] for r in rows), 89468762)

    def test_download_and_cached_verify(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(model_io.urllib.request, 'urlopen', return_value=Response(b'model')) as fetch:
                path = model_io.download(folder, self.row())
                self.assertEqual(path.read_bytes(), b'model')
                model_io.download(folder, self.row())
                self.assertEqual(fetch.call_count, 1)
            self.assertEqual(list(Path(folder).glob('*.part')), [])

    def test_mismatch_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.onnx'
            path.write_bytes(b'other')
            with patch.object(model_io.urllib.request, 'urlopen') as fetch:
                with self.assertRaises(ValueError): model_io.download(folder, self.row())
                fetch.assert_not_called()
            self.assertEqual(path.read_bytes(), b'other')

    def test_rejects_bad_download(self):
        for data in (b'wrong', b'too large', b''):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as folder:
                with patch.object(model_io.urllib.request, 'urlopen', return_value=Response(data)):
                    with self.assertRaises(ValueError): model_io.download(folder, self.row())
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_rejects_path_and_url(self):
        for row in ({**self.row(), 'file': '../test.onnx'}, {**self.row(), 'url': 'http://example.com/file'}):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError): model_io.download(folder, row)

    def test_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'test.onnx'
            target.symlink_to(Path(folder) / 'missing')
            with self.assertRaises(ValueError): model_io.download(folder, self.row())

    def test_launch_is_opt_in_and_no_production_output(self):
        source = (Path(__file__).parents[1] / 'launch/candidate.launch.py').read_text()
        self.assertIn("'enable_candidate': 'false'", source)
        self.assertIn("'cuda_preprocess': False", source)
        self.assertNotIn('/classification/car/traffic_signals', source)
        self.assertIn("'/korean_tl/candidate/signals'", source)


if __name__ == '__main__': unittest.main()
