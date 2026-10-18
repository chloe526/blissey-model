"""Real-data packaging and transport tests; no pretrained weights or paid calls."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from archive_preflight import measure,preflight,write_archive
from package_veri import build_package
from fetch_dataset import copy_verified,validate_source,DATA_FILES,ALL_DATA_FILES,materialize,selected_files
from prepare_hosting import prepare
from training_data import load_data


@unittest.skipUnless((ROOT/'data/processed/preprocessing_config.json').exists(),'Prepared CSI required')
class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=ROOT/'data/processed'
        cls.before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.source.iterdir() if p.is_file()}
        cls.stage=build_package(ROOT,cls.source,'smoke')

    @classmethod
    def tearDownClass(cls):
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.source.iterdir() if p.is_file()}
        if cls.before!=after:raise AssertionError('Corrected data changed during packaging')
        shutil.rmtree(cls.stage)

    def test_fixture_is_four_actual_unchanged_windows_and_original_scaler(self):
        data,config=load_data(self.stage/'data/smoke',splits=('train','val'))
        original=json.loads((self.source/'preprocessing_config.json').read_text())
        self.assertTrue(config['diagnostic_subset'])
        self.assertEqual(config['mean'],original['mean']);self.assertEqual(config['std'],original['std'])
        self.assertEqual(config['packet_quality'],original['packet_quality'])
        self.assertFalse((self.stage/'data/smoke/test.npz').exists())
        for split in ('train','val'):
            self.assertEqual(len(data[split]),2);self.assertEqual(data[split].y.tolist(),[1,0])
            with np.load(self.source/f'{split}.npz',allow_pickle=False) as source:
                idx=config['subset_provenance']['source_indices'][split]
                np.testing.assert_array_equal(data[split].X,source['X'][idx])
                np.testing.assert_array_equal(data[split].y,source['y'][idx])
        with self.assertRaises(ValueError):load_data(self.stage/'data/smoke')

    def test_archive_contains_only_allowlisted_files_and_is_small(self):
        stats=preflight(self.stage);self.assertTrue(stats['passed']);self.assertLess(stats['tar_gz_bytes'],1024**2)
        buf=io.BytesIO();write_archive(self.stage,buf)
        self.assertEqual(len(buf.getvalue()),stats['tar_gz_bytes'])
        buf.seek(0)
        with tarfile.open(fileobj=buf,mode='r:gz') as archive:
            names={m.name for m in archive.getmembers() if m.isfile()}
        self.assertEqual({n for n in names if n.endswith('.npz')},{'data/smoke/train.npz','data/smoke/val.npz'})
        self.assertFalse(any('.git' in n or '.venv' in n or 'legacy' in n or 'test.npz' in n for n in names))

    def test_low_size_budget_fails(self):
        self.assertFalse(preflight(self.stage,max_mib=.1)['passed'])

    def test_tampered_or_unexpected_file_rejected(self):
        p=self.stage/'unnecessary.npz';p.write_bytes(b'x')
        try:
            with self.assertRaisesRegex(ValueError,'Unexpected file'):preflight(self.stage)
        finally:p.unlink()

    def test_code_only_full_package_has_no_arrays(self):
        stage=build_package(ROOT,self.source,'full',ROOT/'configs/dataset_source.github.example.json')
        try:
            self.assertFalse(list(stage.rglob('*.npz')));self.assertTrue(preflight(stage)['passed'])
            self.assertTrue((stage/'fetch_dataset.py').exists())
        finally:shutil.rmtree(stage)

    def test_hosting_bundle_preserves_train_val_and_config_without_test(self):
        with tempfile.TemporaryDirectory() as folder:
            target=prepare(self.source,folder)
            self.assertEqual({p.name for p in target.iterdir()},set(DATA_FILES)|{'SHA256SUMS','artifact_manifest.json'})
            for name in DATA_FILES:self.assertEqual(hashlib.sha256((target/name).read_bytes()).hexdigest(),self.before[name])
            manifest=json.loads((target/'artifact_manifest.json').read_text())
            self.assertIsNone(manifest['remote_commit_revision'])
            self.assertEqual(manifest['splits']['train']['class_counts'],[786,49])
            self.assertEqual(manifest['splits']['val']['shape'],[465,52,512])
            self.assertEqual(prepare(self.source,folder),target)

    def test_full_remote_entrypoint_fetches_only_train_val_and_saves_source(self):
        stage=build_package(ROOT,self.source,'full',ROOT/'configs/dataset_source.github.example.json')
        try:
            with tempfile.TemporaryDirectory() as folder:
                fake=Path(folder)/'python';log=Path(folder)/'calls.txt';output=Path(folder)/'output'
                fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_PYTHON_LOG"\n')
                fake.chmod(0o755)
                env=dict(os.environ,PATH=folder+os.pathsep+os.environ['PATH'],FAKE_PYTHON_LOG=str(log),VERI_OUTPUT_DIR=str(output))
                subprocess.run(['bash','scripts/remote_train.sh','--epochs','2'],cwd=stage,env=env,check=True)
                self.assertEqual(log.read_text().splitlines(),[
                    'fetch_dataset.py --source configs/dataset_source.json --output data/remote --splits train val',
                    'train.py --data-dir data/remote --epochs 2'])
                self.assertEqual((output/'dataset_source.json').read_bytes(),(stage/'configs/dataset_source.json').read_bytes())
        finally:shutil.rmtree(stage)

    def test_shell_stops_before_submission_when_over_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            fake=Path(folder)/'veri';log=Path(folder)/'calls.txt'
            fake.write_text('#!/usr/bin/env python3\nimport sys,os\nfrom pathlib import Path\nwith open(os.environ["FAKE_VERI_LOG"],"a") as f:f.write(" ".join(sys.argv[1:])+"\\n")\nif sys.argv[1:]==["run-script","--help"]:\n print("--entrypoint --gpu-type --gpu-count --base-image --requirements")\nelse:sys.exit(99)\n')
            fake.chmod(0o755)
            env=dict(os.environ,PATH=folder+os.pathsep+os.environ['PATH'],FAKE_VERI_LOG=str(log),
                PYTHON=sys.executable,VERI_MAX_UPLOAD_MIB='0.1')
            result=subprocess.run(['bash','scripts/submit_veri.sh','smoke','--execute'],cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('exceeds local size budget',result.stderr)
            self.assertEqual(log.read_text().splitlines(),['run-script --help'])


def reference(payload=b'1234',include_test=False):
    return dict(schema_version=1,preprocessing_version=2,backend='github',repo_id='chloe526/blissey-data',revision='a'*40,
        path_prefix='processed/quality-v2',files={name:dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()) for name in (ALL_DATA_FILES if include_test else DATA_FILES)})


class DatasetTransportTests(unittest.TestCase):
    def test_requires_immutable_revision_and_safe_paths(self):
        for field,value in [('revision','main'),('path_prefix','../recordings'),('repo_id','owner/..')]:
            source=reference();source[field]=value
            with self.assertRaises(ValueError):validate_source(source)

    def test_stream_hash_verification_and_mismatch(self):
        source=reference();sink=io.BytesIO()
        self.assertEqual(copy_verified(source,'train.npz',sink,lambda url:io.BytesIO(b'1234')),4)
        self.assertEqual(sink.getvalue(),b'1234')
        with self.assertRaisesRegex(ValueError,'SHA256'):copy_verified(source,'train.npz',opener=lambda url:io.BytesIO(b'5678'))
        with self.assertRaisesRegex(ValueError,'exceeds pinned size'):copy_verified(source,'train.npz',opener=lambda url:io.BytesIO(b'12345'))

    def test_default_download_verifies_only_train_val_even_with_test_manifest(self):
        urls=[]
        def opener(url):urls.append(url);return io.BytesIO(b'1234')
        materialize(reference(include_test=True),'unused',verify_only=True,opener=opener)
        self.assertEqual({u.rsplit('/',1)[1] for u in urls},set(DATA_FILES))
        self.assertFalse(any(u.endswith('/test.npz') for u in urls))

    def test_test_download_requires_explicit_selection_and_available_pin(self):
        with self.assertRaisesRegex(ValueError,'absent'):selected_files(reference(),('test',))
        self.assertEqual(selected_files(reference(include_test=True),('test',)),('preprocessing_config.json','test.npz'))
        with self.assertRaisesRegex(ValueError,'Invalid split'):selected_files(reference(),('train','train'))

    def test_github_lfs_pointer_fetches_pinned_media_bytes(self):
        urls=[]
        def opener(url):
            urls.append(url)
            return io.BytesIO(b'1234' if 'media.githubusercontent.com' in url else b'version https://git-lfs.github.com/spec/v1\n')
        copy_verified(reference(),'train.npz',opener=opener)
        self.assertEqual(len(urls),2)
        self.assertTrue(all('/'+'a'*40+'/' in url for url in urls))

if __name__=='__main__':unittest.main()
