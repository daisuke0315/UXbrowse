import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient

from viewsync_builder.server import create_app
from test_ct import write_series


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.client = TestClient(create_app(self.root/'settings'), base_url='http://127.0.0.1')
        self.token = self.client.get('/api/session').json()['token']
        self.headers = {'X-ViewSync-Token': self.token}

    def test_mutations_need_local_session_token_and_host(self):
        body = dict(path=str(self.root/'project'), name='test', mode='create')
        self.assertEqual(self.client.post('/api/project', json=body).status_code, 403)
        result = self.client.post('/api/project', json=body, headers=self.headers)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.client.get('/api/session', headers={'Host': 'evil.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/points', json=[], headers={**self.headers, 'Origin':'https://evil.example'}).status_code, 403)

    def test_missing_gcp_prevents_full_run_and_no_data_is_overwritten(self):
        body = dict(path=str(self.root/'project'), name='test', mode='create')
        self.client.post('/api/project', json=body, headers=self.headers)
        result = self.client.post('/api/jobs', json=dict(settings={}, stop_after='export'), headers=self.headers)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.client.post('/api/project', json=body, headers=self.headers).status_code, 400)
        self.assertTrue((self.root/'project/manifest.json').is_file())

    def test_project_path_traversal_is_rejected(self):
        self.client.post('/api/project', json=dict(path=str(self.root/'project'), name='test', mode='create'), headers=self.headers)
        result = self.client.get('/api/file', params={'path':'../../secret.txt'})
        self.assertEqual(result.status_code, 400)

    def test_ct_choices_distinguish_equal_size_series(self):
        groups = {f'1.2.{n}': [(self.root/f'{n}.dcm', SimpleNamespace(
            SeriesNumber=n, SeriesDescription='same protocol'))] for n in (3, 12)}
        with patch('viewsync_builder.server.CTVolume.series', return_value=groups):
            response = self.client.post('/api/ct/series', json={'path':str(self.root)}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), [dict(series_uid=f'1.2.{n}', slices=1,
            series_number=str(n), description='same protocol') for n in (3, 12)])

    def test_import_fills_gaps_and_persists_point_provenance_across_reopen(self):
        write_series(self.root/'dicom')
        body = dict(path=str(self.root/'project'), name='test', mode='create')
        self.client.post('/api/project', json=body, headers=self.headers)
        response = self.client.post('/api/ct', json={'path':str(self.root/'dicom')}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        metadata = response.json()['metadata']
        self.assertEqual(metadata['dimensions'], [5, 4, 5])
        self.assertEqual(len(metadata['regularization']['interpolated_slices']), 1)
        points = [dict(id='P1',role='evaluation',xyz=[2, 2, 2],observations=[],enabled=True,
                       ct_sampling={'status':'original'})]
        response = self.client.post('/api/points', json=points, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()[0]['ct_sampling']['status'], 'interpolated')
        self.client.post('/api/project', json={**body,'mode':'open'}, headers=self.headers)
        self.assertEqual(self.client.get('/api/ct').json()['metadata']['regularization'],metadata['regularization'])
        self.assertEqual(self.client.get('/api/points').json()[0]['ct_sampling']['status'], 'interpolated')
