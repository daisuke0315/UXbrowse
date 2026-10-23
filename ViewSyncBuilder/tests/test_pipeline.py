import tempfile
import unittest
from pathlib import Path
import numpy as np
import pycolmap as pc

from test_bundle import fixture
from viewsync_builder.pipeline import export_result, feature_fingerprint, settings, start_run, seed_features
from viewsync_builder.project import Project
from viewsync_builder.ct import CTVolume
from test_ct import write_series
from PIL import Image


class PipelineTests(unittest.TestCase):
    def test_cached_features_keep_their_uv_descriptor_and_new_database_image_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new('RGB', (32, 24), 'red').save(root/'a.png')
            Image.new('RGB', (32, 24), 'blue').save(root/'b.png')
            uv = np.array([[2, 3], [4, 5]], np.float32)
            descriptors = np.ones((2, 128), np.float32)/np.sqrt(128)
            np.savez_compressed(root/'cached.npz', keypoints=uv, descriptors=descriptors)
            hits = seed_features(root/'features.db', root, ['a.png', 'b.png'], settings({}),
                                 {'b.png': root/'cached.npz'})
            self.assertEqual(hits, 1)
            with pc.Database.open(root/'features.db') as db:
                image = db.read_image_with_name('b.png')
                np.testing.assert_array_equal(db.read_keypoints(image.image_id), uv)
                np.testing.assert_allclose(db.read_descriptors(image.image_id).to_float().data, descriptors)
                self.assertFalse(db.exists_keypoints(db.read_image_with_name('a.png').image_id))

    def test_feature_cache_does_not_depend_on_gcp_or_pairing(self):
        images = [dict(id='a', normalized_sha256='abc', mask=None)]
        first = settings({})
        second = settings(dict(overlap=20, refine_intrinsics=True))
        self.assertEqual(feature_fingerprint(images, first, 'engine', 'model'),
                         feature_fingerprint(images, second, 'engine', 'model'))
        second['max_features'] = 3000
        self.assertNotEqual(feature_fingerprint(images, first, 'engine', 'model'),
                            feature_fingerprint(images, second, 'engine', 'model'))

    def test_untriangulated_feature_never_gets_xyz(self):
        model, controls = fixture()
        image = next(iter(model.images.values()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_dir = root/'features'
            feature_dir.mkdir()
            uv = np.array([p.xy for p in image.points2D])
            np.savez_compressed(feature_dir/(str(image.image_id)+'.npz'),
                                uv=uv, descriptors=np.zeros((len(uv), 128), np.float32))
            export_result(model, feature_dir, root/'output', controls, {})
            with np.load(root/'output/feature_map'/f'{image.image_id}.npz') as archive:
                valid = archive['validity']
                self.assertGreater((~valid).sum(), 0)
                self.assertTrue(np.isnan(archive['xyz'][~valid]).all())
                self.assertTrue((archive['point3d_ids'][~valid] == -1).all())
                self.assertTrue(np.isfinite(archive['xyz'][valid]).all())
            self.assertTrue((root/'output/sparse/cameras.bin').is_file())

    def test_settings_reject_unknown_or_invalid_values(self):
        for value in ({'pairing': 'pose_known'}, {'max_features': -1}, {'typo': 10},
                      {'coordinate_mode':'guess'}, {'surface_max_size':-1}, {'surface':'fake'}):
            with self.assertRaises(ValueError):
                settings(value)

    def test_photo_mesh_preview_needs_no_gcp_but_ct_mesh_does(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);project=Project.create(root/'project','preview')
            for index,color in enumerate(('red','blue')):
                path=root/f'{index}.png';Image.new('RGB',(32,24),color).save(path);project.import_image(path)
            run=start_run(project,{'coordinate_mode':'sfm_preview'},'texture')
            self.assertEqual(run['settings']['coordinate_mode'],'sfm_preview')
            with self.assertRaisesRegex(ValueError,'GCP'):
                start_run(project,{'coordinate_mode':'ct'},'texture')

    def test_shared_camera_rejects_mixed_image_dimensions_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = Project.create(root/'project', 'mixed')
            for index, size in enumerate(((32, 24), (40, 30))):
                source = root/f'{index}.png'
                Image.new('RGB', size, (index*50, 0, 0)).save(source)
                project.import_image(source)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                start_run(project, dict(single_camera=True), 'sfm')

    def test_run_recomputes_sampling_from_snapshot_ct_and_keeps_interpolation_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_series(root/'dicom')
            project = Project.create(root/'project', 'sampling')
            for index,color in enumerate(('red','blue')):
                source=root/f'{index}.png'
                Image.new('RGB',(32,24),color).save(source)
                project.import_image(source)
            project.manifest['ct'] = CTVolume.load(root/'dicom').metadata
            project.set_controls([dict(id='test',role='evaluation',xyz=[2,2,2],observations=[],
                                       enabled=False,ct_sampling={'status':'original'})])
            run = start_run(project, {}, 'sfm')
            self.assertEqual(run['controls'][0]['ct_sampling']['status'],'interpolated')
            self.assertEqual(len(run['ct']['regularization']['interpolated_slices']),1)
