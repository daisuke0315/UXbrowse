import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from view_retrieval import ViewRetrievalIndex, crop_box, object_bbox, square_rgb
from object_roi import reference_rgb_mask, bbox_from_patch_mask
from generate_3dgs_reference_views import camera_positions, look_at_opencv


class MeanColorEncoder:
    model_name = "fake-model"

    def encode(self, image):
        value = image.mean(axis=(0, 1)).astype(np.float32)
        return value / np.linalg.norm(value)


class ViewRetrievalTests(unittest.TestCase):
    def analysis_index(self):
        encoder=MeanColorEncoder()
        encoder.encode=Mock(wraps=encoder.encode)
        with patch.object(ViewRetrievalIndex,'_build_index',return_value=np.array([[1.,0.,0.]])):
            index=ViewRetrievalIndex(Path('.'),[],mode='dino-global',encoder=encoder)
        index.reference_preview=Mock(return_value=None)
        index.views=[dict(image_name='reference.png')]
        index.auto_roi=True
        index.crop=True
        bbox=(.25,.2,.5,.6)
        index.detector=Mock()
        index.detector.detect.return_value=(bbox,dict(mask=np.ones((16,16),np.uint8)))
        return index,encoder,bbox

    def test_photo_analysis_resizes_only_retrieval_and_preserves_normalized_roi(self):
        index,encoder,bbox=self.analysis_index()
        index.query_max_width=1280
        image=np.full((1200,1600,3),80,np.uint8)
        original=image.copy()
        ranked,_,_,debug=index.rank_debug(image,top_k=1)
        self.assertEqual(ranked[0][2],0)
        self.assertEqual(index.detector.detect.call_args.args[0].shape,(960,1280,3))
        self.assertEqual(encoder.encode.call_args.args[0].shape,(576,640,3))
        self.assertEqual(debug['bbox'],bbox)
        self.assertEqual(crop_box(image,debug['bbox']).shape,(720,800,3))
        # The detector-only route used by fixed-reference validation is identical.
        fixed_bbox,_=index.detect_query(image)
        self.assertEqual(fixed_bbox,bbox)
        self.assertEqual(index.detector.detect.call_args.args[0].shape,(960,1280,3))
        np.testing.assert_array_equal(image,original)

    def test_analysis_does_not_upscale_small_images_or_change_live_default(self):
        index,encoder,bbox=self.analysis_index()
        large=np.full((1200,1600,3),80,np.uint8)
        index.rank(large,top_k=1)
        self.assertIs(index.detector.detect.call_args.args[0],large)
        index.query_max_width=1280
        for shape in ((720,960,3),(1800,900,3)):
            image=np.full(shape,80,np.uint8)
            index.rank_debug(image,top_k=1)
            self.assertIs(index.detector.detect.call_args.args[0],image)
            index.detect_query(image)
            self.assertIs(index.detector.detect.call_args.args[0],image)

    def test_crop_and_object_bbox(self):
        image = np.zeros((10, 20, 3), np.uint8)
        self.assertEqual(crop_box(image, (.25, .2, .5, .6)).shape, (6, 10, 3))
        mask = np.zeros((10, 20), bool)
        mask[2:8, 5:15] = True
        self.assertEqual(object_bbox(mask, margin=0), (.25, .2, .5, .6))
        self.assertEqual(square_rgb(image).shape, (224, 224, 3))

    def test_embedding_ranks_and_excludes_without_reordering_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = []
            for index, color in enumerate(((200, 20, 20), (20, 200, 20))):
                image = np.full((20, 30, 3), 255, np.uint8)
                image[3:17, 5:25] = color
                cv2.imwrite(str(root / f"view_{index}.png"), image)
                np.savez(root / f"view_{index}.npz", valid=np.ones((20, 30), bool))
                views.append(dict(image_name=f"view_{index}.png",
                                  data_name=f"view_{index}.npz"))
            index = ViewRetrievalIndex(root, views, mode="dino-global",
                                       crop=True, roi=(0, 0, 1, 1),
                                       encoder=MeanColorEncoder())
            query = np.full((20, 30, 3), (200, 20, 20), np.uint8)
            ranked, _, all_ranked = index.rank(query, top_k=1)
            self.assertEqual(ranked[0][2], 0)
            self.assertEqual(len(all_ranked), 2)
            excluded, _, _ = index.rank(query, top_k=1, excluded={0})
            self.assertEqual(excluded[0][2], 1)
            again = ViewRetrievalIndex(root, views, mode="dino-global",
                                       crop=True, roi=(0, 0, 1, 1),
                                       encoder=MeanColorEncoder())
            np.testing.assert_allclose(again.reference, index.reference)

    def test_roi_validation(self):
        with self.assertRaises(ValueError):
            ViewRetrievalIndex(Path("."), [], roi=(.9, .1, .2, .8))

    def test_negative_z_upper_view_changes_camera_not_model_xyz(self):
        center = np.array([12., 34., -200.])
        views = list(camera_positions(center, 100., 4, [0, 45], [2.],
                                      [2, 4], -1.))
        self.assertEqual(len(views), 6)
        eye = views[-1][-1]
        self.assertLess(eye[2], center[2])
        world_to_camera, _ = look_at_opencv(eye, center, up_sign=-1.)
        np.testing.assert_allclose(world_to_camera[:3, :3].T @
                                   world_to_camera[:3, :3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(center, [12., 34., -200.])

    def test_true_overhead_roll_rotates_camera_without_moving_eye(self):
        center = np.array([12., 34., -200.])
        eye = center + [0., 0., -250.]
        zero, zero_inverse = look_at_opencv(
            eye, center, up_sign=-1., preferred_up=np.array([0., 1., 0.]))
        ninety, ninety_inverse = look_at_opencv(
            eye, center, up_sign=-1., preferred_up=np.array([1., 0., 0.]))
        np.testing.assert_allclose(zero_inverse[:3, 3], eye)
        np.testing.assert_allclose(ninety_inverse[:3, 3], eye)
        np.testing.assert_allclose(zero[:3, :3].T @ zero[:3, :3], np.eye(3))
        np.testing.assert_allclose(ninety[:3, :3].T @ ninety[:3, :3], np.eye(3))
        np.testing.assert_allclose(zero[:3, :3] @ (center - eye), [0., 0., 250.])
        np.testing.assert_allclose(ninety[:3, :3] @ (center - eye), [0., 0., 250.])
        self.assertAlmostEqual(float(np.trace(zero[:3, :3] @ ninety[:3, :3].T)), 1.)

        with self.assertRaisesRegex(RuntimeError, 'nonzero'):
            look_at_opencv(eye, center, preferred_up=np.zeros(3))

    def test_reference_rgb_mask_uses_rendered_silhouette(self):
        image = np.full((80, 100, 3), 255, np.uint8)
        image[20:60, 30:70] = (100, 160, 200)
        mask = reference_rgb_mask(image)
        bbox = bbox_from_patch_mask(mask, margin=0)
        self.assertEqual(bbox, (.3, .25, .4, .5))


if __name__ == "__main__":
    unittest.main()
