import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_reference_rgb_mask_uses_rendered_silhouette(self):
        image = np.full((80, 100, 3), 255, np.uint8)
        image[20:60, 30:70] = (100, 160, 200)
        mask = reference_rgb_mask(image)
        bbox = bbox_from_patch_mask(mask, margin=0)
        self.assertEqual(bbox, (.3, .25, .4, .5))


if __name__ == "__main__":
    unittest.main()
