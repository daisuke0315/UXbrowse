import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from object_roi import GSObjectDetector


class ObjectROITests(unittest.TestCase):
    def detector(self):
        calls = []
        def patch_grid(image):
            calls.append(image.shape)
            foreground = cv2.resize(image[:, :, 0], (16, 16),
                                    interpolation=cv2.INTER_NEAREST) > 100
            return np.stack((foreground, ~foreground), axis=-1).astype(np.float32)
        detector = GSObjectDetector.__new__(GSObjectDetector)
        detector.encoder = SimpleNamespace(patch_grid=patch_grid)
        detector.foreground = np.array([[1., 0.]], np.float32)
        detector.background = np.array([[0., 1.]], np.float32)
        detector.foreground_threshold = .55
        detector.margin_threshold = .02
        return detector, calls

    def test_small_object_is_found_by_independent_enlarged_search(self):
        detector, calls = self.detector()
        image = np.zeros((120, 160, 3), np.uint8)
        image[50:65, 90:105, 0] = 255
        original = image.copy()
        bbox, finding = detector.detect(image)
        self.assertIsNotNone(bbox)
        x, y, width, height = bbox
        self.assertTrue(x <= 98/160 <= x+width)
        self.assertTrue(y <= 58/120 <= y+height)
        self.assertLess(width, .35)
        self.assertLess(height, .35)
        self.assertEqual(finding['search_pass'], 'tile')
        self.assertEqual(len(calls), 5)
        mask = cv2.resize(finding['mask'], (160, 120), interpolation=cv2.INTER_NEAREST)
        self.assertTrue(mask[58, 98])
        self.assertFalse(mask[5, 5])
        np.testing.assert_array_equal(image, original)

    def test_full_detection_does_not_do_extra_inference(self):
        detector, calls = self.detector()
        image = np.zeros((120, 160, 3), np.uint8)
        image[30:80, 40:100, 0] = 255
        bbox, finding = detector.detect(image)
        self.assertIsNotNone(bbox)
        self.assertEqual(len(calls), 1)
        self.assertEqual(finding['search_pass'], 'full')

    def test_background_still_rejected_after_bounded_search(self):
        detector, calls = self.detector()
        bbox, finding = detector.detect(np.zeros((120, 160, 3), np.uint8))
        self.assertIsNone(bbox)
        self.assertEqual(finding['reason'], 'no_confident_object')
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(finding['trials']), 5)


if __name__ == '__main__':
    unittest.main()
