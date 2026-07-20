import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("rc_frame_extractor.py")
SPEC = importlib.util.spec_from_file_location("rc_frame_extractor", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
import sys
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class MetricTests(unittest.TestCase):
    def test_cli_defaults_to_proven_direct_workflow(self):
        args = module.parser().parse_args(["input.mov"])
        self.assertEqual(args.mode, "direct")
        self.assertEqual(args.sample_fps, 10.0)
        self.assertEqual(args.quality, 1)

    def test_identical_images_have_zero_difference(self):
        image = np.full((20, 20), 100, dtype=np.float32)
        self.assertEqual(module.difference_score(image, image), 0.0)

    def test_edge_is_sharper_than_flat_image(self):
        flat = np.full((20, 20), 100, dtype=np.float32)
        edge = flat.copy()
        edge[:, 10:] = 220
        self.assertGreater(module.sharpness_score(edge), module.sharpness_score(flat))

    def test_gap_guard_preserves_coverage(self):
        preview = np.full((20, 20), 100, dtype=np.float32)
        metrics = [
            module.FrameMetric(Path("a"), 0, 0.0, 20.0, preview),
            module.FrameMetric(Path("b"), 1, 2.0, 20.0, preview),
        ]
        kept, decisions = module.choose_frames(metrics, 18.0, 0.02, 1.5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(decisions[-1][1], "gap_guard")


if __name__ == "__main__":
    unittest.main()
