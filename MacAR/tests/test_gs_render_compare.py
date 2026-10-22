import json
from pathlib import Path
import tempfile
import unittest
import subprocess

from gs_render_compare import comparison_config, comparison_files, ReferenceCapture
import io
from PIL import Image


class GaussianRenderComparisonTests(unittest.TestCase):
    def make_capture(self, directory):
        source = Path(directory) / 'source'
        source.mkdir()
        (source / 'geometry.npz').write_bytes(b'geometry stays unchanged')
        manifest = source / 'manifest.json'
        manifest.write_text(json.dumps(dict(width=4, height=3, ply='/same.ply',
            views=[dict(image_name='view.png', data_name='geometry.npz', eye=[1, 2, 3])],
            geometry_renderer='ray-legacy-depth')))
        return ReferenceCapture(manifest, Path(directory) / 'rebuilt')

    def png(self, size=(4, 3)):
        stream = io.BytesIO()
        Image.new('RGB', size, 'white').save(stream, format='PNG')
        return stream.getvalue()

    def test_capture_keeps_geometry_and_requires_both_complete_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = self.make_capture(temp)
            with self.assertRaises(ValueError):
                capture.finish()
            for condition in ('alpha10', 'alpha15'):
                capture.save(condition, 0, self.png())
            outputs = capture.finish()
            for gain, folder in zip((1.0, 1.5), outputs):
                self.assertEqual((folder / 'geometry.npz').read_bytes(), b'geometry stays unchanged')
                manifest = json.loads((folder / 'manifest.json').read_text())
                self.assertEqual(manifest['rgb_alpha_gain'], gain)
                self.assertEqual(manifest['geometry_renderer'], 'ray-legacy-depth')
                self.assertEqual(manifest['views'][0]['eye'], [1, 2, 3])
                self.assertEqual(manifest['views'][0]['image_name'], 'view.png')
                self.assertFalse((folder / '.aliked_cache').exists())
            with self.assertRaises(ValueError):
                capture.save('alpha15', 0, self.png())
            with self.assertRaises(ValueError):
                capture.finish()

    def test_capture_rejects_wrong_pixels_targets_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = self.make_capture(temp)
            for condition, index, pixels in [('bad', 0, self.png()),
                    ('alpha15', -1, self.png()), ('alpha15', 1, self.png()),
                    ('alpha15', 0, self.png((3, 3))), ('alpha15', 0, b'not png')]:
                with self.assertRaises(ValueError):
                    capture.save(condition, index, pixels)
            with self.assertRaises(FileExistsError):
                ReferenceCapture(capture.manifest, Path(temp) / 'rebuilt')

    def test_relocated_reference_keeps_relative_ply_target(self):
        with tempfile.TemporaryDirectory() as temp:
            old = self.make_capture(temp)
            source = old.manifest.parent
            (source / 'splat.ply').write_bytes(b'ply\nend_header\n')
            data = dict(old.data, ply='splat.ply', center=[0,0,0], radius=100)
            old.manifest.write_text(json.dumps(data))
            capture = ReferenceCapture(old.manifest, Path(temp) / 'relative')
            for key in ('alpha10', 'alpha15'):
                capture.save(key, 0, self.png())
            for output in capture.finish():
                self.assertEqual(comparison_config(output/'manifest.json')['ply'],
                                 str(source/'splat.ply'))

    def test_reference_camera_preserves_roll_at_top_view(self):
        node = Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.is_file():
            self.skipTest('Bundled Node runtime is required for browser camera math')
        module = Path(__file__).resolve().parents[1] / 'browser_ui/gs-compare/profiles.js'
        code = '''import {referencePose} from MODULE;
const pose=referencePose({center:[0,0,0],up:[0,0,-1]},
 {eye:[0,0,-10],camera_to_world:[[0,1,0,0],[1,0,0,0],[0,0,1,-10],[0,0,0,1]]});
console.log(JSON.stringify(pose));'''.replace('MODULE', json.dumps(module.as_uri()))
        result = subprocess.check_output([str(node), '--input-type=module', '-e', code], text=True)
        pose = json.loads(result)
        self.assertEqual(pose['eye'], [0, 0, -10])
        self.assertEqual(pose['up'], [-1, 0, 0])
        self.assertEqual(pose['target'], [0, 0, 0])

    def test_same_model_and_camera_metadata_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            ply = root / "splat.ply"
            ply.write_bytes(b"ply\nend_header\n")
            manifest = root / "manifest.json"
            content = {"ply": str(ply), "center": [1, 2, 3],
                       "radius": 120, "up_axis": "negative-z",
                       "width": 960, "height": 720, "vertical_fov_deg": 65,
                       "views": [{"filename": "view_0000.png", "eye": [1, 500, 3]}]}
            manifest.write_text(json.dumps(content))
            before = manifest.read_bytes(), ply.read_bytes()
            config = comparison_config(manifest)
            self.assertEqual(config["center"], [1, 2, 3])
            self.assertEqual(config["up"], [0, 0, -1])
            self.assertEqual(config["views"][0]["eye"], [1, 500, 3])
            self.assertEqual(config["ply"], str(ply))
            files = comparison_files(manifest)
            self.assertEqual(files["/asset.ply"][0], ply)
            self.assertNotIn("/manifest.json", files)
            self.assertEqual(before, (manifest.read_bytes(), ply.read_bytes()))

    def test_missing_source_fails_before_browser_opens(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps({"ply": str(Path(temp) / "missing.ply")}))
            with self.assertRaises(FileNotFoundError):
                comparison_config(manifest)

    def test_relative_ply_is_resolved_against_manifest_not_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "splat.ply").write_bytes(b"ply\nend_header\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"ply": "splat.ply", "center": [0, 0, 0],
                "radius": 100, "views": [], "up_axis": "positive-z"}))
            config = comparison_config(manifest)
            self.assertEqual(config["ply"], str(root / "splat.ply"))
            self.assertEqual(config["up"], [0, 0, 1])
