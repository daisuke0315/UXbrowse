"""Native picker safety and audited default selection (no real UI/camera)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_file_picker import choose_local_file, LOCK
from map_preprocessing import atomic_json, recommended_map, selected_map


class FileSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.sources = dict(images='/dataset/Images.txt', cameras='/dataset/Cameras.txt', splat='/dataset/splat.ply')
        self.pilot = self.map('pilot', .16, 100)
        self.jobs = self.root/'jobs'

    def tearDown(self):
        self.temp.cleanup()

    def map(self, name, ratio, count, source=None, ok=True):
        path = self.root/name
        (path/'reference_features').mkdir(parents=True)
        (path/'reference_features/00000.npz').write_bytes(b'fixture')
        (path/'object_aliked_map.npz').write_bytes(b'fixture')
        atomic_json(path/'summary.json', dict(input_files=source or self.sources))
        atomic_json(path/'audit.json', dict(geometry_ok=ok, xyz_ratio=ratio, xyz_observations=count))
        return path

    def job(self, *paths):
        atomic_json(self.jobs/'test/state.json', dict(status='paused', passes=[dict(output=str(p)) for p in paths]))

    def test_best_saved_stage_default_even_when_later_job_paused(self):
        twelve = self.map('stage_12', .59, 523450)
        twentyfour = self.map('stage_24', .67, 600394)
        self.job(twelve, twentyfour)
        self.assertEqual(recommended_map(self.pilot, self.jobs), twentyfour)
        with patch('map_preprocessing.JOBS', self.jobs), patch('map_preprocessing.SELECTION', self.root/'absent'):
            self.assertEqual(selected_map(self.pilot), twentyfour)

    def test_other_dataset_failed_and_stale_maps_not_recommended(self):
        good = self.map('good', .6, 400)
        other = self.map('other', .99, 999, source=dict(self.sources, images='/other/Images.txt'))
        failed = self.map('failed', .99, 999, ok=False)
        stale = self.map('stale', .99, 999)
        stamp = (stale/'audit.json').stat().st_mtime_ns+1_000_000_000
        os.utime(stale/'reference_features/00000.npz', ns=(stamp, stamp))
        self.job(good, other, failed, stale)
        self.assertEqual(recommended_map(self.pilot, self.jobs), good)

    def test_explicit_selection_kept_and_missing_snapshot_falls_back(self):
        better = self.map('better', .8, 800)
        self.job(better)
        selection = self.root/'selection.json'
        atomic_json(selection, dict(map_path=str(self.pilot)))
        with patch('map_preprocessing.JOBS', self.jobs), patch('map_preprocessing.SELECTION', selection):
            self.assertEqual(selected_map(self.pilot), self.pilot)
        self.assertEqual(recommended_map(self.pilot, self.root/'missing'), self.pilot)

    def pick(self, field, value):
        with patch('browser_file_picker.sys.platform', 'darwin'), patch('browser_file_picker.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=value, stderr='')) as run:
            result = choose_local_file(field)
            self.assertEqual(run.call_args.args[0][:3], ['/usr/bin/osascript', '-e', __import__('browser_file_picker').SCRIPT])
            return result

    def test_map_file_resolves_related_directory_and_cancel_has_no_change(self):
        result = self.pick('map_path', str(self.pilot/'object_aliked_map.npz')+'\n')
        self.assertEqual(result['path'], str(self.pilot))
        self.assertEqual(result['xyz_ratio'], .16)
        self.assertEqual(self.pick('map_path', '\n'), dict(cancelled=True))

    def test_invalid_map_and_unknown_fields_rejected(self):
        bad = self.map('bad', .9, 900, ok=False)
        with self.assertRaises(ValueError):
            self.pick('map_path', str(bad/'object_aliked_map.npz'))
        with self.assertRaises(ValueError):
            choose_local_file('command')
        unrelated = self.pilot/'other.npz'
        unrelated.write_bytes(b'x')
        with self.assertRaises(ValueError):
            self.pick('map_path', str(unrelated))

    def test_paths_with_spaces_unicode_and_views_manifest(self):
        volume = self.root/'CT 日本語 volume.mha'
        volume.write_bytes(b'x')
        self.assertEqual(self.pick('volume_path', str(volume))['path'], str(volume))
        manifest = self.root/'manifest.json'
        manifest.write_text('{}')
        self.assertEqual(self.pick('views_path', str(manifest))['path'], str(self.root))
        with self.assertRaises(ValueError):
            self.pick('calibration_path', str(volume))

    def test_timeout_and_concurrent_dialog_release_lock(self):
        with patch('browser_file_picker.sys.platform', 'darwin'), patch('browser_file_picker.subprocess.run', side_effect=subprocess.TimeoutExpired('osascript', 300)):
            with self.assertRaises(ValueError):
                choose_local_file('map_path')
        self.assertFalse(LOCK.locked())
        LOCK.acquire()
        try:
            with patch('browser_file_picker.sys.platform', 'darwin'):
                with self.assertRaises(ValueError):
                    choose_local_file('map_path')
        finally:
            LOCK.release()

    def test_browser_file_controls_are_discoverable_and_paths_not_editable(self):
        ui = Path(__file__).resolve().parents[1]/'browser_ui'
        html = (ui/'index.html').read_text()
        self.assertEqual(html.count('id="map_path"'), 1)
        self.assertLess(html.index('id="map_path"'), html.index('<summary>登録データ・校正ファイル'))
        for field in ('map_path','views_path','volume_path','coordinates_path','calibration_path'):
            self.assertIn(f'id="{field}" readonly', html)
            self.assertIn(f'data-picker="{field}"', html)
        self.assertIn('id="recommendedMap"', html)
        self.assertIn('id="chooseMap"', (ui/'preprocess.html').read_text())
        self.assertIn("'/api/select-file'", (ui/'app.js').read_text())


if __name__ == '__main__':
    unittest.main()
