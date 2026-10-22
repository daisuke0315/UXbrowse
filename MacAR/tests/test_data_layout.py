"""Data placement is independent of code; relocation never deletes results."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DataLayoutTests(unittest.TestCase):
    def test_default_and_explicit_data_roots_are_outside_code(self):
        with patch.dict(os.environ, {}, clear=True):
            import macar_paths
            importlib.reload(macar_paths)
            self.assertEqual(macar_paths.DATA_DIR, macar_paths.CODE_DIR.parent/'data'/'MacAR')
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {'VIEWSYNC_MACAR_DATA_DIR': folder}):
                importlib.reload(macar_paths)
                self.assertEqual(macar_paths.DATA_DIR, Path(folder).resolve())
        importlib.reload(macar_paths)

    def setup_tree(self, root):
        root = root.resolve()
        code, data = root/'MacAR', root/'data'/'MacAR'
        code.mkdir()
        for name in ('browser_live.py','realtime_pnp.py'):
            (code/name).write_text('# code\n')
        (code/'tests').mkdir()
        (code/'.venv').mkdir()
        (code/'output_map').mkdir()
        return code, data

    def test_preview_never_creates_or_moves_files(self):
        from separate_macar_data import relocate_data
        with tempfile.TemporaryDirectory() as folder:
            code, data = self.setup_tree(Path(folder))
            result = relocate_data(code, data)
            self.assertEqual(len(result['moves']), 1)
            self.assertTrue((code/'output_map').is_dir())
            self.assertFalse(data.exists())

    def test_relocation_preserves_data_logs_and_original_metadata(self):
        from separate_macar_data import relocate_data
        with tempfile.TemporaryDirectory() as folder:
            code, data = self.setup_tree(Path(folder))
            original = json.dumps(dict(source=str(code/'output_map'),
                script=str(code/'browser_live.py'), count=52))
            summary = code/'output_map'/'summary.json'
            summary.write_text(original)
            stamp = summary.stat().st_mtime_ns
            (code/'output_map'/'points.npz').write_bytes(b'unchanged XYZ')
            log = json.dumps(dict(path=str(code/'output_map'), time_ms=123.45))+'\n'
            (code/'output_map'/'results.jsonl').write_text(log)
            (code/'.model_cache').mkdir()
            result = relocate_data(code, data, apply=True)
            self.assertFalse((code/'output_map').exists())
            self.assertTrue((code/'tests').is_dir())
            self.assertTrue((code/'.venv').is_dir())
            self.assertTrue((data/'.model_cache').is_dir())
            updated = json.loads((data/'output_map'/'summary.json').read_text())
            self.assertEqual(updated['source'],str(data/'output_map'))
            self.assertEqual(updated['script'],str(code/'browser_live.py'))
            self.assertEqual(updated['count'],52)
            self.assertEqual((data/'output_map'/'summary.json').stat().st_mtime_ns,stamp)
            backup = Path(result['record_dir'])/'original_metadata'/'output_map'/'summary.json'
            self.assertEqual(backup.read_text(),original)
            self.assertEqual((data/'output_map'/'points.npz').read_bytes(),b'unchanged XYZ')
            self.assertEqual((data/'output_map'/'results.jsonl').read_text(),log)

    def test_any_target_collision_aborts_all_moves(self):
        from separate_macar_data import relocate_data
        with tempfile.TemporaryDirectory() as folder:
            code, data = self.setup_tree(Path(folder))
            (code/'image').mkdir()
            (data/'output_map').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                relocate_data(code,data,apply=True)
            self.assertTrue((code/'image').is_dir())
            self.assertTrue((code/'output_map').is_dir())
            self.assertFalse((data/'image').exists())

    def test_bad_metadata_stops_before_any_move(self):
        from separate_macar_data import relocate_data
        with tempfile.TemporaryDirectory() as folder:
            code,data=self.setup_tree(Path(folder))
            (code/'output_map'/'summary.json').write_text('{not JSON')
            with self.assertRaises(ValueError): relocate_data(code,data,apply=True)
            self.assertTrue((code/'output_map').is_dir())
            self.assertFalse(data.exists())

    def test_interrupted_move_restores_already_moved_folders(self):
        from separate_macar_data import relocate_data
        with tempfile.TemporaryDirectory() as folder:
            code,data=self.setup_tree(Path(folder))
            (code/'image').mkdir()
            rename=Path.rename
            def interrupted(path,target):
                if path==code/'output_map': raise OSError('simulated move failure')
                return rename(path,target)
            with patch.object(Path,'rename',interrupted):
                with self.assertRaises(OSError): relocate_data(code,data,apply=True)
            self.assertTrue((code/'image').is_dir())
            self.assertTrue((code/'output_map').is_dir())
            self.assertFalse((data/'image').exists())

    def test_metadata_changes_only_paths_with_exact_moved_prefix(self):
        from separate_macar_data import relocate_data
        with tempfile.TemporaryDirectory() as folder:
            code,data=self.setup_tree(Path(folder))
            value=dict(real=str(code/'output_map'/'features'),
                unrelated=str(code/'output_map_other'), message='historical '+str(code/'output_map'))
            (code/'output_map'/'summary.json').write_text(json.dumps(value))
            relocate_data(code,data,apply=True)
            result=json.loads((data/'output_map'/'summary.json').read_text())
            self.assertEqual(result['real'],str(data/'output_map'/'features'))
            self.assertEqual(result['unrelated'],value['unrelated'])
            self.assertEqual(result['message'],value['message'])

    def test_data_inside_code_or_nonproject_source_is_rejected(self):
        from separate_macar_data import relocate_data
        with tempfile.TemporaryDirectory() as folder:
            code,_=self.setup_tree(Path(folder))
            with self.assertRaises(ValueError): relocate_data(code,code/'data',apply=True)
            with self.assertRaises(ValueError): relocate_data(Path(folder),Path(folder)/'elsewhere',apply=True)

    def test_runtime_defaults_keep_data_and_code_separate(self):
        from macar_paths import DATA_DIR
        from realtime_pnp import DEFAULT_SPLAT_VIEWS, DEFAULT_GLB_VIEWS, DEFAULT_IMAGES
        from map_preprocessing import JOBS, SELECTION
        from browser_live import reference_presets
        self.assertEqual(DEFAULT_SPLAT_VIEWS.parent,DATA_DIR)
        self.assertEqual(DEFAULT_GLB_VIEWS.parent,DATA_DIR)
        self.assertEqual(DEFAULT_IMAGES.parent.parent,DATA_DIR)
        self.assertEqual(JOBS.parent,DATA_DIR)
        self.assertEqual(SELECTION.parent,DATA_DIR)
        for preset in reference_presets(): self.assertEqual(Path(preset['path']).parent,DATA_DIR)
