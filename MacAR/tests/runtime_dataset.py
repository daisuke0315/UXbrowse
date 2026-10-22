"""Tiny file-existence fixture for runtime settings, not localization truth."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from unittest.mock import patch


class RuntimeDataset:
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        stack=ExitStack()
        cls.addClassCleanup(stack.close)
        data=Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        ply=data/'splat.ply'
        ply.write_text('ply\nformat ascii 1.0\nend_header\n')
        for name,gain in [('output_3dgs_reference_views_72_upper_negz_v2',1.),
                          ('output_3dgs_reference_views_upper_rgb_20261003_alpha15',1.5)]:
            directory=data/name
            directory.mkdir()
            (directory/'manifest.json').write_text(json.dumps(dict(
                source_type='3dgs',runtime_transform='identity',ply=str(ply),rgb_alpha_gain=gain,
                up_axis='negative-z')))
        mapping=data/'map'
        mapping.mkdir()
        (mapping/'object_aliked_map.npz').write_bytes(b'file-existence fixture only')
        (mapping/'summary.json').write_text('{}')
        volume=data/'volume.mha'
        volume.touch()
        coordinates=data/'coordinates.json'
        coordinates.write_text('{}')
        import browser_live
        stack.enter_context(patch.object(browser_live,'DATA_DIR',data))
        stack.enter_context(patch.object(browser_live,'DEFAULT_SPLAT_VIEWS',
            data/'output_3dgs_reference_views_72_upper_negz_v2'))
        stack.enter_context(patch.dict(browser_live.DEFAULTS,dict(
            views_path=str(data/'output_3dgs_reference_views_72_upper_negz_v2'),
            map_path=str(mapping),volume_path=str(volume),coordinates_path=str(coordinates))))
