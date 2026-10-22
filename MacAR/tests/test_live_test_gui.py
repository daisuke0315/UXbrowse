import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import live_test_gui
from live_test_gui import (CT_DISPLAY, LEGACY_COMPARE, MATCH_ONLY,
                           build_live_arguments, render_form)


class LiveTestGuiArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.object_map = root / 'map'
        self.views = root / 'views'
        self.object_map.mkdir()
        self.views.mkdir()
        (self.object_map / 'summary.json').touch()
        (self.object_map / 'object_aliked_map.npz').touch()
        (self.views / 'manifest.json').touch()
        self.volume = root / 'ct.mha'
        self.coordinates = root / 'coordinates.json'
        self.volume.touch()
        self.coordinates.touch()

    def options(self, mode):
        return build_live_arguments(
            mode, '1', '5', str(self.object_map), str(self.views),
            str(self.volume), str(self.coordinates), '')

    def test_ct_display_uses_registered_map_and_frozen_validation(self):
        options = self.options(CT_DISPLAY)
        self.assertEqual(options[options.index('--matching-mode') + 1],
                         'registered-aliked')
        self.assertIn('--ct-validation', options)
        self.assertNotIn('--no-mpr', options)

    def test_match_only_does_not_show_ct(self):
        options = self.options(MATCH_ONLY)
        self.assertEqual(options[options.index('--matching-mode') + 1],
                         'registered-aliked')
        self.assertIn('--no-mpr', options)
        self.assertNotIn('--ct-validation', options)

    def test_legacy_mode_is_explicitly_separate(self):
        options = self.options(LEGACY_COMPARE)
        self.assertEqual(options[options.index('--matching-mode') + 1],
                         'splat-track')
        self.assertIn('--no-mpr', options)
        self.assertNotIn('--ct-validation', options)

    def test_ct_mode_requires_both_ct_files(self):
        self.volume.unlink()
        with self.assertRaisesRegex(ValueError, 'CTボリューム'):
            self.options(CT_DISPLAY)

    def test_form_escapes_paths_and_shows_three_purposes(self):
        form = render_form({**live_test_gui.DEFAULT_FORM,
                            'map_path': '<unexpected>'}, 'test-token')
        self.assertIn('① CT重畳', form)
        self.assertIn('② 位置合わせ', form)
        self.assertIn('③ 旧3DGS', form)
        self.assertIn('&lt;unexpected&gt;', form)
        self.assertNotIn('value="<unexpected>"', form)

    def test_browser_submission_starts_selected_pipeline(self):
        values = {
            **live_test_gui.DEFAULT_FORM,
            'map_path': str(self.object_map),
            'views_path': str(self.views),
            'volume_path': str(self.volume),
            'coordinates_path': str(self.coordinates),
        }
        errors = []

        def open_browser(address, new):
            self.assertEqual(new, 2)

            def submit():
                try:
                    with urlopen(address, timeout=3) as response:
                        page = response.read().decode('utf-8')
                    import re
                    token = re.search(r'name="token" value="([^"]+)"', page).group(1)
                    payload = urlencode({**values, 'token': token}).encode('utf-8')
                    request = Request(address + 'start', data=payload, method='POST')
                    with urlopen(request, timeout=3) as response:
                        self.assertIn('カメラを起動', response.read().decode('utf-8'))
                except Exception as exc:
                    errors.append(exc)

            threading.Thread(target=submit, daemon=True).start()
            return True

        with mock.patch.dict(live_test_gui.DEFAULT_FORM, values), \
             mock.patch.object(live_test_gui.webbrowser, 'open',
                               side_effect=open_browser):
            options = live_test_gui.choose_settings()
        self.assertFalse(errors)
        self.assertIn('--ct-validation', options)
        self.assertIn('registered-aliked', options)


if __name__ == '__main__':
    unittest.main()
