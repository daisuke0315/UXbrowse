import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from viewsync_link import ViewSyncLink, ViewSyncRegistry, build_viewer_url


class ViewSyncLinkTests(unittest.TestCase):
    def test_registry_only_returns_available_supported_studies(self):
        payload = {
            "viewerUrl": "http://127.0.0.1:5173/",
            "studies": {
                "study-a": {"available": True, "dataset": "mikan"},
                "study-b": {"available": False, "dataset": "banana"},
                "study-c": {"available": True, "dataset": "unknown"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            registry = ViewSyncRegistry.from_file(path)
        self.assertIsNotNone(registry.find("study-a"))
        self.assertIsNone(registry.find("study-b"))
        self.assertIsNone(registry.find("study-c"))

    def test_browser_url_contains_study_but_no_patient_identifier(self):
        url = build_viewer_url(
            "http://127.0.0.1:5173/",
            ViewSyncLink("1.2.3", "banana", "demo"),
        )
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["study"], ["1.2.3"])
        self.assertEqual(query["dataset"], ["banana"])
        self.assertNotIn("patient", query)


if __name__ == "__main__":
    unittest.main()
