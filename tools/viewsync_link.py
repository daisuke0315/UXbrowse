"""ViewSync study registry and safe browser-link construction.

The PSP mock uses Study Instance UID as the lookup key. Patient identifiers are
intentionally never copied into the browser URL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit


SUPPORTED_DATASETS = frozenset({"mikan", "banana", "all"})


@dataclass(frozen=True)
class ViewSyncLink:
    study_uid: str
    dataset: str
    label: str


class ViewSyncRegistry:
    def __init__(self, viewer_url: str, studies: dict[str, ViewSyncLink]) -> None:
        self.viewer_url = viewer_url
        self.studies = studies

    @classmethod
    def from_file(cls, path: Path) -> "ViewSyncRegistry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        viewer_url = str(payload.get("viewerUrl", "http://127.0.0.1:5173/"))
        studies: dict[str, ViewSyncLink] = {}
        for study_uid, entry in payload.get("studies", {}).items():
            if not isinstance(entry, dict) or entry.get("available") is False:
                continue
            dataset = str(entry.get("dataset", ""))
            if dataset not in SUPPORTED_DATASETS:
                continue
            studies[str(study_uid)] = ViewSyncLink(
                study_uid=str(study_uid),
                dataset=dataset,
                label=str(entry.get("label", dataset)),
            )
        return cls(viewer_url, studies)

    def find(self, study_uid: str) -> ViewSyncLink | None:
        return self.studies.get(study_uid)


def build_viewer_url(base_url: str, link: ViewSyncLink) -> str:
    """Build an allow-listed ViewSync URL without patient-identifying fields."""
    if link.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported ViewSync dataset: {link.dataset}")
    parts = urlsplit(base_url)
    query = urlencode({
        "dataset": link.dataset,
        "study": link.study_uid,
        "source": "psp-mock",
    })
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))
