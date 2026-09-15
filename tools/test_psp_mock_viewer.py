import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pydicom
from PyQt6.QtWidgets import QApplication

from dicom_viewer import DicomViewer
from viewsync_link import ViewSyncLink, ViewSyncRegistry


class PspMockViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        link = ViewSyncLink("1.2.3", "mikan", "テストデータ")
        self.viewer = DicomViewer(ViewSyncRegistry("http://127.0.0.1:5173/", {"1.2.3": link}))

    def tearDown(self):
        self.viewer.close()

    def set_study(self, study_uid: str, patient_id: str = "MOCK-001"):
        dataset = pydicom.Dataset()
        dataset.StudyInstanceUID = study_uid
        dataset.PatientID = patient_id
        self.viewer.datasets = [dataset]
        self.viewer.update_view_sync_state()

    def test_matching_study_enables_viewsync(self):
        self.set_study("1.2.3")
        self.assertTrue(self.viewer.view_sync_action.isEnabled())
        self.assertIn("利用可能", self.viewer.view_sync_status_label.text())
        self.assertEqual(self.viewer.patient_label.text(), "患者ID: MOCK-001")

    def test_missing_study_disables_viewsync(self):
        self.set_study("9.9.9")
        self.assertFalse(self.viewer.view_sync_action.isEnabled())
        self.assertIn("取得されていません", self.viewer.view_sync_status_label.text())


if __name__ == "__main__":
    unittest.main()
