import unittest
import json
from pathlib import Path
import tempfile

from evaluate_alpha_references import summarize, validate_reference_geometry


class AlphaReferenceEvaluationTests(unittest.TestCase):
    def test_rgb_ablation_rejects_changed_geometry_or_cameras(self):
        with tempfile.TemporaryDirectory() as temporary:
            a,b=Path(temporary)/'a',Path(temporary)/'b'
            a.mkdir();b.mkdir()
            manifest=dict(width=4,height=3,views=[dict(data_name='view.npz',eye=[0,0,-10])])
            for directory in (a,b):
                (directory/'manifest.json').write_text(json.dumps(manifest))
                (directory/'view.npz').write_bytes(b'same geometry')
            validate_reference_geometry(dict(original=a,alpha15=b))
            (b/'view.npz').write_bytes(b'different geometry')
            with self.assertRaises(ValueError):
                validate_reference_geometry(dict(original=a,alpha15=b))
            (b/'view.npz').write_bytes(b'same geometry')
            manifest['views'][0]['eye']=[0,0,10]
            (b/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_reference_geometry(dict(original=a,alpha15=b))

    def test_equivalent_relative_and_absolute_ply_are_same_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            a,b=root/'a',root/'b'
            a.mkdir();b.mkdir()
            for directory,ply in ((a,str(root/'splat.ply')),(b,'../splat.ply')):
                (directory/'manifest.json').write_text(json.dumps(dict(ply=ply,
                    views=[dict(data_name='view.npz')], width=4,height=3)))
                (directory/'view.npz').write_bytes(b'same geometry')
            validate_reference_geometry(dict(original=a,alpha15=b))

    def test_summary_counts_failures_without_mixing_error_with_accuracy(self):
        rows = [dict(result=dict(status='PASS', inliers=20, processing_ms=100,
                                  mean_reprojection_error=2)),
                dict(result=dict(status='FAIL', inliers=4, processing_ms=300,
                                  mean_reprojection_error=.1)),
                dict(result=dict(status='AMBIGUOUS', inliers=30, processing_ms=200,
                                  mean_reprojection_error=1))]
        result = summarize(rows)
        self.assertEqual(result['cases'], 3)
        self.assertEqual(result['pass_count'], 1)
        self.assertEqual(result['ambiguous_count'], 1)
        self.assertEqual(result['inliers_median_all'], 20)
        self.assertEqual(result['reprojection_median_pass_px'], 2)
        self.assertEqual(result['processing_median_ms'], 200)
        self.assertNotIn('accuracy', result)

    def test_no_pass_has_no_accepted_error_statistic(self):
        result = summarize([dict(result=dict(status='FAIL', inliers=0,
                                processing_ms=30, mean_reprojection_error=None))])
        self.assertIsNone(result['reprojection_median_pass_px'])
        self.assertIsNone(result['inliers_median_pass'])

    def test_total_time_includes_failed_initial_attempt_and_expanded_retry(self):
        result = summarize([dict(wall_ms=900, result=dict(status='PASS', inliers=20,
                                processing_ms=300, mean_reprojection_error=2))])
        self.assertEqual(result['processing_median_ms'], 900)
