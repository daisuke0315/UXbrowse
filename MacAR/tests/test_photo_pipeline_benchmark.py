import unittest

from benchmark_photo_pipeline import summarize_attempts, image_summary


class PhotoPipelineBenchmarkTest(unittest.TestCase):
    def test_all_attempts_and_nested_qc_not_double_counted(self):
        first = dict(timings_ms=dict(retrieval=10, query_features=20),
                     reference_selection=dict(elapsed_ms=30),
                     candidates=[dict(timings_ms=dict(reference_load=1, lightglue=40,
                         geometry=5, xyz_assignment=2, pnp=4, qc=3, spatial_scoring=2))])
        retry = dict(timings_ms=dict(retrieval=1, query_features=1),
                     reference_selection=dict(elapsed_ms=30),
                     candidates=[dict(timings_ms=dict(lightglue=10, qc=2))])
        actual = summarize_attempts([dict(diagnostics=first, selection_computed=True),
                                     dict(diagnostics=retry, selection_computed=False)], 150)
        self.assertEqual(actual['reference_selection_ms'], 30)
        self.assertEqual(actual['qc_ms'], 5)
        self.assertEqual(actual['lightglue_ms'], 50)
        self.assertEqual(actual['other_ms'], 21)
        self.assertEqual(actual['candidate_calls'], 2)
        self.assertEqual(actual['retry_attempts'], 1)

    def test_warm_summary_keeps_failures_excludes_incomplete(self):
        rows = [dict(repeat=0, completed=True, status='PASS', wall_ms=100),
                dict(repeat=1, completed=True, status='FAIL', wall_ms=40),
                dict(repeat=2, completed=True, status='PASS', wall_ms=20),
                dict(repeat=3, completed=False, status='RESOURCE_STOP', wall_ms=900)]
        actual = image_summary(rows)
        self.assertEqual(actual['warm_wall_median_ms'], 30)
        self.assertEqual(actual['warm_pass_count'], 1)
        self.assertEqual(actual['warm_completed'], 2)
        self.assertEqual(actual['incomplete'], 1)


if __name__ == '__main__':
    unittest.main()
