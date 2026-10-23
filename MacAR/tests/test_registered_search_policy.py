"""Scheduling regression tests: real candidate loop, mocked neural inference."""
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registered_aliked_live import RegisteredALIKEDLiveLocalizer, reference_exit_decision
from spatial_pose_scoring import spatial_pose_score


def hypothesis(name='a', count=36, mean=1., offset=0., rotation=0., passed=True, samples=36):
    side=int(np.ceil(np.sqrt(samples)))
    uv = np.array([(x, y) for x in np.linspace(5, 95, side)
                   for y in np.linspace(5, 95, side)], np.float32)[:samples]
    transfers = [SimpleNamespace(query_uv=tuple(p), xyz=(float(p[0])*1000., float(p[1])*1000., 0.)) for p in uv]
    pose = SimpleNamespace(rvec=np.array([0., 0., rotation]),
                           tvec=np.array([offset, 0., 1000.]),
                           inlier_indices=np.arange(count), mean_error=mean)
    quality = dict(passed=passed, reasons=[] if passed else ['few_inliers'],
                   inlier_ratio=count/samples, mean_error=mean, p95_error=mean+1., spread_3d=.5,
                   coverage=.81)
    quality['spatial_score'] = spatial_pose_score(uv[:count], np.full(count, mean),
                                                (0, 0, 100, 100), 4.)
    return pose, transfers, quality, name, 0.


def decision(hypotheses):
    return reference_exit_decision(hypotheses, min_inliers=10, min_ratio=.2,
        max_error=3., max_coarse_angle=55., scene_scale=200., crop_area=10000.)


from tests.runtime_dataset import RuntimeDataset


class RegisteredSearchPolicyTests(RuntimeDataset, unittest.TestCase):
    def test_two_strong_photos_are_required(self):
        self.assertFalse(decision([hypothesis()])['stop'])
        good = decision([hypothesis('a'), hypothesis('b', offset=2.)])
        self.assertTrue(good['stop'])
        self.assertEqual(good['supporting_references'], ['a', 'b'])
        json.dumps(good, allow_nan=False)
        self.assertFalse(decision([hypothesis('a'), hypothesis('a')])['stop'])

    def test_ordinary_pass_is_not_enough(self):
        for bad in (hypothesis('b', count=10), hypothesis('b', mean=2.1),
                    hypothesis('b', passed=False)):
            self.assertFalse(decision([hypothesis('a'), bad])['stop'])
        clustered = hypothesis('b')
        for point in clustered[1]:
            point.query_uv = (50., 50.)
        self.assertFalse(decision([hypothesis('a'), clustered])['stop'])
        for key, value in [('p95_error', 3.1), ('spread_3d', .01), ('mean_error', np.nan)]:
            bad = hypothesis('b'); bad[2][key] = value
            self.assertFalse(decision([hypothesis('a'), bad])['stop'])
        bad = list(hypothesis('b')); bad[4] = 21.
        self.assertFalse(decision([hypothesis('a'), bad])['stop'])

    def test_conflicting_poses_do_not_stop(self):
        for offset, rotation in ((20., 0.), (0., np.radians(10))):
            self.assertFalse(decision([hypothesis('a'),
                hypothesis('b', offset=offset, rotation=rotation)])['stop'])
        actual = decision([hypothesis('a'), hypothesis('b'), hypothesis('c', offset=100.)])
        self.assertFalse(actual['stop'])
        self.assertEqual(actual['reason'], 'competing_pose')

    def test_low_raw_count_broad_conflict_blocks_dense_early_exit(self):
        dense = [hypothesis('a'), hypothesis('b')]
        corners = np.array([(5, 5), (5, 95), (95, 5), (95, 95)]*9)
        for h in dense:
            for transfer, uv in zip(h[1], corners):
                transfer.query_uv = tuple(uv)
            h[2]['spatial_score'] = spatial_pose_score(corners, np.ones(36),
                                                      (0, 0, 100, 100), 4.)
        broad_conflict = hypothesis('c', count=24, mean=2.1, offset=100.)
        actual = decision(dense+[broad_conflict])
        self.assertFalse(actual['stop'])
        self.assertEqual(actual['reason'], 'competing_pose')

    def run_search(self, policy='adaptive', qualities=None, validation=False, weak_first=False,
                   dense_first=False, conflicting_sparse=False, default_selection=False,
                   context=False, support=36):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        engine = RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.search_policy, engine.scene_scale = policy, 200.
        engine.torch, engine.device = torch, 'cpu'
        engine.resize, engine.top_k = 1024, 3
        engine.retrieval_mode = 'dino-patch'
        engine.extractor, engine.matcher = None, None
        engine.fundamental_threshold, engine.ransac_threshold = 3., 4.
        engine.iterations, engine.min_inliers = 10000, 10
        engine.min_inlier_ratio, engine.max_reprojection_error = .2, 3.
        engine.max_coarse_rotation_deg, engine.mpr_renderer = 55., None
        engine.map_data=SimpleNamespace(xyz=np.array([[-100.,-100.,0.],[100.,100.,100.]]))
        engine.views = [dict(image_name=f'v{i}.png', world_to_camera=np.eye(4)) for i in range(3)]
        engine.retriever = SimpleNamespace(rank_debug=Mock(return_value=(
            [(1., 0., i) for i in range(3)], 0, [], dict(bbox=(0, 0, 1, 1)))))
        engine.diagnostic_sink = Mock()
        h = hypothesis(count=support,samples=support)
        uv = np.array([t.query_uv for t in h[1]], np.float32)
        if dense_first:
            uv[:20] = (5., 5.)
            h[1][:20] = [SimpleNamespace(query_uv=(5., 5.), xyz=(5000., 5000., 0.))
                         for _ in range(20)]
        features = dict(keypoints=torch.tensor(uv[None]), image_size=torch.tensor([[100., 100.]]))
        engine._reference = Mock(return_value=(features, np.arange(support),
            np.array([t.xyz for t in h[1]], np.float64)))
        engine.validate_pose = Mock(side_effect=qualities if qualities else
                                    lambda *a: copy.deepcopy(h[2]))
        references = [(str(n), n//2, 0., 0.) for n in range(6)]
        workspace = dict(references=references)
        if default_selection:
            workspace = {}
            engine._appearance_references = Mock(return_value=references)
        if validation:
            workspace['validation_pose'] = (np.zeros(3), np.zeros(3))
        if context:
            workspace['pose_prior']=(np.zeros(3),np.array([-120.,0.,1000.]))
        image = np.zeros((100, 100, 3), np.uint8)
        poses = [hypothesis(count=10)[0] if weak_first and n == 0 else h[0] for n in range(6)]
        if dense_first:
            poses = [copy.deepcopy(h[0]) for _ in range(6)]
            poses[0].inlier_indices = np.arange(20)
            for pose in poses[1:]:
                pose.inlier_indices = np.arange(20, 36)
            if conflicting_sparse:
                poses[1].tvec = np.array([100., 0., 1000.])
        with patch('registered_aliked_live.crop_live_object', return_value=(image, (0, 0, 100, 100))), \
             patch('registered_aliked_live._image_tensor', return_value=None), \
             patch('registered_aliked_live._extract', return_value=features) as extract, \
             patch('registered_aliked_live._match_pair', return_value=(uv, uv,
                 np.column_stack((np.arange(support), np.arange(support))), np.linspace(1., .5, support))) as matcher, \
             patch('registered_aliked_live.cv2.findFundamentalMat', return_value=(np.eye(3), np.ones((support, 1)))), \
             patch('registered_aliked_live.solve_pose', side_effect=poses):
            result, _ = engine._localize_once(image, np.eye(3), np.zeros(5), workspace=workspace)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(engine.retriever.rank_debug.call_count, 1)
            self.assertEqual(matcher.call_count, engine._reference.call_count)
        return engine, result

    def test_context_leaves_old_pose_difference_for_current_common_pool_arbitration(self):
        for support in (36,64):
            engine,result=self.run_search(context=True,support=support)
            self.assertEqual(result.status,'PASS')
            comparison=engine.last_diagnostics['quality']['pose_comparison']
            self.assertGreater(comparison['center_distance'],0.)
            self.assertNotIn('mismatch',comparison)
            self.assertNotIn('pose_prior_mismatch',result.detail)

    def test_context_does_not_override_normal_source_qc(self):
        engine,result=self.run_search(context=True,support=36,
            qualities=[dict(hypothesis()[2],passed=False,reasons=['reprojection_error']) for _ in range(6)])
        self.assertEqual(result.status,'FAIL')
        self.assertIn('reprojection_error',result.detail)

    def test_default_search_uses_appearance_photos_not_angle_shortlist(self):
        engine, result = self.run_search(default_selection=True)
        self.assertEqual(result.status,'PASS')
        engine._appearance_references.assert_called_once()
        self.assertEqual(engine.last_diagnostics['reference_selection']['mode'],'gs-real')

    def test_final_selection_uses_regions_and_records_score(self):
        engine, result = self.run_search(policy='exhaustive', dense_first=True)
        self.assertEqual(result.status, 'PASS')
        self.assertEqual(engine.last_reference_name, '1')
        self.assertEqual(result.inliers, 16)
        self.assertIn('spatial_score', engine.last_diagnostics['quality'])

    def test_broad_low_count_incompatible_pose_is_ambiguous(self):
        engine, result = self.run_search(policy='exhaustive', dense_first=True,
                                        conflicting_sparse=True)
        self.assertEqual(result.status, 'FAIL')
        self.assertIn('ambiguous_pose', engine.last_diagnostics['quality']['reasons'])

    def test_adaptive_loop_stops_after_top1_pair(self):
        engine, result = self.run_search()
        self.assertEqual(result.status, 'PASS')
        self.assertIn('refs 2/6 early', result.detail)
        summary = engine.last_diagnostics['reference_search']
        self.assertEqual(summary['processed'], 2)
        self.assertEqual(summary['skipped_names'], ['2', '3', '4', '5'])
        events = [c for c in engine.diagnostic_sink.call_args_list
                  if c.args[0] == 'reference_search_decision']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kwargs['skipped'], 4)

    def test_exhaustive_and_validation_preserve_all_candidates(self):
        for kwargs in (dict(policy='exhaustive'), dict(validation=True)):
            engine, result = self.run_search(**kwargs)
            self.assertEqual(result.status, 'PASS')
            self.assertEqual(engine._reference.call_count, 6)
            self.assertFalse(engine.last_diagnostics['reference_search']['early_exit'])

    def test_weak_top1_expands_but_can_stop_at_next_group(self):
        engine, result = self.run_search(weak_first=True)
        self.assertEqual(result.status, 'PASS')
        self.assertEqual(engine._reference.call_count, 4)
        self.assertEqual([c['processed'] for c in engine.last_diagnostics['reference_search']['checkpoints']], [2, 4])

    def test_weak_quality_processes_complete_shortlist(self):
        qualities = [hypothesis(mean=2.5)[2] for _ in range(6)]
        engine, result = self.run_search(qualities=qualities)
        self.assertEqual(result.status, 'PASS')  # normal QC is deliberately unchanged
        self.assertEqual(engine._reference.call_count, 6)

    def test_browser_and_cli_propagate_policy(self):
        from browser_live import DEFAULTS, runtime_arguments
        from realtime_pnp import build_parser
        self.assertEqual(DEFAULTS['registered_search_policy'], 'adaptive')
        self.assertEqual(build_parser().parse_args([]).registered_search_policy, 'adaptive')
        for policy in ('adaptive', 'exhaustive'):
            args, _, _ = runtime_arguments(dict(mpr=False, registered_search_policy=policy))
            self.assertEqual(args.registered_search_policy, policy)
        with self.assertRaisesRegex(ValueError, 'search policy'):
            runtime_arguments(dict(registered_search_policy='invalid'))
        html = (Path(__file__).resolve().parents[1]/'browser_ui/index.html').read_text()
        self.assertIn('id="registered_search_policy"', html)


if __name__ == '__main__':
    unittest.main()
