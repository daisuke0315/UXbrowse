import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_diagnostics import RuntimeDiagnostics
from glb_virtual_live import GLBVirtualLiveLocalizer, AsyncTemporalGLBLocalizer


class DiagnosticTests(unittest.TestCase):
    def test_wall_clock_ids_configuration_and_flush(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = RuntimeDiagnostics(folder)
            logger.emit('configuration', profile='precision')
            logger.emit('search_completed', result={'error': float('nan')},
                        points=np.ones((2000, 2)))
            logger.close()
            logger.close()
            rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],
                             ['session_open', 'configuration', 'search_completed', 'session_close'])
            self.assertTrue(all(r['session_id'] == logger.session_id for r in rows))
            self.assertEqual([r['record'] for r in rows], [1, 2, 3, 4])
            self.assertIn('T', rows[1]['wall_time'])
            self.assertIsNone(rows[2]['result']['error'])
            self.assertEqual(rows[2]['points'], {'omitted_array_shape': [2000, 2]})
            self.assertFalse(logger.worker.is_alive())

    def test_events_detach_mutable_worker_results(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = RuntimeDiagnostics(folder)
            result = {'quality': {'reasons': ['original']}}
            logger.emit('completed', result=result)
            result['quality']['reasons'].append('later')
            logger.close()
            rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
            self.assertEqual(rows[1]['result']['quality']['reasons'], ['original'])

    def test_sampling_throttled_but_status_changes_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = RuntimeDiagnostics(folder)
            with patch('runtime_diagnostics.time.monotonic', return_value=logger.started+1.):
                logger.sample({'status': 'PASS', 'sequence': 1})
                logger.sample({'status': 'PASS', 'sequence': 2})
                logger.sample({'status': 'SEARCH', 'sequence': 3})
            logger.close()
            rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
            self.assertEqual([r['sequence'] for r in rows if r['event'] == 'frame_sample'], [1, 3])

    def test_records_actual_pipeline_rates_and_slice_source_age(self):
        with tempfile.TemporaryDirectory() as folder:
            logger=RuntimeDiagnostics(folder)
            rates=dict(tracking=29.5,mpr=7.0,packets=29.5)
            logger.sample(dict(status='PASS',pipeline_fps=rates,mpr_source_id=7,
                               mpr_age_ms=150.,layer_mode='projective'))
            logger.close()
            rows=[json.loads(line) for line in logger.path.read_text().splitlines()]
            sample=next(r for r in rows if r['event']=='frame_sample')
            self.assertEqual(sample['pipeline_fps'],rates)
            self.assertEqual(sample['mpr_source_id'],7)
            self.assertEqual(sample['mpr_age_ms'],150.)

    def test_bounded_queue_drops_without_blocking_capture(self):
        release = threading.Event()
        def blocked_writer(logger):
            release.wait(2.)
            logger.file.close()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(RuntimeDiagnostics, '_write', blocked_writer):
                logger = RuntimeDiagnostics(folder, capacity=1)
                for i in range(10):
                    logger.emit('frame', number=i)
                self.assertEqual(logger.queue.qsize(), 1)
                self.assertEqual(logger.snapshot()['dropped_records'], 10)
                release.set()
                logger.close()

    def test_progress_reports_exact_stages_and_search_context(self):
        engine = GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        sink = engine.diagnostic_sink = Mock()
        engine.diagnostic_context = dict(search_id=42, search_frame_id=17)
        engine._progress('features', query_features=2000)
        sink.assert_called_once_with('search_stage', search_id=42, search_frame_id=17,
                                     stage='features', query_features=2000)

    def test_wrapper_events_keep_source_and_job_identity(self):
        wrapper = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        wrapper.search_started = 42.
        wrapper.search_mode = 'guided'
        wrapper.search_frame_id = 11
        wrapper.search_source_stamp = 41.9
        sink = wrapper.diagnostic_sink = Mock()
        wrapper._diagnostic('replenishment', added=7)
        sink.assert_called_once_with('replenishment', search_id=42., search_mode='guided',
            search_frame_id=11, search_capture_monotonic=41.9, added=7)

    def test_clock_is_visible_in_video_panel(self):
        node = Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():
            self.skipTest('Node unavailable')
        source = Path(__file__).resolve().parents[1] / 'browser_ui/localization-progress.js'
        script = """
        import {readFileSync} from 'node:fs';
        const module=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        console.log(module.diagnosticLogSummary({wall_time:'2026-10-01T16:14:09.123+09:00',elapsed_ms:1000,file_name:'run.jsonl',dropped_records:2},17));
        """
        result = subprocess.check_output([str(node), '--input-type=module', '-e', script, str(source)], text=True)
        self.assertIn('16:14:09.123', result)
        self.assertIn('#17', result)
        self.assertIn('記録欠落 2件', result)


if __name__ == '__main__':
    unittest.main()
