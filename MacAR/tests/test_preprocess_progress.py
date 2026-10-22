"""Progress presentation is independent of geometry and uses explicit units."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
NODE = Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(NODE.is_file(), 'Node runtime required')
class ProgressPresentationTests(unittest.TestCase):
    def view(self, **overrides):
        job = dict(status='running', stage='matching', pass_number=1, total_passes=3,
                   neighbors=12, completed=560, total=4405, passes=[],
                   config=dict(max_neighbors=48), elapsed_seconds=48.7)
        job.update(overrides)
        source = (ROOT/'browser_ui/preprocess-progress.js').read_text()
        script = source+'\nconsole.log(JSON.stringify(jobProgress('+json.dumps(job)+')));'
        result = subprocess.run([str(NODE),'--input-type=module','-e',script],capture_output=True,text=True,check=True)
        return json.loads(result.stdout)

    def test_current_pair_progress_not_map_coverage_or_overall_percent(self):
        view = self.view(counts=dict(cached_pairs=550,computed_pairs=10))
        self.assertAlmostEqual(view['fraction'],560/4405)
        self.assertIn('画像ペア',view['counter'])
        self.assertIn('12.7%',view['counter'])
        self.assertEqual([s['status'] for s in view['stages']],['current','pending','pending'])
        self.assertIn('550',view['cache'])
        self.assertEqual(view['duration'],'48秒')

    def test_saved_stage_is_usable_but_not_job_complete(self):
        saved=dict(neighbors=12,xyz_ratio=.756,xyz_observations=668936,output='/job/stage_12')
        view=self.view(stage='audit',pass_number=2,neighbors=24,total=620,completed=600,passes=[saved])
        self.assertEqual([s['status'] for s in view['stages']],['saved','current','pending'])
        self.assertEqual(view['latest'],saved)
        self.assertIn('登録画像',view['counter'])
        self.assertFalse(view['finished'])

    def test_old_worker_export_complete_does_not_falsely_show_completed_job(self):
        view=self.view(stage='complete',completed=600,total=620)
        self.assertEqual(view['phase'],'audit')
        self.assertFalse(view['finished'])
        self.assertIsNone(view['fraction'])
        self.assertEqual(view['process'],'全XYZの幾何監査')

    def test_completed_job_marks_unneeded_later_stages_and_stop_not_completion(self):
        saved=dict(neighbors=12,xyz_ratio=.75,xyz_observations=600)
        view=self.view(status='complete',stage='complete',passes=[saved])
        self.assertTrue(view['finished'])
        self.assertEqual([s['status'] for s in view['stages']],['saved','skipped','skipped'])
        paused=self.view(status='paused')
        stopped=self.view(status='interrupted')
        self.assertFalse(paused['finished'])
        self.assertFalse(stopped['finished'])
        self.assertIn('未完了',stopped['status'])

    def test_unknown_total_is_indeterminate_and_different_limits_supported(self):
        view=self.view(stage='triangulation',completed=0,total=0,config=dict(max_neighbors=96))
        self.assertIsNone(view['fraction'])
        self.assertEqual([s['neighbors'] for s in view['stages']],[12,24,96])
        view=self.view(config=dict(max_neighbors=12))
        self.assertEqual(len(view['stages']),1)

    def test_history_and_loaded_map_are_separate_and_module_served(self):
        html=(ROOT/'browser_ui/preprocess.html').read_text()
        for id in ('mapIdentity','jobList','jobHistory','jobHistoryList','jobSectionTitle'):
            self.assertIn('id="'+id+'"',html)
        self.assertIn('処理の進捗率ではありません',html)
        self.assertIn('/preprocess-progress.js', (ROOT/'browser_live.py').read_text())


if __name__=='__main__':
    unittest.main()
