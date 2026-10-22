import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

from live_search_resources import run_live_search


class LiveSearchResourcesTests(unittest.TestCase):
    def test_mps_uses_explicit_attention_only_during_live_job(self):
        engine,_,_=self.fixture()
        attention=SimpleNamespace(has_sdp=True)
        engine.matcher=SimpleNamespace(modules=lambda:[attention])
        self.assertFalse(run_live_search(engine,lambda:attention.has_sdp))
        self.assertTrue(attention.has_sdp)
        self.assertEqual(engine.diagnostic_sink.call_args.kwargs['attention_backend'],'explicit')

    def test_attention_setting_restored_after_inference_error(self):
        engine,_,_=self.fixture()
        attention=SimpleNamespace(has_sdp=True)
        engine.matcher=SimpleNamespace(modules=lambda:[attention])
        def fail():
            self.assertFalse(attention.has_sdp)
            raise ValueError('failed inference')
        with self.assertRaisesRegex(RuntimeError,'failed inference'):
            run_live_search(engine,fail)
        self.assertTrue(attention.has_sdp)

    def fixture(self, device='mps'):
        events=[]
        mps=SimpleNamespace(synchronize=Mock(side_effect=lambda:events.append('sync')),
            empty_cache=Mock(side_effect=lambda:events.append('release')),
            current_allocated_memory=Mock(return_value=100),
            driver_allocated_memory=Mock(side_effect=[1000,200]))
        engine=SimpleNamespace(device=SimpleNamespace(type=device),torch=SimpleNamespace(mps=mps),
            diagnostic_context=dict(search_id=1.),diagnostic_sink=Mock())
        return engine,mps,events

    def test_success_keeps_result_and_logs_reclamation(self):
        engine,mps,events=self.fixture()
        output=object()
        self.assertIs(run_live_search(engine,lambda:output),output)
        self.assertEqual(events,['sync','release','sync'])
        fields=engine.diagnostic_sink.call_args.kwargs
        self.assertEqual(fields['driver_before_bytes'],1000)
        self.assertEqual(fields['driver_after_bytes'],200)
        self.assertEqual(fields['search_id'],1.)

    def test_exception_does_not_keep_inference_locals_alive_in_future_traceback(self):
        engine,mps,events=self.fixture()
        references=[]
        class Temporary: pass
        def fail():
            temporary=Temporary()
            references.append(weakref.ref(temporary))
            raise RuntimeError('MPS backend out of memory')
        try:
            run_live_search(engine,fail)
        except RuntimeError as exc:
            retained=exc
        else:
            self.fail('Inference error must not become PASS')
        gc.collect()
        self.assertIsNone(references[0]())
        self.assertIn('MPS backend out of memory',str(retained))
        self.assertIn('release',events)
        self.assertTrue(engine.diagnostic_sink.call_args.kwargs['failed'])

    def test_cpu_does_not_touch_mps(self):
        engine,mps,events=self.fixture('cpu')
        attention=SimpleNamespace(has_sdp=True)
        engine.matcher=SimpleNamespace(modules=lambda:[attention])
        self.assertEqual(run_live_search(engine,lambda n:n+1,4),5)
        self.assertTrue(attention.has_sdp)
        self.assertEqual(events,[])
        mps.driver_allocated_memory.assert_not_called()


if __name__=='__main__':unittest.main()
