import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from browser_live import BrowserApplication,make_handler
from runtime_diagnostics import RuntimeDiagnostics
from http.server import ThreadingHTTPServer


NODE=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
MODULE=Path(__file__).resolve().parents[1]/'browser_ui/browser-benchmark.js'


class BrowserBenchmarkTests(unittest.TestCase):
    def js(self,script):
        if not NODE.exists():self.skipTest('Node unavailable')
        prefix="""
        import assert from 'node:assert/strict';
        import {readFileSync} from 'node:fs';
        const {ClockModel,BrowserBenchmark}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        """
        subprocess.check_call([str(NODE),'--input-type=module','-e',prefix+script,str(MODULE)])

    def test_clock_offset_minimum_rtt_and_expiry(self):
        self.js("""
        const clock=new ClockModel();
        assert.equal(clock.snapshot(0).valid,false);
        clock.add(100,1110,1112,122); // network RTT20, offset1000
        clock.add(200,1202,1204,206); // network RTT4, offset1000
        const best=clock.snapshot(250);
        assert.equal(best.valid,true);assert.equal(best.offset_ms,1000);
        assert.equal(best.uncertainty_ms,2);assert.equal(best.round_trip_ms,6);
        assert.equal(clock.snapshot(31000).valid,false);
        assert.throws(()=>clock.add(10,100,110,12));
        """)

    def test_per_frame_latency_distinct_pose_counts_and_actual_render_fps(self):
        self.js("""
        let now=100,raf=[],batches=[],renders=0;
        const b=new BrowserBenchmark({clientId:'client-1',now:()=>now,raf:f=>raf.push(f),
            send:async p=>batches.push(p),gsState:()=>({render_count:renders,active:true,ready:true})});
        b.clock.add(0,1000,1000,0);
        const meta={generation:1,sequence:1,source_id:7,pose_source_id:7,pose:{},status:'PASS',
          input_source:'camera',frozen:false,demo:false,capture_monotonic:1.05,sent_at:1.08,
          diagnostic_log:{session_id:'session-a'},mpr_source_id:6,mpr_age_ms:50,layer_mode:'projective'};
        b.drawn(meta,100,105,110,true);
        now=116;raf.shift()();now=133;raf.shift()();
        now=140;b.drawn({...meta,sequence:2},140,142,145,true);
        now=150;raf.shift()();now=166;raf.shift()();
        renders=30;now=1100;await b.flush();
        assert.equal(batches.length,1);const p=batches[0];
        assert.equal(p.window.received_frames,2);assert.equal(p.window.new_pose_frames,1);
        assert.equal(p.window.gs_frames,30);assert.equal(p.window.gs_fps,30);
        assert.equal(p.frames[0].python_receive_to_browser_receive_ms,50);
        assert.equal(p.frames[0].python_receive_to_paint_opportunity_ms,83);
        assert.equal(p.frames[0].paint_ms,133);assert.equal(p.frames[0].mpr_source_id,6);
        assert.equal(p.frames[0].sensor_exposure_time,null);
        assert.equal(p.frames[0].physical_presentation_time,null);
        """)

    def test_hidden_superseded_frames_and_stale_clock_do_not_invent_paints(self):
        self.js("""
        let now=100,raf=[],out=[];
        const b=new BrowserBenchmark({clientId:'client-2',now:()=>now,raf:f=>raf.push(f),send:async p=>out.push(p)});
        const m={generation:1,sequence:1,source_id:1,pose_source_id:null,status:'SEARCH',
          input_source:'camera',frozen:false,demo:false,capture_monotonic:1,
          diagnostic_log:{session_id:'session-a'}};
        b.drawn(m,100,101,102,true);b.drawn({...m,sequence:2},103,104,105,true);
        now=116;raf.shift()();raf.shift()();now=133;raf.shift()();
        b.drawn({...m,sequence:3},134,135,136,false);
        now=1100;await b.flush();
        assert.equal(out[0].frames.length,3);
        assert.equal(out[0].frames[0].paint_reason,'superseded');
        assert.equal(out[0].frames[0].paint_ms,null);
        assert.equal(out[0].frames[2].paint_reason,'hidden');
        assert.equal(out[0].frames[1].python_receive_to_paint_opportunity_ms,null);
        """)

    def test_bounded_transport_reports_failures_without_retry_queue(self):
        self.js("""
        let now=0,raf=[],reject=true,out=[];
        const b=new BrowserBenchmark({clientId:'client-3',now:()=>now,raf:f=>raf.push(f),
          send:async p=>{if(reject)throw Error('offline');out.push(p)}});
        const m={generation:1,sequence:1,source_id:1,status:'SEARCH',input_source:'camera',
          diagnostic_log:{session_id:'session-a'}};
        for(let i=0;i<100;i++)b.drawn({...m,sequence:i+1},i,i,i,false);
        now=1000;await b.flush();reject=false;
        b.drawn({...m,sequence:101},1001,1001,1001,false);
        now=2000;await b.endSession();
        assert.equal(out.at(-1).counters.dropped_frames+out.reduce((n,p)=>n+p.frames.length,0),101);
        assert.equal(out.at(-1).counters.http_failures,1);assert.ok(out.every(p=>p.frames.length<=64));
        const completed=out.length;
        b.drawn({...m,generation:2,diagnostic_log:{session_id:'session-b'}},2001,2001,2001,false);
        now=3000;await b.flush();assert.equal(out[completed].session_id,'session-b');
        assert.equal(out[completed].counters.dropped_frames,0);
        """)

    def test_http_clock_and_authenticated_batched_log(self):
        with tempfile.TemporaryDirectory() as directory:
            app=BrowserApplication(log_directory=Path(directory))
            app.diagnostics=logger=RuntimeDiagnostics(directory);app.generation=7
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(app))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            def post(body,token=None):
                request=Request(base+'/api/browser-telemetry',data=json.dumps(body).encode(),
                    headers={'Content-Type':'application/json','X-ViewSync-Token':token or app.token})
                return urlopen(request)
            try:
                clock=json.load(urlopen(base+'/api/clock'))
                self.assertGreaterEqual(clock['server_send_ms'],clock['server_receive_ms'])
                payload=dict(client_id='test-client',session_id=logger.session_id,generation=7,batch_id=1,
                    window=dict(start_ms=0,end_ms=1000,received_frames=1,new_pose_frames=0,
                        painted_frames=0,gs_frames=0,receive_fps=1,pose_fps=0,paint_fps=0,gs_fps=0),
                    counters=dict(dropped_frames=0,http_failures=0,transport_errors=0,clock_failures=0),
                    environment=dict(user_agent='test',width=800,height=600,pixel_ratio=1),
                    frames=[dict(sequence=1,source_id=1,pose_source_id=None,mpr_source_id=None,
                        status='SEARCH',input_source='camera',visibility='hidden',frozen=False,demo=False,
                        has_pose=False,received_ms=1,decoded_ms=2,drawn_ms=3,paint_ms=None,
                        paint_reason='hidden',capture_monotonic=1,sent_at=1,layer_mode='projective',
                        clock=None,python_receive_to_browser_receive_ms=None,
                        python_receive_to_paint_opportunity_ms=None,
                        sensor_exposure_time=None,physical_presentation_time=None)])
                with self.assertRaises(HTTPError):post(payload,'wrong')
                self.assertEqual(post(payload).status,202)
                with self.assertRaises(HTTPError):post(payload) # duplicate batch
                with self.assertRaises(HTTPError):post(dict(payload,generation=8,batch_id=2))
                with self.assertRaises(HTTPError):post(dict(payload,batch_id=2,frames=payload['frames']*65))
                with self.assertRaises(HTTPError):post(dict(payload,batch_id=2,frames=[dict(payload['frames'][0],received_ms=float('nan'))]))
                with self.assertRaises(HTTPError):post(dict(payload,batch_id=2,frames=[dict(payload['frames'][0],status=[])]))
                with self.assertRaises(HTTPError):post(dict(payload,batch_id=2,unexpected='no arbitrary data'))
                logger.close()
                rows=[json.loads(s) for s in logger.path.read_text().splitlines()]
                saved=[r for r in rows if r['event']=='browser_batch']
                self.assertEqual(len(saved),1)
                self.assertEqual(saved[0]['client_id'],'test-client')
                self.assertEqual(saved[0]['frames'][0]['paint_ms'],None)
            finally:
                app.diagnostics=None;logger.close();server.shutdown();server.server_close()
                app.executor.shutdown(wait=True);app.layer_executor.shutdown(wait=True)

    def test_stop_flushes_bounded_tail_and_ignores_late_same_session_frames(self):
        self.js("""
        let now=0,out=[];
        const b=new BrowserBenchmark({clientId:'client-4',now:()=>now,send:async p=>out.push(p)});
        const m={generation:1,source_id:1,status:'SEARCH',input_source:'camera',
          diagnostic_log:{session_id:'session-a'}};
        for(let i=0;i<70;i++)b.drawn({...m,sequence:i+1},i,i,i,false);
        now=1000;await b.endSession();
        assert.equal(out.reduce((n,p)=>n+p.frames.length,0),64);
        assert.equal(out.at(-1).counters.dropped_frames,6);
        b.drawn({...m,sequence:71},1001,1001,1001,false);
        assert.equal(b.queue.length,0);
        """)

    def test_negative_latency_is_flagged_and_expired_paint_clock_is_not_used(self):
        self.js("""
        let now=0,raf=[],out=[];
        const b=new BrowserBenchmark({clientId:'client-5',now:()=>now,raf:f=>raf.push(f),send:async p=>out.push(p)});
        b.clock.add(0,1000,1000,0);
        const m={generation:1,sequence:1,source_id:1,status:'SEARCH',input_source:'camera',
          capture_monotonic:2,diagnostic_log:{session_id:'session-a'}};
        b.drawn(m,100,101,102,true);now=110;raf.shift()();now=31000;raf.shift()();
        await b.flush();const r=out[0].frames[0];
        assert.equal(r.python_receive_to_browser_receive_ms,-900);
        assert.equal(r.clock_inconsistent,true);
        assert.equal(r.python_receive_to_paint_opportunity_ms,null);
        assert.equal(r.paint_clock_valid,false);
        """)


if __name__=='__main__':unittest.main()
