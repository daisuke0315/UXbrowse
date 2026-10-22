"""Synthetic anatomy is display-only, world-fixed, and clipped by the accepted CT plane."""
import json
from pathlib import Path
import subprocess
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NODE=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')


class AnatomyDemoTests(unittest.TestCase):
    def run_js(self, script):
        if not NODE.exists():self.skipTest('Node unavailable')
        setup="""
        const {buildAnatomy,anatomyPolygons:project}=await import(process.argv[1]);
        const bounds=[[-50,-50,-50],[50,50,50]],meshes=buildAnatomy(bounds);
        const pose={eye:[0,0,-200],target:[0,0,-199],up:[0,-1,0],K:[[200,0,100],[0,200,100],[0,0,1]],width:200,height:200};
        const plane=z=>({center:[0,0,z],normal:[0,0,1]});
        """
        return json.loads(subprocess.check_output([str(NODE),'--input-type=module','-e',setup+script,(ROOT/'browser_ui/anatomy-demo.js').as_uri()],text=True))

    def test_generated_geometry_is_inside_world_bounds_and_deterministic(self):
        data=self.run_js("""
        console.log(JSON.stringify({points:meshes.flatMap(m=>m.faces.flat()),kinds:[...new Set(meshes.map(m=>m.kind))],tumors:meshes.filter(m=>m.kind==='tumor').length,same:JSON.stringify(meshes)===JSON.stringify(buildAnatomy(bounds)),bad:buildAnatomy([[0,0,0],[0,1,1]])}));
        """)
        xyz=np.asarray(data['points'])
        self.assertTrue(np.isfinite(xyz).all())
        self.assertTrue(((xyz>-50)&(xyz<50)).all())
        self.assertEqual(set(data['kinds']),{'tumor','artery','vein'})
        self.assertEqual(data['tumors'],2)
        self.assertTrue(data['same'])
        self.assertEqual(data['bad'],[])

    def test_cutaway_removes_foreground_and_disappears_beyond_geometry(self):
        data=self.run_js("""
        const near=project(meshes,pose,plane(-60),'cutaway'),mid=project(meshes,pose,plane(0),'cutaway'),far=project(meshes,pose,plane(60),'cutaway');
        console.log(JSON.stringify({whole:project(meshes,pose,null,'transparent').length,near:near.length,mid:mid.length,far:far.length,depths:mid.map(p=>p.depth),caps:mid.filter(p=>p.cap).length}));
        """)
        self.assertEqual(data['whole'],data['near'])
        self.assertGreater(data['mid'],0)
        self.assertEqual(data['far'],0)
        self.assertGreater(data['caps'],0)
        self.assertTrue(all(d>=200-1e-6 for d in data['depths']))

    def test_section_only_is_exactly_on_slice_and_off_slice_is_empty(self):
        data=self.run_js("""
        const section=project(meshes,pose,plane(0),'section');
        console.log(JSON.stringify({section,before:project(meshes,pose,plane(-60),'section'),after:project(meshes,pose,plane(60),'section')}));
        """)
        self.assertGreater(len(data['section']),0)
        self.assertTrue(all(p['cap'] for p in data['section']))
        np.testing.assert_allclose([p['depth'] for p in data['section']],200,atol=1e-6)
        self.assertEqual(data['before'],[])
        self.assertEqual(data['after'],[])

    def test_invalid_or_lost_pose_never_uses_old_geometry(self):
        data=self.run_js("""
        console.log(JSON.stringify([project(meshes,null,plane(0),'transparent'),project(meshes,pose,null,'cutaway'),project(meshes,{...pose,target:pose.eye},plane(0),'cutaway'),project(meshes,pose,{center:[0,0,0],normal:[0,0,0]},'section'),project(meshes,pose,plane(0),'off'),project(meshes,pose,plane(0),'transparent',{tumor:false,artery:false,vein:false})]));
        """)
        self.assertEqual(data,[[]]*6)

    def test_rendering_does_not_mutate_geometry_pose_or_plane(self):
        data=self.run_js("""
        const p=plane(0),before=JSON.stringify({meshes,pose,p});
        for(const mode of ['transparent','cutaway','section'])project(meshes,pose,p,mode);
        console.log(JSON.stringify(before===JSON.stringify({meshes,pose,p})));
        """)
        self.assertTrue(data)

    def test_full_intrinsics_projection_agrees_with_opencv(self):
        data=self.run_js("""
        const p={...pose,K:[[210,0,99],[0,180,88],[0,0,1]]},point=[12,-13,20];
        const m=[{kind:'tumor',color:[255,100,20],faces:[[point,[13,-13,20],[12,-12,20]]]}];
        console.log(JSON.stringify(project(m,p,null,'transparent')[0].pixels));
        """)
        xyz=np.array([[12,-13,20],[13,-13,20],[12,-12,20]],np.float64)
        K=np.array([[210,0,99],[0,180,88],[0,0,1]],np.float64)
        expected=cv2.projectPoints(xyz,np.zeros(3),np.array([0.,0.,200.]),K,np.zeros(5))[0].reshape(-1,2)
        np.testing.assert_allclose(data,expected,atol=1e-9)

    def test_camera_near_plane_projection_is_finite(self):
        data=self.run_js("""
        const p={...pose,eye:[0,0,0],target:[0,0,1]};
        const polygons=project(meshes,p,null,'transparent');
        console.log(JSON.stringify({count:polygons.length,finite:polygons.every(p=>p.pixels.flat().every(Number.isFinite)),inFront:polygons.every(p=>p.depth>=.1-1e-8)}));
        """)
        self.assertGreater(data['count'],0)
        self.assertTrue(data['finite'])
        self.assertTrue(data['inFront'])


if __name__=='__main__':unittest.main()
