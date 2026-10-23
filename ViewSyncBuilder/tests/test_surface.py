import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from viewsync_builder.surface import embed_glb_images, ply_counts, verify_fixed_cameras, build_surface


def tiny_glb(path, image_uri='texture.png'):
    doc = dict(asset={'version':'2.0'},buffers=[{'byteLength':4}],
               bufferViews=[{'buffer':0,'byteOffset':0,'byteLength':4}],
               images=[{'uri':image_uri}])
    chunk=json.dumps(doc).encode();chunk+=b' '*((-len(chunk))%4)
    binary=b'ABCD'
    path.write_bytes(struct.pack('<III',0x46546C67,2,12+8+len(chunk)+8+len(binary))+
                     struct.pack('<II',len(chunk),0x4E4F534A)+chunk+
                     struct.pack('<II',len(binary),0x004E4942)+binary)


class SurfaceTests(unittest.TestCase):
    def test_glb_embeds_local_texture_and_preserves_geometry_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);Image.new('RGB',(8,8),'red').save(root/'texture.png')
            tiny_glb(root/'source.glb')
            report=embed_glb_images(root/'source.glb',root/'model.glb')
            self.assertEqual(report['embedded_images'],1)
            raw=(root/'model.glb').read_bytes();size,kind=struct.unpack_from('<II',raw,12)
            doc=json.loads(raw[20:20+size]);image=doc['images'][0]
            self.assertNotIn('uri',image)
            self.assertEqual(image['mimeType'],'image/png')
            binary=raw[28+size:];self.assertEqual(binary[:4],b'ABCD')
            view=doc['bufferViews'][image['bufferView']]
            self.assertEqual(binary[view['byteOffset']:view['byteOffset']+view['byteLength']],(root/'texture.png').read_bytes())

    def test_glb_rejects_remote_and_outside_textures(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for uri in ('https://example.org/texture.png','../secret.png'):
                tiny_glb(root/'source.glb',uri)
                with self.subTest(uri=uri), self.assertRaises(ValueError):
                    embed_glb_images(root/'source.glb',root/'model.glb')

    def test_glb_root_records_ct_mm_to_y_up_m_without_modifying_vertices(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);Image.new('RGB',(8,8),'red').save(root/'texture.png')
            tiny_glb(root/'source.glb')
            matrix=np.array([[-.001,0,0,0],[0,0,.001,0],[0,.001,0,0],[0,0,0,1]])
            embed_glb_images(root/'source.glb',root/'model.glb',matrix)
            raw=(root/'model.glb').read_bytes();size=struct.unpack_from('<I',raw,12)[0]
            doc=json.loads(raw[20:20+size]);node=doc['nodes'][doc['scenes'][0]['nodes'][0]]
            np.testing.assert_allclose(np.array(node['matrix']).reshape(4,4,order='F'),matrix)
            self.assertEqual(raw[28+size:32+size],b'ABCD')

    def test_empty_point_cloud_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'empty.ply'
            path.write_text('ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n')
            with self.assertRaisesRegex(ValueError,'empty'):
                ply_counts(path)

    def test_undistortion_cannot_change_camera_poses(self):
        from test_bundle import fixture
        import pycolmap as pc
        original,_=fixture();changed=pc.Reconstruction(original)
        verify_fixed_cameras(original,changed)
        frame=next(iter(changed.frames.values()));pose=frame.rig_from_world
        frame.rig_from_world=pc.Rigid3d(pose.rotation,pose.translation+np.array([1,0,0]))
        with self.assertRaisesRegex(ValueError,'pose'):
            verify_fixed_cameras(original,changed)

    def test_ct_surface_requires_successful_ba_before_any_engine_runs(self):
        with tempfile.TemporaryDirectory() as temp, patch('viewsync_builder.surface.engine.run') as run:
            with self.assertRaisesRegex(ValueError,'BA'):
                build_surface(Path(temp),Path(temp)/'images',{},'DICOM_LPS',None,lambda *a:None)
            run.assert_not_called()
