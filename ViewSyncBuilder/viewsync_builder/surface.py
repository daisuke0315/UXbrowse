"""Fixed-camera CPU MVS, mesh/texturing, and self-contained GLB export."""
import json
import shutil
import struct
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pycolmap as pc
from PIL import Image

from . import engine
from .project import sha256, write_json


def ply_counts(path, require_faces=False):
    path=Path(path)
    if not path.is_file():
        raise ValueError('Dense/mesh output is missing; inspect the surface log')
    counts={}
    with path.open('rb') as stream:
        if stream.readline().strip()!=b'ply':
            raise ValueError('Invalid PLY output')
        for _ in range(1000):
            line=stream.readline().decode('ascii').strip()
            if line.startswith('element '):
                _,name,count=line.split();counts[name]=int(count)
            if line=='end_header':
                break
        else:
            raise ValueError('Invalid PLY header')
        if counts.get('vertex',0)<=0 or (require_faces and counts.get('face',0)<=0):
            raise ValueError('Dense point-cloud or mesh is empty; check image overlap and camera poses')
        if stream.tell()>=path.stat().st_size:
            raise ValueError('Truncated PLY output')
    return dict(vertices=counts['vertex'],faces=counts.get('face',0))


def embed_glb_images(source, destination, output_from_source=None):
    """Keep native geometry unchanged, embedding OpenMVS's external texture images."""
    source,destination=Path(source),Path(destination)
    raw=source.read_bytes()
    if len(raw)<20 or struct.unpack_from('<III',raw)!=(0x46546C67,2,len(raw)):
        raise ValueError('Invalid GLB output')
    chunks={};offset=12
    while offset<len(raw):
        size,kind=struct.unpack_from('<II',raw,offset);offset+=8
        if offset+size>len(raw) or kind in chunks:
            raise ValueError('Invalid GLB chunk')
        chunks[kind]=raw[offset:offset+size];offset+=size
    doc=json.loads(chunks[0x4E4F534A]);binary=bytearray(chunks.get(0x004E4942,b''))
    if not doc.get('buffers') or any('uri' in b and not b['uri'].startswith('data:') for b in doc['buffers']):
        raise ValueError('GLB has external geometry buffers')
    embedded=0
    for image in doc.get('images',[]):
        if 'uri' not in image:
            continue
        uri=image['uri'];parsed=urlparse(uri)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError('External/remote GLB texture is not allowed')
        path=(source.parent/unquote(uri)).resolve()
        if not path.is_relative_to(source.parent.resolve()) or not path.is_file():
            raise ValueError('GLB texture is outside the output directory or missing')
        with Image.open(path) as texture:
            if texture.format not in ('PNG','JPEG'):
                raise ValueError('Unsupported GLB texture format')
            mime='image/png' if texture.format=='PNG' else 'image/jpeg'
        data=path.read_bytes();binary.extend(b'\0'*((-len(binary))%4))
        view=dict(buffer=0,byteOffset=len(binary),byteLength=len(data))
        views=doc.setdefault('bufferViews',[]);image['bufferView']=len(views);views.append(view)
        image['mimeType']=mime;del image['uri'];binary.extend(data);embedded+=1
    doc['buffers'][0]['byteLength']=len(binary)
    if output_from_source is not None:
        nodes=doc.setdefault('nodes',[])
        for scene in doc.setdefault('scenes',[{'nodes':[]} ]):
            root=dict(name='source_to_glTF_Y_up_m',children=list(scene.get('nodes',[])),
                      matrix=np.asarray(output_from_source).flatten(order='F').tolist())
            scene['nodes']=[len(nodes)];nodes.append(root)
        doc.setdefault('scene',0)
    text=json.dumps(doc,separators=(',',':')).encode();text+=b' '*((-len(text))%4)
    binary.extend(b'\0'*((-len(binary))%4))
    length=12+8+len(text)+8+len(binary)
    destination.write_bytes(struct.pack('<III',0x46546C67,2,length)+
        struct.pack('<II',len(text),0x4E4F534A)+text+
        struct.pack('<II',len(binary),0x004E4942)+binary)
    return dict(embedded_images=embedded,images=len(doc.get('images',[])),bytes=length)


def verify_fixed_cameras(before, after):
    source={im.name:im for im in before.images.values() if im.has_pose}
    target={im.name:im for im in after.images.values() if im.has_pose}
    if source.keys()!=target.keys():
        raise ValueError('Undistortion changed the registered camera set')
    for name,image in source.items():
        if not np.allclose(image.cam_from_world().matrix(),target[name].cam_from_world().matrix(),atol=1e-8,rtol=1e-10):
            raise ValueError('Undistortion changed a fixed camera pose')


def build_surface(run_dir, image_dir, config, coordinate_system, ba, progress, stop_after='texture'):
    run_dir,image_dir=Path(run_dir),Path(image_dir)
    if coordinate_system=='DICOM_LPS' and (not ba or not ba['converged'] or
            ba['gcp_outlier_observations'] or ba['gcp_reprojection_px']['failures']):
        raise ValueError('Successful CT-constrained BA is required before surface generation')
    model_dir=run_dir/('ct_ba' if coordinate_system=='DICOM_LPS' else 'sparse_initial')
    source=pc.Reconstruction(model_dir)
    work=run_dir/'surface';work.mkdir(exist_ok=True)
    log=work/'surface.log'
    threads=config['threads'];resolution=config['surface_max_size']
    for name in engine.MVS_HASHES:
        engine.mvs_executable(name)
    hashes={name:sha256(model_dir/name) for name in ('cameras.bin','images.bin','points3D.bin')}
    manifest=dict(engine='OpenMVS '+engine.MVS_VERSION,engine_hashes=engine.MVS_HASHES,
        coordinate_system=coordinate_system,unit='mm' if coordinate_system=='DICOM_LPS' else 'arbitrary',
        camera_source=str(model_dir.relative_to(run_dir)),source_model_hashes=hashes,
        fixed_camera_poses=True,camera_pose_reestimation=False,settings=config,files={})
    manifest['processing_policy']=dict(coordinate_normalization=False,tower_points=False,
        mesh_smoothing_iterations=0,hole_closing=0,global_seam_leveling=False,local_seam_leveling=False)
    write_json(work/'manifest.json',manifest)
    progress('dense')
    undistorted=work/'undistorted'
    engine.run('image_undistorter',dict(image_path=image_dir,input_path=model_dir,
        output_path=undistorted,output_type='COLMAP',max_image_size=resolution,num_threads=threads),log)
    normalized=pc.Reconstruction(undistorted/'sparse');verify_fixed_cameras(source,normalized)
    args=['--max-threads',threads,'--archive-type',2]
    engine.run_mvs('InterfaceCOLMAP',['-i',undistorted.resolve(),'-o','scene.mvs',*args],work,log)
    engine.run_mvs('DensifyPointCloud',['-i','scene.mvs','-o','dense.mvs',
        '--resolution-level',config['dense_resolution_level'],'--max-resolution',resolution,
        '--min-resolution',min(320,resolution),'--number-views',5,'--number-views-fuse',2,
        '--tower-mode',0,'--normalize-coordinates',0,'--estimate-roi',0,'--crop-to-roi',0,*args],work,log)
    manifest['dense']=ply_counts(work/'dense.ply');manifest['files']['dense']='dense.ply'
    write_json(work/'manifest.json',manifest)
    if stop_after=='dense':
        return manifest
    progress('mesh')
    engine.run_mvs('ReconstructMesh',['-i','dense.mvs','-o','mesh.mvs','--smooth',0,
        '--close-holes',0,'--crop-to-roi',0,'--target-face-num',config['mesh_faces'],*args],work,log)
    manifest['mesh']=ply_counts(work/'mesh.ply',require_faces=True);manifest['files']['mesh']='mesh.ply'
    write_json(work/'manifest.json',manifest)
    if stop_after=='mesh':
        return manifest
    progress('texture')
    texture_args=['--close-holes',0,'--global-seam-leveling',0,'--local-seam-leveling',0,
                  '--max-texture-size',config['texture_size'],*args]
    engine.run_mvs('TextureMesh',['-i','mesh.mvs','-o','textured.mvs','--export-type','obj',*texture_args],work,log)
    obj=work/'textured.obj';mtl=work/'textured.mtl'
    if not obj.is_file() or not mtl.is_file() or 'vt ' not in obj.read_text() or 'map_Kd' not in mtl.read_text():
        raise ValueError('Texture generation did not produce a textured OBJ/MTL')
    engine.run_mvs('TextureMesh',['-i','textured.mvs','-o','textured_export.mvs','--export-type','glb',*texture_args],work,log)
    transform=np.array([[-.001,0,0,0],[0,0,.001,0],[0,.001,0,0],[0,0,0,1]]) if coordinate_system=='DICOM_LPS' else np.eye(4)
    manifest['glb_from_source']=transform.flatten(order='F').tolist()
    manifest['source_from_glb']=np.linalg.inv(transform).flatten(order='F').tolist()
    manifest['glb_unit']='m' if coordinate_system=='DICOM_LPS' else 'arbitrary'
    manifest['glb']=embed_glb_images(work/'textured_export.glb',work/'model.glb',transform)
    if not manifest['glb']['images']:
        raise ValueError('Texture GLB contains no texture images')
    manifest['files'].update(texture='model.glb',obj='textured.obj',mtl='textured.mtl')
    manifest['texture_images']=[line.split(maxsplit=1)[1].strip() for line in mtl.read_text().splitlines() if line.startswith('map_Kd ')]
    for name in manifest['texture_images']:
        path=(work/name).resolve()
        if not path.is_relative_to(work.resolve()) or not path.is_file():
            raise ValueError('Missing or invalid OBJ texture image')
    write_json(work/'manifest.json',manifest)
    return manifest


def export_surface(work, output, manifest):
    work,output=Path(work),Path(output)/'appearance';output.mkdir(parents=True,exist_ok=True)
    for name in {*manifest['files'].values(),*manifest.get('texture_images',[])}:
        shutil.copyfile(work/name,output/name)
    write_json(output/'manifest.json',manifest)
