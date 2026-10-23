"""UI-independent, restartable Sparse reconstruction and export."""
import csv
import hashlib
import importlib.metadata
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pycolmap as pc

from . import __version__
from . import engine
from .bundle import align_to_ct, constrained_bundle, evaluate_points, statistics
from .ct import annotate_points
from .surface import build_surface, export_surface
from .project import Project, now, sha256, write_json, write_npz, exclusive_project


DEFAULTS = dict(aliked_model='n16rot', max_features=2048, max_image_size=1600,
                threads=2, camera_model='SIMPLE_RADIAL', single_camera=False, camera_params='',
                pairing='sequential', overlap=10, ransac_px=4., gcp_max_error_px=4., refine_intrinsics=False,
                coordinate_mode='ct', surface='texture', surface_max_size=1600,
                dense_resolution_level=1, mesh_faces=0, texture_size=4096)
STAGES = ('features','matches','sfm','alignment','ba','dense','mesh','texture','export')


def settings(values):
    if set(values) - set(DEFAULTS):
        raise ValueError('Unknown reconstruction settings')
    result = {**DEFAULTS, **values}
    if (result['aliked_model'] not in ('n16rot', 'n32') or
            result['pairing'] not in ('sequential', 'exhaustive') or
            result['camera_model'] not in ('SIMPLE_RADIAL', 'PINHOLE', 'OPENCV') or
            result['coordinate_mode'] not in ('ct','sfm_preview') or result['surface'] not in ('sparse','texture')):
        raise ValueError('Unsupported model or pairing')
    for key, low, high in (('max_features', 128, 10000), ('max_image_size', 256, 6000),
                           ('threads', 1, 32), ('overlap', 1, 100), ('surface_max_size',320,4000),
                           ('dense_resolution_level',0,3), ('mesh_faces',0,2000000), ('texture_size',512,8192)):
        if not isinstance(result[key], int) or isinstance(result[key], bool) or not low <= result[key] <= high:
            raise ValueError('Invalid setting: ' + key)
    if not isinstance(result['ransac_px'], (float, int)) or not .1 <= result['ransac_px'] <= 20:
        raise ValueError('Invalid RANSAC error threshold')
    if not isinstance(result['gcp_max_error_px'], (float, int)) or not .1 <= result['gcp_max_error_px'] <= 100:
        raise ValueError('Invalid GCP reprojection error threshold')
    if not isinstance(result['refine_intrinsics'], bool) or not isinstance(result['single_camera'], bool):
        raise ValueError('Camera settings must be booleans')
    if not isinstance(result['camera_params'], str):
        raise ValueError('Camera parameters must be comma-separated numbers')
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def feature_fingerprint(images, config, engine_hash, model_hash):
    extraction = {k: config[k] for k in ('aliked_model', 'max_features', 'max_image_size',
                                        'camera_model', 'single_camera', 'camera_params')}
    return fingerprint(dict(images=images, settings=extraction, engine=engine_hash, model=model_hash))


def seed_features(database, image_dir, names, config, cached_by_name):
    """Reuse inference, while assigning cameras and image IDs in the current DB."""
    if not any(path.exists() for path in cached_by_name.values()):
        return 0
    with pc.Database.open(database):
        pass
    reader = pc.ImageReaderOptions(camera_model=config['camera_model'], camera_params=config['camera_params'])
    pc.import_images(database, image_dir, image_names=names, options=reader,
                     camera_mode=pc.CameraMode.SINGLE if config['single_camera'] else pc.CameraMode.AUTO)
    reused = 0
    with pc.Database.open(database) as db:
        for image in db.read_all_images():
            path = cached_by_name.get(image.name)
            if path is None or not path.exists() or db.exists_keypoints(image.image_id):
                continue
            with np.load(path, allow_pickle=False) as features:
                keypoints, descriptors = features['keypoints'], features['descriptors']
            data = pc.FeatureDescriptorsFloat(getattr(pc.FeatureExtractorType, 'ALIKED_'+config['aliked_model'].upper()), descriptors)
            with pc.DatabaseTransaction(db):
                db.write_keypoints(image.image_id, keypoints)
                db.write_descriptors(image.image_id, pc.FeatureDescriptors.from_float(data))
            reused += 1
    return reused


def export_result(model, feature_dir, output, controls, qc):
    output = Path(output)
    for folder in ('sparse', 'feature_map', 'cameras', 'qc'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    model.write(output/'sparse')
    model.write_text(output/'sparse')
    model.export_PLY(output/'sparse/points.ply')
    cameras, feature_records, rows = [], [], []
    for image in model.images.values():
        if not image.has_pose:
            continue
        camera = model.cameras[image.camera_id]
        matrix = np.eye(4)
        matrix[:3] = image.cam_from_world().matrix()
        cameras.append(dict(image_id=int(image.image_id), name=image.name, camera_id=int(image.camera_id),
                            model=camera.model_name, parameters=camera.params.tolist(),
                            width=camera.width, height=camera.height,
                            T_camera_from_CT_LPS=matrix.tolist(), center_CT_LPS_mm=image.projection_center().tolist()))
        feature_file = Path(feature_dir) / f'{image.image_id}.npz'
        if not feature_file.exists():
            continue
        with np.load(feature_file, allow_pickle=False) as features:
            uv, descriptors = features['uv'], features['descriptors']
        count = len(uv)
        ids = np.full(count, -1, np.int64)
        xyz = np.full((count, 3), np.nan, np.float64)
        errors = np.full(count, np.nan)
        for index in range(count):
            observation = image.points2D[index]
            if observation.has_point3D():
                point = model.points3D[observation.point3D_id]
                ids[index], xyz[index], errors[index] = int(observation.point3D_id), point.xyz, point.error
                rows.append([image.image_id, index, *uv[index], ids[index], *xyz[index], errors[index]])
        np.savez_compressed(output/'feature_map'/f'{image.image_id}.npz',
                            image_id=np.array(image.image_id), keypoint_ids=np.arange(count),
                            uv=uv, descriptors=descriptors, point3d_ids=ids, xyz=xyz,
                            reprojection_error_px=errors, validity=ids >= 0)
        feature_records.append(dict(image_id=int(image.image_id), name=image.name,
                                    file=f'{image.image_id}.npz', keypoints=count,
                                    valid_xyz=int(np.sum(ids >= 0)), score_available=False))
    with (output/'feature_map/observations.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id', 'keypoint_id', 'u', 'v', 'point3d_id', 'X', 'Y', 'Z', 'reprojection_error_px'])
        writer.writerows(rows)
    tracks = {str(pid): [[int(e.image_id), int(e.point2D_idx)] for e in p.track.elements]
              for pid, p in model.points3D.items()}
    write_json(output/'feature_map/tracks.json', tracks)
    write_json(output/'feature_map/manifest.json', dict(schema_version=1, coordinate_system='DICOM_LPS', unit='mm',
               descriptor='ALIKED', score_available=False,
               unavailable_reason='Native COLMAP database does not retain detector scores', images=feature_records))
    write_json(output/'cameras/cameras.json', dict(coordinate_system='DICOM_LPS', unit='mm', cameras=cameras))
    write_json(output/'qc/report.json', qc)
    write_json(output/'control_points.json', controls)
    write_json(output/'coordinates.json', dict(coordinate_system='DICOM_LPS', unit='mm',
               camera_convention='COLMAP right/down/forward', pose_direction='T_camera_from_CT_LPS',
               output_from_CT_LPS=np.eye(4).tolist()))


def start_run(project, config, stop_after='export'):
    if stop_after not in STAGES:
        raise ValueError('Unknown stopping stage')
    config = settings(config)
    selected = [im for im in project.manifest['images'] if im['enabled']]
    if len(selected) < 2:
        raise ValueError('Select at least two images')
    if config['single_camera'] and len({(im['width'], im['height']) for im in selected}) != 1:
        raise ValueError('A shared camera requires identical image dimensions')
    if config['coordinate_mode']=='sfm_preview' and stop_after in ('alignment','ba'):
        raise ValueError('CT alignment/BA requires CT coordinate mode')
    if config['surface']=='sparse' and stop_after in ('dense','mesh','texture'):
        raise ValueError('Select surface generation for Dense/Mesh/Texture stages')
    if config['coordinate_mode']=='ct' and stop_after in ('alignment', 'ba', 'dense','mesh','texture','export'):
        controls = [p for p in project.controls() if p['role'] == 'control' and p.get('enabled', True)]
        if len(controls) < 3:
            raise ValueError('Register at least three non-collinear control GCPs before CT alignment')
    run_id = uuid.uuid4().hex
    run_dir = project.root / 'reconstruction' / run_id
    run_dir.mkdir()
    state = dict(id=run_id, created=now(), status='queued', stage='queued', stop_after=stop_after,
                 settings=config, image_ids=[im['id'] for im in selected], software_version=__version__,
                 controls=annotate_points(project.controls(), project.manifest['ct']),
                 images=selected, ct=project.manifest['ct'], stages={})
    write_json(run_dir/'run.json', state)
    project.manifest['runs'].append(run_id)
    project.save()
    return state


def execute(project_root, run_id):
    with exclusive_project(project_root):
        return _execute(project_root, run_id)


def _execute(project_root, run_id):
    project = Project.open(project_root)
    run_dir = project.file('reconstruction/'+run_id)
    state = json.loads((run_dir/'run.json').read_text())
    config, selected = state['settings'], state['images']
    begun = time.perf_counter()
    log = run_dir/'engine.log'

    def progress(stage, **extra):
        state.update(stage=stage, updated=now(), elapsed_seconds=time.perf_counter()-begun, **extra)
        write_json(run_dir/'run.json', state)

    def complete_stage(stage):
        state['stages'][stage] = dict(completed=now())
        progress(stage)
        return state['stop_after'] == stage

    try:
        progress('features', status='running')
        dependencies = {n: importlib.metadata.version(n) for n in ('pycolmap', 'numpy', 'scipy', 'pydicom')}
        exe_hash = sha256(engine.executable())
        model_name = 'aliked-'+config['aliked_model']+'.onnx'
        engine.model_file(model_name)
        engine.model_file('aliked-lightglue.onnx')
        inputs = []
        for image in selected:
            normalized_hash = sha256(project.file(image['path']))
            if normalized_hash != image['normalized_sha256']:
                raise ValueError('Normalized input changed; re-import as a new image')
            inputs.append(dict(id=image['id'], normalized_sha256=normalized_hash,
                               mask=sha256(project.file(image['mask'])) if image['mask'] else None))
        key = feature_fingerprint(inputs, config, exe_hash, engine.MODEL_HASHES[model_name])
        cache = project.root/'cache/aliked'/key
        cache.mkdir(parents=True, exist_ok=True)
        database = cache/'features.db'
        feature_dir = cache/'features'
        feature_dir.mkdir(exist_ok=True)
        listing = run_dir/'image-list.txt'
        listing.write_text('\n'.join(Path(im['path']).name for im in selected))
        individual_cache = project.root/'cache/aliked/images'
        individual_cache.mkdir(exist_ok=True)
        per_image = {}
        for record, image in zip(inputs, selected):
            extraction = {k: config[k] for k in ('aliked_model', 'max_features', 'max_image_size')}
            image_key = fingerprint(dict(image=record['normalized_sha256'], mask=record['mask'],
                         settings=extraction, engine=exe_hash, model=engine.MODEL_HASHES[model_name]))
            per_image[Path(image['path']).name] = individual_cache/(image_key+'.npz')
        mask_dir = None
        if any(im['mask'] for im in selected):
            from PIL import Image
            mask_dir = run_dir/'masks'
            mask_dir.mkdir(exist_ok=True)
            for image in selected:
                target = mask_dir/(Path(image['path']).name+'.png')
                if image['mask']:
                    shutil.copyfile(project.file(image['mask']), target)
                else:
                    Image.new('L', (image['width'], image['height']), 255).save(target)
        if not (cache/'complete.json').exists():
            reused = seed_features(database, project.root/'input/images',
                                   [Path(im['path']).name for im in selected], config, per_image)
            engine.extract(database, project.root/'input/images', listing, config, log, mask_dir)
            with pc.Database.open(database) as db:
                if {im.name for im in db.read_all_images()} != {Path(im['path']).name for im in selected}:
                    raise ValueError('COLMAP skipped input images; see extraction log')
                for image in db.read_all_images():
                    keypoints = db.read_keypoints(image.image_id)
                    uv = keypoints[:, :2]
                    descriptors = db.read_descriptors(image.image_id).to_float().data
                    if len(uv) == 0:
                        raise ValueError('An image has no usable ALIKED features')
                    np.savez_compressed(feature_dir/f'{image.image_id}.npz', uv=uv, descriptors=descriptors)
                    write_npz(per_image[image.name], keypoints=keypoints, descriptors=descriptors)
                counts = {str(im.image_id): int(db.num_keypoints_for_image(im.image_id)) for im in db.read_all_images()}
            write_json(cache/'complete.json', dict(images=counts, reused_images=reused))
        else:
            # Populate image-level cache from an existing completed group without inference.
            with pc.Database.open(database) as db:
                for image in db.read_all_images():
                    if not per_image[image.name].exists():
                        write_npz(per_image[image.name], keypoints=db.read_keypoints(image.image_id),
                                  descriptors=db.read_descriptors(image.image_id).to_float().data)
        progress('features', feature_cache=key, dependencies=dependencies, engine_sha256=exe_hash,
                 model_hashes={model_name: engine.MODEL_HASHES[model_name],
                               'aliked-lightglue.onnx': engine.MODEL_HASHES['aliked-lightglue.onnx']},
                 feature_counts=json.loads((cache/'complete.json').read_text())['images'])
        if complete_stage('features'):
            progress('features', status='completed'); return state
        progress('matches')
        pair_key = fingerprint(dict(feature_cache=key, pairing=config['pairing'], overlap=config['overlap'],
                                     ransac_px=config['ransac_px'], lightglue=engine.MODEL_HASHES['aliked-lightglue.onnx']))
        match_cache = project.root/'cache/lightglue'/pair_key
        match_cache.mkdir(exist_ok=True)
        matched_db = match_cache/'matches.db'
        if not (match_cache/'complete.json').exists():
            if not matched_db.exists():
                shutil.copyfile(database, matched_db)
            engine.match(matched_db, config, log)
            with pc.Database.open(matched_db) as db:
                write_json(match_cache/'complete.json', dict(verified_pairs=int(db.num_verified_image_pairs()),
                           inlier_matches=int(db.num_inlier_matches()), matches=int(db.num_matches())))
        matching = json.loads((match_cache/'complete.json').read_text())
        if matching['verified_pairs'] == 0:
            raise ValueError('No geometrically verified image pairs; check overlap and images')
        progress('matches', matching=matching)
        if complete_stage('matches'):
            progress('matches', status='completed'); return state
        progress('sfm')
        sparse_cache = match_cache/('sparse_initial-'+pc.__version__)
        if not (sparse_cache/'complete.json').exists():
            models = pc.incremental_mapping(matched_db, project.root/'input/images', sparse_cache,
                                            options=pc.IncrementalPipelineOptions(num_threads=config['threads'], random_seed=0))
            if not models:
                raise ValueError('SfM did not produce a connected Sparse model')
            best_id = max(models, key=lambda n: (models[n].num_reg_images(), models[n].num_points3D()))
            write_json(sparse_cache/'complete.json', dict(selected=int(best_id), components=len(models)))
        chosen = json.loads((sparse_cache/'complete.json').read_text())
        model = pc.Reconstruction(sparse_cache/str(chosen['selected']))
        initial = run_dir/'sparse_initial'
        initial.mkdir(exist_ok=True)
        model.write(initial)
        registered = {im.name for im in model.images.values() if im.has_pose}
        progress('sfm', registered_images=len(registered), input_images=len(selected), points3d=model.num_points3D(),
                 registration_rate=len(registered)/len(selected), components=chosen['components'],
                 failed_image_ids=[im['id'] for im in selected if Path(im['path']).name not in registered])
        if complete_stage('sfm'):
            progress('sfm', status='completed'); return state
        coordinate_system = 'DICOM_LPS' if config['coordinate_mode']=='ct' else 'SfM_arbitrary'
        if config['coordinate_mode']=='ct':
            by_id = {im['id']: Path(im['path']).name for im in selected}
            controls = []
            for point in state['controls']:
                controls.append({**point, 'observations': [dict(image_id=by_id[o['image_id']], uv=o['uv'])
                                 for o in point.get('observations', []) if o['image_id'] in by_id]})
            progress('alignment')
            aligned, alignment = align_to_ct(model, controls)
            (run_dir/'ct_aligned').mkdir(exist_ok=True)
            aligned.write(run_dir/'ct_aligned')
            write_json(run_dir/'alignment.json', alignment)
            if complete_stage('alignment'):
                progress('alignment', status='completed'); return state
            progress('ba')
            optimized, ba = constrained_bundle(aligned, controls, config['refine_intrinsics'],
                                                gcp_max_error_px=config['gcp_max_error_px'])
            write_json(run_dir/'ba.json', ba)
            (run_dir/'ct_ba').mkdir(exist_ok=True)
            optimized.write(run_dir/'ct_ba')
            if (not ba['converged'] or ba['gcp_reprojection_px']['failures'] or
                    ba['after_reprojection_px']['failures'] or ba['gcp_outlier_observations']):
                write_json(run_dir/'qc.json', dict(accuracy_status='failed', bundle_adjustment=ba,
                           registration_rate=state['registration_rate'], matching=matching,
                           evaluation_count=sum(p['role']=='evaluation' and p.get('enabled', True) for p in controls),
                           tre_mm=statistics([]), evaluation_reprojection_px=statistics([]), points=[],
                           ct_regularization=(state['ct'] or {}).get('regularization'),
                           performance_target_validated=False, reason='Bundle adjustment QC failed; TRE not evaluated'))
                raise ValueError('GCP-constrained BA did not converge, or a GCP observation failed QC; inspect ba.json')
            complete_stage('ba')
            qc = {**evaluate_points(optimized, controls), 'bundle_adjustment': ba, 'initial_alignment': alignment,
                  'registration_rate': state['registration_rate'], 'matching': matching,
                  'ct_regularization': (state['ct'] or {}).get('regularization'),
                  'coordinate_system': 'DICOM_LPS', 'unit': 'mm', 'display_overlay_error': 'not_measured'}
            write_json(run_dir/'qc.json', qc)
            if state['stop_after'] == 'ba':
                progress('ba', status='completed', accuracy_status=qc['accuracy_status']); return state
        else:
            optimized=model;ba=None;controls=[]
            qc=dict(accuracy_status='unverified',evaluation_count=0,tre_mm=statistics([]),
                evaluation_reprojection_px=statistics([]),points=[],performance_target_validated=False,
                coordinate_system='SfM_arbitrary',unit='arbitrary',registration_rate=state['registration_rate'],
                reason='Photo-only surface preview; no CT alignment or physical accuracy validation')
            write_json(run_dir/'qc.json',qc)
        surface=None
        if config['surface']=='texture':
            def surface_progress(stage):
                if stage=='mesh':complete_stage('dense')
                elif stage=='texture':complete_stage('mesh')
                progress(stage)
            target=state['stop_after'] if state['stop_after'] in ('dense','mesh','texture') else 'texture'
            surface=build_surface(run_dir,project.root/'input/images',config,coordinate_system,ba,surface_progress,target)
            complete_stage(target)
            progress(target,surface=surface)
        progress('export')
        output = project.root/'output'/run_id
        if config['coordinate_mode']=='ct':
            export_result(optimized,feature_dir,output,controls,qc)
        else:
            output.mkdir(parents=True,exist_ok=True)
            (output/'sparse').mkdir()
            optimized.write(output/'sparse');optimized.write_text(output/'sparse')
            optimized.export_PLY(output/'sparse/points.ply')
            write_json(output/'qc/report.json',qc)
            write_json(output/'coordinates.json',dict(coordinate_system='SfM_arbitrary',unit='arbitrary',
                output_from_SfM=np.eye(4).tolist(),ct_alignment=False))
        if surface:
            export_surface(run_dir/'surface',output,surface)
            coordinates=json.loads((output/'coordinates.json').read_text())
            coordinates.update(glb_from_source=surface.get('glb_from_source'),glb_unit=surface.get('glb_unit'))
            write_json(output/'coordinates.json',coordinates)
        write_json(output/'ct_provenance.json', state['ct'])
        if config['coordinate_mode']=='ct':write_json(output/'alignment.json',alignment)
        complete_stage('export')
        progress('export', status='completed', accuracy_status=qc['accuracy_status'], output=str(output.relative_to(project.root)))
        write_json(output/'run.json', state)
        return state
    except Exception as error:
        progress(state.get('stage', 'unknown'), status='failed', error=str(error))
        raise
