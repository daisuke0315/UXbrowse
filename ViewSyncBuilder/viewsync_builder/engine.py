"""Pinned COLMAP CLI adapter. Runtime never downloads models."""
import os
import subprocess
import sys
from pathlib import Path

from .project import sha256


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/ViewSyncBuilder'
MODEL_HASHES = {
    'aliked-n16rot.onnx': '39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547',
    'aliked-n32.onnx': 'a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c',
    'aliked-lightglue.onnx': 'b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d',
}
MVS_VERSION = '2.4.0'
MVS_ARCHIVE_SHA256 = '3d4c616c97031b1ab6e2eecb0ddd5614fb99513c0782a32c9350602faf38799b'
MVS_SOURCE_SHA256 = 'c6284166e105d2332fb1a13c2c162d0df85f7fc82da9ab253c83522e69e8b9ea'
MVS_HASHES = {
    'InterfaceCOLMAP':'34d7ee529fe232dc26e23db53097ad4567aa502e9e42f901d1cb1baf613a4d98',
    'DensifyPointCloud':'8ea970b0349270754d23f31a11dea87417c06d1001fa1ef437ee69b6146a1ce0',
    'ReconstructMesh':'05f0eae93628ce50fa481db2879e9f62d1636b68f06422f76fd8b5d8d4773b0e',
    'TextureMesh':'02887713214539f59286f8a2af430a48f6e237a5a98c8a83d050489bc426837f',
}


def mvs_executable(name):
    path = resources()/'engines'/('openmvs-'+MVS_VERSION)/name
    if name not in MVS_HASHES or not path.is_file() or sha256(path) != MVS_HASHES[name]:
        raise ValueError('Verified OpenMVS engine is missing: '+name)
    return path


def run_mvs(name, arguments, work, log):
    argv = [str(mvs_executable(name)), *map(str, arguments)]
    with Path(log).open('a', encoding='utf-8') as output:
        output.write('\nOpenMVS '+name+'\n');output.flush()
        result = subprocess.run(argv, cwd=work, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'OpenMVS {name} failed (exit {result.returncode}); see run log')


def resources():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent / 'resources'
    return DATA


def default_projects():
    return (Path(sys.executable).resolve().parent/'Projects' if getattr(sys, 'frozen', False)
            else DATA/'Projects')


def executable():
    configured = os.environ.get('VIEWSYNC_COLMAP')
    if configured:
        return Path(configured).resolve()
    return resources() / 'engines/colmap-4.2.1/COLMAP.app/Contents/MacOS/colmap'


def model_file(name):
    path = resources() / 'models' / name
    if not path.is_file() or sha256(path) != MODEL_HASHES[name]:
        raise ValueError('Required verified model is missing: ' + name)
    return path


def capabilities():
    exe = executable()
    result = dict(colmap=exe.is_file(), aliked=False, lightglue=False, dense=False,
                  texture=False, models={}, issues=[])
    if exe.is_file():
        help_text = subprocess.run([str(exe), 'feature_extractor', '-h'], capture_output=True, text=True, timeout=15)
        text = help_text.stdout + help_text.stderr
        result['aliked'] = help_text.returncode == 0 and 'AlikedExtraction' in text
        version = subprocess.run([str(exe), '-h'], capture_output=True, text=True, timeout=15)
        result['colmap_cuda'] = 'with CUDA' in version.stdout + version.stderr
        result['engine_version'] = (version.stdout + version.stderr).splitlines()[0]
    for name in MODEL_HASHES:
        try:
            model_file(name)
            result['models'][name] = True
        except ValueError:
            result['models'][name] = False
    result['lightglue'] = result['aliked'] and result['models']['aliked-lightglue.onnx']
    try:
        for name in MVS_HASHES:
            mvs_executable(name)
        result.update(dense=True, texture=True, surface_engine='OpenMVS '+MVS_VERSION+' / CPU')
    except ValueError as error:
        result['issues'].append(str(error))
    result['issues'].append('COLMAP database does not retain ALIKED detector scores or per-match LightGlue confidence.')
    return result


def run(command, arguments, log):
    exe = executable()
    if not exe.is_file():
        raise ValueError('COLMAP engine is missing')
    argv = [str(exe), command, '--log_target', 'stderr', '--log_color', '0']
    for key, value in arguments.items():
        argv.extend(['--'+key, str(value)])
    with Path(log).open('a', encoding='utf-8') as output:
        completed = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if completed.returncode:
        raise RuntimeError(f'COLMAP {command} failed (exit {completed.returncode}); see run log')


def extract(database, images, image_list, settings, log, mask_path=None):
    model = settings['aliked_model']
    if model not in ('n16rot', 'n32'):
        raise ValueError('Select ALIKED n16rot or n32')
    arguments = dict(database_path=database, image_path=images, image_list_path=image_list,
                     **{'FeatureExtraction.type': 'ALIKED_'+model.upper(),
                        'FeatureExtraction.use_gpu': 0, 'FeatureExtraction.num_threads': settings['threads'],
                        'FeatureExtraction.max_image_size': settings['max_image_size'],
                        'AlikedExtraction.max_num_features': settings['max_features'],
                        'AlikedExtraction.'+model+'_model_path': model_file('aliked-'+model+'.onnx'),
                        'ImageReader.camera_model': settings['camera_model'],
                        'ImageReader.single_camera': int(settings['single_camera'])})
    if settings.get('camera_params'):
        arguments['ImageReader.camera_params'] = settings['camera_params']
    if mask_path:
        arguments['ImageReader.mask_path'] = mask_path
    run('feature_extractor', arguments, log)


def match(database, settings, log):
    arguments = dict(database_path=database, **{
        'FeatureMatching.type': 'ALIKED_LIGHTGLUE', 'FeatureMatching.use_gpu': 0,
        'FeatureMatching.num_threads': settings['threads'],
        'AlikedMatching.lightglue_model_path': model_file('aliked-lightglue.onnx'),
        'TwoViewGeometry.max_error': settings['ransac_px'], 'TwoViewGeometry.random_seed': 0})
    if settings['pairing'] == 'sequential':
        arguments['SequentialMatching.overlap'] = settings['overlap']
        arguments['SequentialMatching.quadratic_overlap'] = 1
        run('sequential_matcher', arguments, log)
    elif settings['pairing'] == 'exhaustive':
        run('exhaustive_matcher', arguments, log)
    else:
        raise ValueError('Unsupported pairing method')
