"""Build a local macOS arm64 developer preview, never include research inputs."""
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from viewsync_builder.engine import MODEL_HASHES, executable, model_file, mvs_executable, MVS_HASHES, MVS_SOURCE_SHA256
from viewsync_builder import __version__
from viewsync_builder.project import sha256


def main():
    root = Path(__file__).resolve().parent
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('This package definition is currently verified only on macOS arm64')
    # Build into a new directory, preserving every prior package.
    from datetime import datetime
    output = root.parent/'data/ViewSyncBuilder/packages'/datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(output/'pyinstaller-config'))
    clt = Path('/Library/Developer/CommandLineTools/usr/bin')
    if clt.is_dir():
        environment['PATH'] = str(clt) + os.pathsep + os.environ.get('PATH', '')
        environment['DEVELOPER_DIR'] = '/Library/Developer/CommandLineTools'
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onedir',
        '--name', 'ViewSyncBuilder', '--distpath', str(output), '--workpath', str(output/'build'),
        '--specpath', str(output/'build'), '--paths', str(root),
        '--collect-all', 'pycolmap', '--collect-submodules', 'uvicorn',
        '--copy-metadata', 'numpy', '--copy-metadata', 'scipy',
        '--copy-metadata', 'pydicom', '--copy-metadata', 'pycolmap',
        '--add-data', str(root/'frontend')+':frontend',
        '--add-data', str(root.parent/'vendor')+':vendor', str(root/'launch.py')], check=True, cwd=root, env=environment)
    package = output/'ViewSyncBuilder'
    resources = package/'resources'
    resources.mkdir()
    engine_dir = resources/'engines/colmap-4.2.1'
    engine_dir.mkdir(parents=True)
    shutil.copytree(executable().parents[2], engine_dir/'COLMAP.app', symlinks=True)
    mvs=resources/'engines/openmvs-2.4.0';mvs.mkdir()
    for name in MVS_HASHES:
        shutil.copyfile(mvs_executable(name),mvs/name);(mvs/name).chmod(0o755)
    source=mvs_executable('TextureMesh').parent/'openMVS-v2.4.0-source.zip'
    if not source.is_file() or sha256(source)!=MVS_SOURCE_SHA256:
        raise ValueError('Verified OpenMVS corresponding source archive is required')
    shutil.copyfile(source,mvs/source.name)
    import zipfile
    with zipfile.ZipFile(source) as archive:
        for name in ('LICENSE','COPYRIGHT.md'):
            (mvs/name).write_bytes(archive.read('openMVS-2.4.0/'+name))
    (resources/'models').mkdir()
    for name in MODEL_HASHES:
        shutil.copyfile(model_file(name), resources/'models'/name)
    licenses = package/'licenses'
    licenses.mkdir()
    for distribution in importlib.metadata.distributions():
        for item in distribution.files or []:
            if any('license' in part.lower() or 'copying' in part.lower() or 'notice' in part.lower() for part in item.parts):
                source = Path(distribution.locate_file(item))
                if source.is_file():
                    target = licenses/distribution.metadata['Name']/item.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    shutil.copyfile(root/'docs/distribution.md', package/'READ-ME-FIRST.md')
    shutil.copyfile(root/'requirements-lock.txt', package/'requirements-lock.txt')
    manifest = dict(platform='macOS arm64', software_version=__version__, developer_preview=True,
                    models=MODEL_HASHES, colmap_sha256=sha256(executable()),openmvs=MVS_HASHES,
                    openmvs_source_sha256=MVS_SOURCE_SHA256,openmvs_license='AGPL-3.0; third-party restrictions require review',
                    clean_machine_acceptance_test=False, license_audit_complete=False)
    (package/'bundle-manifest.json').write_text(json.dumps(manifest, indent=2))
    launcher = package/'Start ViewSync Builder.command'
    launcher.write_text('#!/bin/zsh\nset -eu\ncd -- "$(dirname -- "$0")"\nexec ./ViewSyncBuilder serve "$@"\n')
    launcher.chmod(0o755)
    # Retain the unpacked build for local inspection; archive contains only application resources.
    archive = output/'ViewSyncBuilder-macOS-arm64-preview.zip'
    subprocess.run(['/usr/bin/ditto', '-c', '-k', '--keepParent', str(package), str(archive)], check=True)
    print(archive, flush=True)


if __name__ == '__main__':
    main()
