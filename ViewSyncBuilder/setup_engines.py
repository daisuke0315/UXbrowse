"""Developer-only download of pinned official engines/models; not called by the app."""
import hashlib
import platform
import shutil
import ssl
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import certifi

from viewsync_builder.engine import DATA, MODEL_HASHES, MVS_ARCHIVE_SHA256, MVS_SOURCE_SHA256


def download(url, destination, digest):
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        name = Path(temporary.name)
        try:
            with urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where())) as source:
                shutil.copyfileobj(source, temporary)
            temporary.flush()
            if hashlib.sha256(name.read_bytes()).hexdigest() != digest:
                raise ValueError('Downloaded asset hash mismatch: '+destination.name)
            name.replace(destination)
        finally:
            name.unlink(missing_ok=True)


def main():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('The pinned engine package targets macOS arm64')
    model_dir = DATA/'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in MODEL_HASHES.items():
        print('Verify/download '+name, flush=True)
        download('https://github.com/colmap/colmap/releases/download/3.13.0/'+name, model_dir/name, digest)
    engine_dir = DATA/'engines'
    engine_dir.mkdir(exist_ok=True)
    archive = engine_dir/'colmap-4.2.1-arm64.zip'
    download('https://github.com/colmap/colmap/releases/download/4.2.1/colmap-arm64-macos.zip', archive,
             '1fe9d65bddd61e0598f3087641cb14c4d4d4c74529e9f58911e8a5238a140a04')
    destination = engine_dir/'colmap-4.2.1'
    if (destination/'COLMAP.app').exists():
        print('Engine already present; preserving it. Verify with the doctor command.')
    else:
        with tempfile.TemporaryDirectory(dir=engine_dir) as temporary:
            subprocess.run(['/usr/bin/ditto', '-xk', str(archive), temporary], check=True)
            subprocess.run(['/usr/bin/ditto', '-xk', str(Path(temporary)/'COLMAP-mac.zip'), str(destination)], check=True)
    mvs=engine_dir/'openmvs-2.4.0'
    mvs.mkdir(exist_ok=True)
    archive=engine_dir/'OpenMVS-2.4.0-macOS-arm64.zip'
    download('https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_macOS_arm64.zip',archive,MVS_ARCHIVE_SHA256)
    if not (mvs/'DensifyPointCloud').exists():
        subprocess.run(['/usr/bin/ditto','-xk',str(archive),str(mvs)],check=True)
    source=mvs/'openMVS-v2.4.0-source.zip'
    download('https://codeload.github.com/cdcseacave/openMVS/zip/refs/tags/v2.4.0',source,MVS_SOURCE_SHA256)


if __name__ == '__main__':
    main()
