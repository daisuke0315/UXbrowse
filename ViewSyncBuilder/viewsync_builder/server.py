"""Loopback-only Web API. Reconstruction workers run outside the HTTP process."""
import io
import json
import os
import secrets
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pycolmap as pc
from fastapi import Body, FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from . import engine
from .ct import CTVolume, annotate_points
from .pipeline import start_run
from .project import Project, now, sha256, write_json


def create_app(storage=None):
    storage = Path(storage or engine.resources()/'settings').resolve()
    storage.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    active, volume, worker = None, None, None
    worker_run = None

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if worker is not None and worker.poll() is None:
                try:
                    os.killpg(worker.pid, signal.SIGTERM)
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
                except ProcessLookupError:
                    pass

    app = FastAPI(title='ViewSync Builder', docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.middleware('http')
    async def local_only(request, call_next):
        host = urlparse('http://'+request.headers.get('host', '')).hostname
        if host not in ('127.0.0.1', 'localhost', '[::1]', '::1'):
            return JSONResponse({'error': 'Loopback host required'}, status_code=403)
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            if origin and origin != str(request.base_url).rstrip('/'):
                return JSONResponse({'error': 'Same-origin request required'}, status_code=403)
            if request.headers.get('x-viewsync-token') != token:
                return JSONResponse({'error': 'Local session token required'}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    @app.exception_handler(FileExistsError)
    @app.exception_handler(FileNotFoundError)
    def invalid(request, error):
        return JSONResponse({'error': str(error)}, status_code=400)

    def project(edit=False):
        nonlocal active
        if active is None:
            raise ValueError('Create or open a project first')
        if edit and worker is not None and worker.poll() is None:
            raise ValueError('Cancel or finish the active job before editing inputs')
        active = Project.open(active.root)
        return active

    def run_state(run_id):
        nonlocal worker, worker_run
        p = project()
        if run_id not in p.manifest['runs']:
            raise ValueError('Unknown run')
        path = p.root/'reconstruction'/run_id/'run.json'
        state = json.loads(path.read_text())
        if state['status'] in ('running', 'queued') and (worker is None or worker.poll() is not None or worker_run != run_id):
            state.update(status='interrupted', error='Worker stopped; start a new run to reuse completed caches')
            write_json(path, state)
        if state['status'] == 'running':
            state['elapsed_seconds'] = (datetime.now(timezone.utc)-datetime.fromisoformat(state['created'])).total_seconds()
            engine_log = path.parent/'engine.log'
            if engine_log.exists():
                with engine_log.open('rb') as stream:
                    stream.seek(max(0, engine_log.stat().st_size-12000))
                    processed = re.findall(rb'Processed file \[(\d+)/(\d+)\]', stream.read())
                if processed:
                    state['processed_images'], state['total_images'] = map(int, processed[-1])
        return state

    @app.get('/api/session')
    def session():
        recent = json.loads((storage/'recent.json').read_text()) if (storage/'recent.json').exists() else []
        return dict(token=token, default_projects=str(engine.default_projects()), recent=recent)

    @app.get('/api/capabilities')
    def capabilities():
        return engine.capabilities()

    @app.get('/api/project')
    def get_project():
        if active is None:
            return None
        p = project()
        return dict(path=str(p.root), manifest=p.manifest, points=p.controls())

    @app.post('/api/project')
    def set_project(data: dict = Body(...)):
        nonlocal active, volume
        if worker is not None and worker.poll() is None:
            raise ValueError('Finish or cancel the active job first')
        path = Path(data['path']).expanduser()
        if not path.is_absolute():
            raise ValueError('Choose an absolute project folder')
        if data.get('mode') == 'create':
            active = Project.create(path, data['name'])
        elif data.get('mode') == 'open':
            active = Project.open(path)
        else:
            raise ValueError('Project mode must be create or open')
        volume = None
        recent = json.loads((storage/'recent.json').read_text()) if (storage/'recent.json').exists() else []
        write_json(storage/'recent.json', [str(active.root)] + [p for p in recent if p != str(active.root)][:9])
        return get_project()

    @app.post('/api/images')
    def import_images(files: list[UploadFile] = File(...)):
        p = project(edit=True)
        records = []
        with tempfile.TemporaryDirectory(prefix='viewsync-images-') as folder:
            for uploaded in files:
                target = Path(folder)/Path(uploaded.filename).name
                with target.open('wb') as stream:
                    shutil.copyfileobj(uploaded.file, stream)
                records.append(p.import_image(target))
        return records

    @app.patch('/api/images/{image_id}')
    def set_image(image_id: str, data: dict = Body(...)):
        p = project(edit=True)
        image = next((im for im in p.manifest['images'] if im['id'] == image_id), None)
        if image is None:
            raise ValueError('Unknown image')
        if 'enabled' in data:
            if not isinstance(data['enabled'], bool):
                raise ValueError('enabled must be a boolean')
            image['enabled'] = data['enabled']
        if 'roi' in data:
            roi = data['roi']
            if roi is None:
                image['mask'] = None
            else:
                xy = np.asarray(roi, float)
                if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
                    raise ValueError('ROI needs at least three finite UV vertices')
                if np.any(xy < 0) or np.any(xy[:, 0] >= image['width']) or np.any(xy[:, 1] >= image['height']):
                    raise ValueError('ROI outside image')
                mask = Image.new('L', (image['width'], image['height']))
                ImageDraw.Draw(mask).polygon([tuple(v) for v in xy], fill=255)
                relative = f'control/masks/{image_id}-{uuid.uuid4().hex}.png'
                p.file(relative).parent.mkdir(exist_ok=True)
                mask.save(p.file(relative))
                image['mask'] = relative
        p.save()
        return image

    @app.get('/api/points')
    def points():
        return project().controls()

    @app.post('/api/points')
    def set_points(points: list = Body(...)):
        p = project(edit=True)
        p.set_controls(annotate_points(points, p.manifest['ct']))
        return p.controls()

    @app.post('/api/ct/series')
    def ct_series(data: dict = Body(...)):
        groups = CTVolume.series(Path(data['path']).expanduser())
        return [dict(series_uid=uid, slices=len(items),
                     series_number=str(getattr(items[0][1], 'SeriesNumber', '')),
                     description=str(getattr(items[0][1], 'SeriesDescription', '')))
                for uid, items in groups.items()]

    @app.post('/api/ct')
    def import_ct(data: dict = Body(...)):
        nonlocal volume
        p = project(edit=True)
        source = Path(data['path']).expanduser()
        loaded = CTVolume.load(source, data.get('series_uid'))
        items = CTVolume.series(source)[loaded.metadata['series_uid']]
        import_id = uuid.uuid4().hex
        directory = p.root/'input/ct'/import_id
        directory.mkdir()
        hashes = []
        for index, (path, _) in enumerate(items):
            dest = directory/f'{index:06d}.dcm'
            shutil.copyfile(path, dest)
            hashes.append(sha256(dest))
        cache = p.root/'cache'/f'ct-{import_id}.npz'
        np.savez_compressed(cache, voxels=loaded.voxels, affine=loaded.affine)
        p.manifest['ct'] = {**loaded.metadata, 'cache':str(cache.relative_to(p.root)),
                            'source_hashes':hashes, 'input':str(directory.relative_to(p.root))}
        p.save()
        p.set_controls(annotate_points(p.controls(), p.manifest['ct']))
        volume = loaded
        return ct_info()

    def get_volume():
        nonlocal volume
        p = project()
        if not p.manifest['ct']:
            raise ValueError('Import CT first')
        if volume is None:
            with np.load(p.file(p.manifest['ct']['cache']), allow_pickle=False) as arrays:
                volume = CTVolume(arrays['voxels'], arrays['affine'], p.manifest['ct'])
        return volume

    @app.get('/api/ct')
    def ct_info():
        ct = get_volume()
        return dict(metadata=ct.metadata, bounds=ct.bounds.tolist())

    @app.get('/api/ct/mpr')
    def ct_mpr(plane: str, position: float, level: float = 40, width: float = 400):
        image, mapping = get_volume().mpr(plane, position, level=level, width=width)
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        return Response(stream.getvalue(), media_type='image/png', headers={'X-MPR-Mapping':json.dumps(mapping)})

    @app.get('/api/file')
    def file(path: str):
        result = project().file(path)
        if not result.is_file():
            raise ValueError('File does not exist')
        return FileResponse(result)

    @app.post('/api/jobs')
    def start_job(data: dict = Body(...)):
        nonlocal worker, worker_run
        p = project(edit=True)
        state = start_run(p, data.get('settings', {}), data.get('stop_after', 'export'))
        run_dir = p.root/'reconstruction'/state['id']
        if getattr(sys, 'frozen', False):
            argv = [sys.executable, 'worker', str(p.root), state['id']]
        else:
            argv = [sys.executable, '-m', 'viewsync_builder', 'worker', str(p.root), state['id']]
        try:
            with (run_dir/'worker.log').open('a') as log:
                worker = subprocess.Popen(argv, cwd=Path(__file__).resolve().parents[1],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            worker_run = state['id']
        except Exception:
            state.update(status='failed', error='Could not launch reconstruction worker')
            write_json(run_dir/'run.json', state)
            raise
        return state

    @app.get('/api/jobs')
    def jobs():
        return [run_state(run_id) for run_id in reversed(project().manifest['runs'])]

    @app.get('/api/jobs/{run_id}')
    def job(run_id: str):
        return run_state(run_id)

    @app.post('/api/jobs/{run_id}/cancel')
    def cancel(run_id: str):
        nonlocal worker
        state = run_state(run_id)
        if worker_run == run_id and worker is not None and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            state.update(status='cancelled', updated=now())
            write_json(project().root/'reconstruction'/run_id/'run.json', state)
        return state

    @app.get('/api/jobs/{run_id}/log')
    def logs(run_id: str):
        run_state(run_id)
        root = project().root/'reconstruction'/run_id
        result = {}
        for name in ('engine.log', 'worker.log', 'surface/surface.log'):
            path = root/name
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size-16000))
                    result[name] = stream.read().decode('utf-8', errors='replace')
        return result

    @app.get('/api/results/{run_id}')
    def results(run_id: str):
        state = run_state(run_id)
        root = project().root/'reconstruction'/run_id
        model_path = root/'ct_ba' if (root/'ct_ba/cameras.bin').exists() else root/'sparse_initial'
        if not (model_path/'cameras.bin').exists():
            raise ValueError('No Sparse model available yet')
        model = pc.Reconstruction(model_path)
        points = list(model.points3D.values())
        step = max(1, len(points)//100000)
        cameras = []
        for image in model.images.values():
            if image.has_pose:
                cameras.append(dict(id=int(image.image_id), name=image.name,
                               center=image.projection_center().tolist(), rotation=image.cam_from_world().rotation.matrix().tolist()))
        qc = json.loads((root/'qc.json').read_text()) if (root/'qc.json').exists() else None
        surface=None
        if (root/'surface/manifest.json').is_file():
            manifest=json.loads((root/'surface/manifest.json').read_text())
            from urllib.parse import quote
            assets={}
            for kind,name in manifest['files'].items():
                path=root/'surface'/name
                if path.is_file():
                    assets[kind]='/api/file?path='+quote(str(path.relative_to(project().root)))
            surface=dict(manifest=manifest,assets=assets)
        return dict(run=state, coordinate_system='DICOM_LPS_mm' if model_path.name=='ct_ba' else 'SfM_arbitrary',
                    xyz=[p.xyz.tolist() for p in points[::step]], rgb=[p.color.tolist() for p in points[::step]],
                    cameras=cameras, qc=qc, points=state['controls'],surface=surface)

    @app.get('/api/results/{run_id}/download')
    def download(run_id: str):
        state = run_state(run_id)
        if state['status'] != 'completed' or 'output' not in state:
            raise ValueError('A completed output is required')
        p = project()
        archive = p.root/'reconstruction'/run_id/'export.zip'
        if not archive.exists():
            shutil.make_archive(str(archive.with_suffix('')), 'zip', p.file(state['output']))
        return FileResponse(archive, filename='ViewSyncBuilder-'+run_id[:8]+'.zip')

    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    app.mount('/vendor', StaticFiles(directory=base/'vendor' if getattr(sys, 'frozen', False) else engine.ROOT/'vendor'), name='vendor')
    app.mount('/', StaticFiles(directory=base/'frontend', html=True), name='ui')
    return app
