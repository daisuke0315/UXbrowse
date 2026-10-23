"""Project storage with immutable originals and separate run directories."""
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import __version__


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_project(root):
    path = Path(root)/'.builder-job.lock'
    if path.exists():
        try:
            owner = json.loads(path.read_text())['pid']
            os.kill(int(owner), 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
        except (ValueError, KeyError, PermissionError):
            raise ValueError('Project job lock cannot be verified; another job may be running')
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError('A reconstruction job is already running for this project')
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'pid': os.getpid()}, stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


class Project:
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root).resolve(), manifest

    @classmethod
    def create(cls, root, name):
        root = Path(root).resolve()
        if not str(name).strip():
            raise ValueError('Project name is required')
        root.mkdir(parents=True, exist_ok=False)
        for folder in ('input/images', 'input/originals', 'input/ct', 'control', 'cache/aliked',
                       'cache/lightglue', 'reconstruction', 'output', 'logs'):
            (root / folder).mkdir(parents=True, exist_ok=True)
        project = cls(root, dict(schema_version=1, software_version=__version__, id=uuid.uuid4().hex,
                                name=str(name).strip(), created=now(), images=[], ct=None, runs=[]))
        project.save()
        project.set_controls([])
        return project

    @classmethod
    def open(cls, root):
        manifest = json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported project schema')
        project = cls(root, manifest)
        for image in manifest['images']:
            for key in ('path', 'original'):
                project.file(image[key])
        return project

    def file(self, relative):
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError('Path escapes project directory')
        return result

    def save(self):
        self.manifest['updated'] = now()
        write_json(self.root / 'manifest.json', self.manifest)

    def import_image(self, source):
        source = Path(source)
        digest = sha256(source)
        for record in self.manifest['images']:
            if record['sha256'] == digest:
                return record
        with Image.open(source) as image:
            if image.format not in ('JPEG', 'PNG'):
                raise ValueError('Only JPEG and PNG images are supported')
            orientation = int(image.getexif().get(274, 1))
            camera = {str(k): str(image.getexif()[k]) for k in (271, 272, 37386, 41989) if k in image.getexif()}
            normalized = ImageOps.exif_transpose(image).convert('RGB')
        image_id = uuid.uuid4().hex
        original = 'input/originals/' + image_id + source.suffix.lower()
        path = f'input/images/{len(self.manifest["images"]):06d}_{image_id}.png'
        shutil.copyfile(source, self.file(original))
        normalized.save(self.file(path))
        record = dict(id=image_id, name=source.name, path=path, original=original, sha256=digest,
                      normalized_sha256=sha256(self.file(path)), enabled=True,
                      width=normalized.width, height=normalized.height, exif_orientation=orientation,
                      camera=camera, mask=None)
        self.manifest['images'].append(record)
        self.save()
        return record

    def controls(self):
        return json.loads((self.root / 'control/points.json').read_text(encoding='utf-8'))

    def set_controls(self, points):
        images = {im['id']: im for im in self.manifest['images']}
        ids = set()
        for point in points:
            if not isinstance(point.get('id'), str) or not point['id'].strip() or point['id'] in ids:
                raise ValueError('Control IDs must be unique nonempty strings')
            ids.add(point['id'])
            if point.get('role') not in ('control', 'evaluation'):
                raise ValueError('Point role must be control or evaluation')
            xyz = np.asarray(point.get('xyz'), float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError('CT coordinates must be finite LPS XYZ in mm')
            seen = set()
            for observation in point.get('observations', []):
                image_id = observation['image_id']
                uv = np.asarray(observation['uv'], float)
                if image_id not in images or image_id in seen:
                    raise ValueError('Observation image is missing or repeated')
                if uv.shape != (2,) or not np.isfinite(uv).all():
                    raise ValueError('Observation must be finite UV')
                image = images[image_id]
                if not (0 <= uv[0] < image['width'] and 0 <= uv[1] < image['height']):
                    raise ValueError('UV lies outside the EXIF-normalized image')
                seen.add(image_id)
        write_json(self.root / 'control/points.json', points)
        write_json(self.root / 'control/gcp.json', [p for p in points if p['role'] == 'control'])
        write_json(self.root / 'control/evaluation_points.json', [p for p in points if p['role'] == 'evaluation'])
