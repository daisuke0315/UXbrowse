"""Read-only map inspection and durable, subprocess-based offline jobs.

No camera access, no guessed XYZ, no writes to the source map. Completion
means geometric audit passed, not CT accuracy or universal view coverage.
"""
from __future__ import annotations

from macar_paths import DATA_DIR
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from offline_pnp_test import (load_reference_images, load_camera_parameters,
                              camera_matrix_and_distortion, quaternion_to_rotation)

ROOT = Path(__file__).resolve().parent
JOBS = DATA_DIR / 'output_map_preprocessing'
SELECTION = DATA_DIR / 'registered_map_selection.json'


def audit_is_current(path):
    path = Path(path)
    audit_time = (path/'audit.json').stat().st_mtime_ns
    files = [path/'summary.json', path/'object_aliked_map.npz',
             *list((path/'reference_features').glob('*.npz'))]
    return all(f.stat().st_mtime_ns <= audit_time for f in files)


def audited_map(path):
    """Read only: accept immutable, fully audited saved maps, not active work."""
    path = Path(path).expanduser().resolve()
    report = json.loads((path / 'audit.json').read_text())
    if not report.get('geometry_ok') or not audit_is_current(path):
        raise ValueError('地図の幾何監査が未完了、失敗、または保存後に変更されています。')
    if not (path/'object_aliked_map.npz').is_file() or not (path/'reference_features').is_dir():
        raise ValueError('地図の関連ファイルが不足しています。')
    return path, report


def recommended_map(fallback, directory=None):
    """Best coverage among audited snapshots of the SAME registration dataset.

    Coverage is an operational recommendation, not proof of CT accuracy.
    Saved stages are usable even if later stages are paused/interrupted.
    """
    fallback = Path(fallback).expanduser().resolve()
    directory = Path(directory) if directory is not None else JOBS
    try:
        source = json.loads((fallback/'summary.json').read_text())['input_files']
    except (OSError, ValueError, KeyError, TypeError):
        return fallback
    best, best_score = fallback, (-1., -1)
    paths = {fallback}
    for state_file in directory.glob('*/state.json'):
        try:
            state = json.loads(state_file.read_text())
            paths.update(Path(p['output']).resolve() for p in state.get('passes', []) if p.get('output'))
            if state.get('output'):
                paths.add(Path(state['output']).resolve())
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for candidate in sorted(paths):
        try:
            path, report = audited_map(candidate)
            candidate_source = json.loads((path/'summary.json').read_text())['input_files']
            if candidate_source != source:
                continue
            score = (float(report['xyz_ratio']), int(report['xyz_observations']))
            if score > best_score:
                best, best_score = path, score
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return best


def selected_map(fallback):
    try:
        value = json.loads(SELECTION.read_text())
        path = Path(value['map_path']).resolve()
        audit = json.loads((path / 'audit.json').read_text())
        if (path / 'summary.json').is_file() and audit.get('geometry_ok') and audit_is_current(path):
            return path
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return recommended_map(fallback)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + secrets.token_hex(4))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def inspect_map(path, *, audit=False, progress=None):
    path = Path(path).expanduser().resolve()
    summary = json.loads((path / 'summary.json').read_text())
    records = load_reference_images(Path(summary['input_files']['images']))
    with np.load(path / 'object_aliked_map.npz', allow_pickle=False) as a:
        xyz = np.asarray(a['xyz'], np.float64)
        lengths = a['track_length'].copy()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid map XYZ')
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    observations = np.zeros(len(xyz), np.int64)
    issues, rows, all_errors = [], [], []
    cameras, params = load_camera_parameters(Path(summary['input_files']['cameras'])) if audit else ({}, {})
    for number, (name, rec) in enumerate(sorted(records.items()), 1):
        file = path / 'reference_features' / (Path(name).stem + '.npz')
        if not file.is_file():
            continue
        with np.load(file, allow_pickle=False) as a:
            ids, values, uv = a['point_ids'], a['xyz'], a['full_uv']
            features = len(ids)
            if ids.shape != (features,) or uv.shape != (features, 2) or values.shape != (features, 3):
                raise ValueError(f'Invalid feature shapes: {name}')
            valid = ids >= 0
            if (np.any(ids < -1) or np.any(ids >= len(xyz)) or not np.isfinite(uv).all()
                or not np.isfinite(values[valid]).all()):
                raise ValueError(f'Invalid IDs/XYZ/UV: {name}')
            if not np.array_equal(values[valid].astype(np.float64), xyz[ids[valid]]):
                issues.append(name + ': xyz mismatch')
            if len(np.unique(ids[valid])) != int(valid.sum()):
                issues.append(name + ': repeated landmark in image')
            if audit:
                if (a['descriptors'].shape[1] != features or a['keypoints'].shape[1] != features
                    or not np.isfinite(a['descriptors']).all() or not np.isfinite(a['keypoints']).all()):
                    issues.append(name + ': invalid descriptors')
            crop = a['crop_box'].tolist()
        np.add.at(observations, ids[valid], 1)
        rotation = quaternion_to_rotation(np, rec.pose)
        translation = np.asarray(rec.pose[4:], np.float64)
        position = -rotation.T @ translation
        direction = position - center
        ratio = float(valid.mean()) if features else 0.
        row = dict(image=name, features=features, xyz=int(valid.sum()), ratio=ratio,
                   missing_xyz=features-int(valid.sum()), crop_box=crop,
                   camera_position=position.tolist(),
                   azimuth_deg=float(np.degrees(np.arctan2(direction[1], direction[0]))),
                   elevation_deg=float(np.degrees(np.arctan2(-direction[2], np.linalg.norm(direction[:2])))))
        if audit and valid.any():
            k, distortion = camera_matrix_and_distortion(np, cameras[rec.camera_id], params[rec.camera_id])
            points = xyz[ids[valid]]
            depths = np.einsum('ni,ji->nj', points, rotation)[:, 2] + translation[2]
            projected = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0], translation, k, distortion)[0].reshape(-1, 2)
            errors = np.linalg.norm(projected - uv[valid], axis=1)
            if np.any(depths <= 0) or np.any(errors > 3.01) or not np.isfinite(errors).all():
                issues.append(name + ': depth/reprojection QC')
            all_errors.extend(errors.tolist())
            row.update(reprojection_mean_px=float(errors.mean()), reprojection_max_px=float(errors.max()))
        rows.append(row)
        if progress and number % 25 == 0:
            progress(dict(stage='audit' if audit else 'inspect', completed=number, total=len(records)))
    if not np.array_equal(observations, lengths) or np.any(observations < 3):
        issues.append('Track lengths disagree or fewer than 3 views')
    errors = np.asarray(all_errors)
    features = sum(r['features'] for r in rows)
    report = dict(map_path=str(path), source_files=summary['input_files'], images=len(rows),
        features=features, xyz_landmarks=len(xyz), xyz_observations=int(observations.sum()),
        xyz_ratio=float(observations.sum()/max(features, 1)),
        images_under_10_percent=sum(r['ratio'] < .1 for r in rows),
        images_without_xyz=sum(r['xyz'] == 0 for r in rows),
        minimum_track_length=int(observations.min()), rows=rows, center=center.tolist(),
        issues=issues, geometry_checked=audit, geometry_ok=audit and not issues,
        notes='XYZ coverage is not CT accuracy. Missing XYZ is retained, never filled with guessed depth.')
    if len(errors):
        report['reprojection_px'] = dict(median=float(np.median(errors)), p95=float(np.percentile(errors, 95)), maximum=float(errors.max()))
    return report


def reference_details(path, name):
    path = Path(path).expanduser().resolve()
    summary = json.loads((path / 'summary.json').read_text())
    records = load_reference_images(Path(summary['input_files']['images']))
    if name not in records or Path(name).name != name:
        raise ValueError('Unknown reference image')
    cameras, params = load_camera_parameters(Path(summary['input_files']['cameras']))
    rec = records[name]
    camera = cameras[rec.camera_id]
    with np.load(path / 'reference_features' / (Path(name).stem + '.npz'), allow_pickle=False) as a:
        uv, ids = a['full_uv'], a['point_ids']
        valid = ids >= 0
        errors = np.full(len(ids), np.nan)
        if valid.any():
            k, distortion = camera_matrix_and_distortion(np, camera, params[rec.camera_id])
            rot = quaternion_to_rotation(np, rec.pose)
            projected = cv2.projectPoints(a['xyz'][valid].astype(np.float64), cv2.Rodrigues(rot)[0],
                np.asarray(rec.pose[4:], np.float64), k, distortion)[0].reshape(-1, 2)
            errors[valid] = np.linalg.norm(projected - uv[valid], axis=1)
        contours, _ = cv2.findContours(a['object_mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        offset = a['crop_box'][:2]
        outlines = [(c[::max(1, len(c)//400), 0]+offset).tolist() for c in contours if len(c)>2]
        return dict(image=name, width=camera.width, height=camera.height, crop_box=a['crop_box'].tolist(),
            mask_outlines=outlines, points=[dict(index=i, uv=p.tolist(), point_id=int(pid),
                xyz=a['xyz'][i].tolist() if pid >= 0 else None,
                reprojection_px=float(errors[i]) if pid >= 0 else None) for i, (p, pid) in enumerate(zip(uv, ids))])


def reference_preview(path, name):
    path = Path(path).resolve()
    summary = json.loads((path / 'summary.json').read_text())
    records = load_reference_images(Path(summary['input_files']['images']))
    if name not in records or Path(name).name != name:
        raise ValueError('Unknown reference image')
    # The dataset path is supplied by the map, never by an HTTP image path.
    image_root = Path(summary['input_files']['images']).parent.parent.parent / 'images'
    if summary.get('image_dir'):
        image_root = Path(summary['image_dir'])
    image = cv2.imread(str(image_root / name))
    if image is None:
        raise ValueError(f'Registered image unreadable: {image_root / name}')
    h, w = image.shape[:2]
    scale = min(1., 1600/max(w, h))
    image = cv2.resize(image, (round(w*scale), round(h*scale)))
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError('Preview encoding failed')
    return encoded.tobytes()


class PreprocessingManager:
    def __init__(self, directory=JOBS):
        self.directory = Path(directory).resolve()
        self.lock = threading.RLock()
        self.inspections = {}
        self.source_overviews = {}
        self.children = []

    def jobs(self):
        # Reap owned workers; a crashed child must not look permanently busy.
        exits = {p.pid:p.poll() for p in self.children}
        result = []
        for file in sorted(self.directory.glob('*/state.json'), reverse=True):
            try:
                value = json.loads(file.read_text())
                # Older workers saved a baseline report but did not include
                # its counters in state.json. Expose it read-only for the UI.
                if 'source_overview' not in value:
                    baseline = file.parent/'before.json'
                    try:
                        if baseline.is_file():
                            key = (str(baseline), baseline.stat().st_mtime_ns)
                            if key not in self.source_overviews:
                                before = json.loads(baseline.read_text())
                                self.source_overviews[key] = {k:before[k] for k in
                                    ('images','features','xyz_observations','xyz_ratio','xyz_landmarks')}
                            value['source_overview'] = self.source_overviews[key]
                    except (OSError, ValueError, KeyError, TypeError):
                        # An optional legacy report must not hide a valid job.
                        pass
                if value['status'] in ('running', 'paused'):
                    pid = int(value['pid'])
                    try:
                        if exits.get(pid) is not None:
                            raise ProcessLookupError()
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        value = dict(value, status='interrupted', detail='Process stopped; pair cache can be reused')
                result.append(value)
            except (OSError, ValueError, KeyError):
                continue
        return result

    def busy(self):
        return any(j['status'] == 'running' for j in self.jobs())

    def status(self):
        return dict(jobs=self.jobs(), default_map=str(selected_map(DATA_DIR / 'output_aliked_object_map_enriched_pilot')))

    def inspect(self, path, audit=False):
        path = Path(path).expanduser().resolve()
        # Completed maps are immutable; include manifest and archive mtimes.
        feature_stamp = tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size)
                              for f in sorted((path/'reference_features').glob('*.npz')))
        stamp = ((path/'summary.json').stat().st_mtime_ns, (path/'object_aliked_map.npz').stat().st_mtime_ns,
                 feature_stamp)
        key = (str(path), stamp, audit)
        with self.lock:
            if audit or key not in self.inspections:
                audit_file = path/'audit.json'
                self.inspections[key] = (inspect_map(path, audit=True) if audit else
                    json.loads(audit_file.read_text()) if audit_file.is_file() and audit_is_current(path)
                    else inspect_map(path))
            return self.inspections[key]

    def start(self, values):
        with self.lock:
            if any(j['status'] in ('running', 'paused') for j in self.jobs()):
                raise ValueError('既存の前処理を再開または中止してから、新しい前処理を開始してください。')
            source = Path(values['source']).expanduser().resolve()
            if not (source/'summary.json').is_file() or not (source/'object_aliked_map.npz').is_file():
                raise ValueError('Choose a completed registered ALIKED map')
            scope = values.get('scope', 'all')
            if scope not in ('all', 'weak', 'selected'):
                raise ValueError('Invalid processing scope')
            neighbors = int(values.get('max_neighbors', 48))
            if not 12 <= neighbors <= 96:
                raise ValueError('Neighbors must be 12–96')
            goal = float(values.get('coverage_goal', .5))
            if not np.isfinite(goal) or not .1 <= goal <= .95:
                raise ValueError('Coverage goal must be 0.1–0.95')
            names = values.get('target_images', [])
            if not isinstance(names, list) or len(names) > 620:
                raise ValueError('Invalid image selection')
            if scope == 'selected' and not names:
                raise ValueError('Select at least one reference')
            job_id = time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(3)
            job_dir = self.directory / job_id
            job_dir.mkdir(parents=True)
            config = dict(source=str(source), scope=scope, max_neighbors=neighbors,
                coverage_goal=goal, target_images=names, device='cpu',
                matching_method='known-pose')
            atomic_json(job_dir/'config.json', config)
            # Early state prevents another request from racing the child startup.
            atomic_json(job_dir/'state.json', dict(job_id=job_id, status='running', pid=os.getpid(),
                stage='launch', source=str(source), detail='Launching offline preprocessing'))
            with (job_dir/'process.log').open('ab') as log:
                try:
                    child = subprocess.Popen([sys.executable, str(ROOT/'preprocess_registered_map.py'),
                        '--job-dir', str(job_dir)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                except Exception:
                    atomic_json(job_dir/'state.json', dict(job_id=job_id, status='error', detail='Worker could not start'))
                    raise
            self.children.append(child)
            return dict(job_id=job_id)

    def cancel(self, job_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,80}', job_id):
            raise ValueError('Invalid job ID')
        path = self.directory / job_id
        if not (path/'state.json').is_file():
            raise ValueError('Unknown job')
        state = json.loads((path/'state.json').read_text())
        pid = self._worker_pid(job_id, state) if state['status'] == 'paused' else None
        (path/'cancel.request').touch()
        if pid is not None:
            state.update(status='running', detail='中止要求を処理中')
            atomic_json(path/'state.json', state)
            os.kill(pid, signal.SIGCONT)

    def _worker_pid(self, job_id, state):
        """Only signal the recorded worker whose command names this exact job.

        This also permits suspending an already running worker created before
        pause support was added, without discarding its in-memory progress.
        """
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,80}', job_id):
            raise ValueError('Invalid job ID')
        if not hasattr(signal, 'SIGSTOP'):
            raise ValueError('この環境では前処理の一時停止に対応していません。')
        try:
            pid = int(state['pid'])
            command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True)
        except (KeyError, ValueError, subprocess.SubprocessError, OSError):
            raise ValueError('前処理プロセスが見つかりません。状態を更新してください。') from None
        if pid <= 1 or 'preprocess_registered_map.py' not in command or str(self.directory/job_id) not in command:
            raise ValueError('対象の前処理プロセスを確認できないため操作しません。')
        return pid

    def pause(self, job_id):
        with self.lock:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,80}', job_id):
                raise ValueError('Invalid job ID')
            path = self.directory/job_id/'state.json'
            if not path.is_file():
                raise ValueError('Unknown job')
            state = json.loads(path.read_text())
            if state['status'] != 'running':
                raise ValueError('実行中の前処理だけ一時停止できます。')
            pid = self._worker_pid(job_id, state)
            try:
                os.kill(pid, signal.SIGSTOP)
                # Wait for the OS acknowledgement before permitting live work.
                for _ in range(50):
                    stopped = subprocess.check_output(['ps','-p',str(pid),'-o','stat='], text=True).strip()
                    if 'T' in stopped:
                        break
                    time.sleep(.02)
                else:
                    raise ValueError('一時停止を確認できませんでした。')
                state = json.loads(path.read_text())
                state.update(status='paused', paused_at=time.time(), detail='一時停止中：作業位置を保持。保存済み地図で検証できます。')
                atomic_json(path, state)
            except Exception as exc:
                try:
                    os.kill(pid, signal.SIGCONT)
                except OSError:
                    pass
                raise ValueError('一時停止できませんでした。状態を更新してください。') from exc
            return dict(accepted=True, status='paused')

    def resume(self, job_id):
        with self.lock:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,80}', job_id):
                raise ValueError('Invalid job ID')
            if self.busy():
                raise ValueError('別の前処理が実行中です。')
            path = self.directory/job_id/'state.json'
            if not path.is_file():
                raise ValueError('Unknown job')
            state = json.loads(path.read_text())
            if state['status'] != 'paused':
                raise ValueError('一時停止中の前処理だけ再開できます。')
            pid = self._worker_pid(job_id, state)
            state.update(status='running', detail='前処理を再開中')
            atomic_json(path, state)
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError as exc:
                state.update(status='interrupted', detail='再開できませんでした。キャッシュは保持されています。')
                atomic_json(path, state)
                raise ValueError(state['detail']) from exc
            return dict(accepted=True, status='running')

    def activate(self, job_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,80}', job_id):
            raise ValueError('Invalid job ID')
        state = json.loads((self.directory/job_id/'state.json').read_text())
        if state['status'] != 'complete':
            raise ValueError('Only audited, completed jobs can be selected')
        path = Path(state['output'])
        audit = json.loads((path/'audit.json').read_text())
        if not audit['geometry_ok'] or not audit_is_current(path):
            raise ValueError('Map failed geometric audit')
        atomic_json(SELECTION, dict(map_path=str(path), job_id=job_id))
        return dict(map_path=str(path))
