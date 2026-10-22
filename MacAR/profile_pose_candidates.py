"""Compare candidate selection on identical query features and unchanged QC.

Diagnostic overrides stay in this standalone process. No application defaults
or map landmarks are changed. A PASS is not physical accuracy ground truth.
"""

from macar_paths import DATA_DIR
import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np

from browser_photo import decode_photo
from registered_aliked_live import RegisteredALIKEDLiveLocalizer, rotation_angle
from realtime_pnp import DEFAULT_REGISTERED_MAP, DEFAULT_SPLAT_VIEWS, approximate_camera_matrix


def main():
    output = DATA_DIR/'output_pose_audit_20261001/candidate_experiment.jsonl'
    if output.exists():
        raise RuntimeError('Use a fresh output')
    photo = json.loads((DATA_DIR/'output_localization_profile_20261001/macar-profile-2594-pipeline-native.jsonl').read_text().splitlines()[0])['photo']
    image = decode_photo(Path(photo).read_bytes())
    matrix = approximate_camera_matrix(image.shape[1], image.shape[0], 72.)
    engine = RegisteredALIKEDLiveLocalizer(DEFAULT_REGISTERED_MAP, DEFAULT_SPLAT_VIEWS,
        device='mps', max_keypoints=2000, top_k=3, resize=1024,
        fundamental_threshold=3., ransac_threshold=4., max_reprojection_error=3., retrieval_mode='dino-patch')
    engine.query_resize = 1600
    workspace = {}
    def emit(row):
        with output.open('a', encoding='utf-8') as target:
            target.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(dict(event=row['event'], case=row.get('case'), result=row.get('result'),
                             references=row.get('references')), ensure_ascii=False), flush=True)
    def experiment(name, space):
        cv2.setRNGSeed(0)
        started = time.perf_counter()
        result, _ = engine._search_attempts(image, matrix, np.zeros(5), None, space)
        debug = engine.last_debug
        emit(dict(event='result', case=name, elapsed_ms=(time.perf_counter()-started)*1000,
            result=result.as_dict(), diagnostics=copy.deepcopy(engine.last_diagnostics),
            references=space['references'], points=debug.matched_xyz.tolist(),
            observed=debug.image_points.tolist() if debug.image_points is not None else None,
            inliers=list(debug.inlier_indices)))
        return result
    experiment('original', workspace)
    ranked = workspace['retrieval'][0]
    original_references = list(workspace['references'])
    top_index = ranked[0][2]
    first_references = [r for r in original_references if r[1] == top_index]
    experiment('top1_original_references', dict(workspace, references=first_references))
    view = engine.views[top_index]
    transform = np.asarray(view['world_to_camera'], np.float64)
    center = np.asarray(engine.manifest['center'], np.float64)
    view_direction = -transform[:3, :3].T @ transform[:3, 3]-center
    view_direction /= np.linalg.norm(view_direction)
    candidates = []
    for name, reference in engine.registered.items():
        angle = rotation_angle(reference['rotation'], transform[:3, :3])
        if angle > 30:
            continue
        with np.load(reference['path'], allow_pickle=False) as archive:
            count = int(((archive['point_ids'] >= 0) & np.isfinite(archive['xyz']).all(axis=1)).sum())
        if count < 400:
            continue
        direction = reference['center']-center
        direction /= np.linalg.norm(direction)
        view_angle = np.degrees(np.arccos(np.clip(np.dot(direction, view_direction), -1., 1.)))
        candidates.append((name, top_index, float(angle+.35*view_angle), angle, count))
    candidates.sort(key=lambda r:r[2])
    emit(dict(event='candidate_pool', references=candidates,
        note='Diagnostic subset: same top-1 direction, angle <=30deg, >=400 valid XYZ; not an adopted threshold'))
    enriched = [r[:4] for r in candidates[:2]]
    experiment('top1_supported_references', dict(workspace, references=enriched))
    experiment('original_plus_supported_references', dict(workspace, references=original_references+enriched))


if __name__ == '__main__':
    main()
