"""Measure fresh image/mask/ALIKED/save work on 12 existing RC source photos."""

from macar_paths import DATA_DIR
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from aliked_lightglue_pnp import _load_models, _image_tensor, _extract
from build_aliked_object_map import load_splat_points, load_scene_transform, object_mask, subset_features
from offline_pnp_test import load_reference_images, load_camera_parameters, camera_matrix_and_distortion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    metadata = json.loads((DATA_DIR/'output_spatial_scoring_20261004/inputs.json').read_text())
    summary = json.loads((Path(metadata['map_path'])/'summary.json').read_text())
    records = load_reference_images(Path(summary['input_files']['images']))
    cameras, params = load_camera_parameters(Path(summary['input_files']['cameras']))
    splat = Path(summary['input_files']['splat'])
    transform = splat.parent/'scene_transform.json'
    mask_points, _, _ = load_splat_points(splat, .005, 80000,
        load_scene_transform(transform if transform.is_file() else None))
    cv2.setNumThreads(4)
    start = time.perf_counter()
    torch, extractor, _, device = _load_models(2048, .01, 'mps', matcher_required=False)
    torch.set_num_threads(4)
    torch.mps.synchronize()
    model_load_ms = (time.perf_counter()-start)*1000.
    rows = []
    for query in metadata['queries']:
        if query['kind'] != 'source':
            continue
        path = Path(query['path']); record = records[path.name]
        matrix, distortion = camera_matrix_and_distortion(np, cameras[record.camera_id], params[record.camera_id])
        started = time.perf_counter()
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError('Missing image: '+str(path))
        read_ms = (time.perf_counter()-started)*1000.
        started = time.perf_counter()
        mask, box = object_mask(image.shape, mask_points, record, matrix, distortion, 8, .035, .03)
        if mask is None:
            raise RuntimeError('Missing object mask')
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        mask_ms = (time.perf_counter()-started)*1000.
        torch.mps.synchronize(); started = time.perf_counter()
        with torch.inference_mode():
            raw = _extract(extractor, _image_tensor(torch, crop, device), 1024)
        features, uv = subset_features(raw, mask, (x0, y0))
        arrays = {key: value.detach().cpu().numpy() for key, value in features.items()}
        torch.mps.synchronize()
        extraction_ms = (time.perf_counter()-started)*1000.
        started = time.perf_counter()
        np.savez_compressed(args.output/(path.stem+'.npz'), **arrays, full_uv=uv, object_mask=mask, crop_box=box)
        save_ms = (time.perf_counter()-started)*1000.
        rows.append(dict(photo=path.name, image_shape=image.shape, crop_shape=crop.shape, features=len(uv),
                         read_ms=read_ms, mask_ms=mask_ms, extraction_ms=extraction_ms, save_ms=save_ms))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(model_load_ms=model_load_ms, device='mps', resize=1024, max_keypoints=2048, rows=rows,
                  warm_means_ms={key:float(np.mean([row[key] for row in rows[1:]])) for key in
                                 ['read_ms', 'mask_ms', 'extraction_ms', 'save_ms']},
                  caveat='12 photos from existing reconstruction; 620-photo extrapolation, not full fresh run')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['warm_means_ms']), flush=True)


if __name__ == '__main__':
    main()
