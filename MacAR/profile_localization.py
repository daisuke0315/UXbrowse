"""Repeat the existing photo pipeline and profile ALIKED without changing outputs.

Native mode synchronizes only around the whole extraction. Stages mode also
synchronizes stage boundaries to attribute asynchronous GPU work correctly;
its total is diagnostic and may differ from native mode. No package files,
model weights, matching thresholds, or application defaults are modified.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

import cv2

from aliked_lightglue_pnp import _extract, _image_tensor, _load_models
from browser_photo import decode_photo, localize_photo
from glb_virtual_live import crop_live_object
from realtime_pnp import DEFAULT_REGISTERED_MAP, DEFAULT_SPLAT_VIEWS, approximate_camera_matrix


def synchronize(torch, device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def memory_snapshot(torch, device):
    data = dict(max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    import os
    if __import__('sys').platform == 'darwin':
        from profile_host_memory import process_memory
        data['process'] = process_memory(os.getpid())
    if device.type == 'mps':
        for name in ('current_allocated_memory', 'driver_allocated_memory'):
            function = getattr(torch.mps, name, None)
            if function is not None:
                data[name] = function()
    return data


class ExtractionProbe:
    """Wrap the original extract call; hooks observe, never replace its values."""

    def __init__(self, torch, extractor, device, stage_sync=False, detailed=False):
        self.torch, self.extractor, self.device = torch, extractor, device
        self.stage_sync, self.detailed = stage_sync, detailed

    def boundary(self):
        if self.stage_sync:
            synchronize(self.torch, self.device)
        return time.perf_counter()

    @contextmanager
    def observe(self, row):
        original = self.extractor.extract_dense_map
        handles, started = [], {}

        def before(name):
            def hook(module, inputs):
                started[name] = self.boundary()
            return hook

        def after(name):
            def hook(module, inputs, output):
                row['stages_ms'][name] = (self.boundary()-started[name])*1000
            return hook

        def dense(image):
            begin = self.boundary()
            row['preprocessed_size'] = list(image.shape[-2:][::-1])
            row['stages_ms']['preprocess'] = (begin-row['_started'])*1000
            output = original(image)
            row['stages_ms']['dense_map'] = (self.boundary()-begin)*1000
            row['dense_map_shape'] = list(output[0].shape)
            return output

        try:
            self.extractor.extract_dense_map = dense
            names = ['dkd', 'desc_head']
            if self.detailed:
                names += ['block1', 'block2', 'block3', 'block4',
                          'conv1', 'conv2', 'conv3', 'conv4', 'score_head']
            for name in names:
                module = getattr(self.extractor, name)
                handles.append(module.register_forward_pre_hook(before(name)))
                handles.append(module.register_forward_hook(after(name)))
            yield
        finally:
            self.extractor.extract_dense_map = original
            for handle in handles:
                handle.remove()

    def run(self, tensor, resize):
        row = dict(input_size=list(tensor.shape[-2:][::-1]), resize=resize,
                   timing_mode='stage_sync' if self.stage_sync else 'native',
                   stages_ms={}, memory_before=memory_snapshot(self.torch, self.device))
        synchronize(self.torch, self.device)
        row['_started'] = time.perf_counter()
        with self.observe(row):
            features = _extract(self.extractor, tensor, resize)
        synchronize(self.torch, self.device)
        row['total_ms'] = (time.perf_counter()-row.pop('_started'))*1000
        row['keypoints'] = int(features['keypoints'].shape[1])
        row['memory_after'] = memory_snapshot(self.torch, self.device)
        # Native substage times are dispatch times, not GPU execution times.
        row['stage_times_include_gpu_completion'] = self.stage_sync
        return features, row


def write_row(path, row):
    with path.open('a', encoding='utf-8') as target:
        target.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1600])
    parser.add_argument('--mode', choices=('features', 'pipeline'), default='features')
    parser.add_argument('--roi', type=float, nargs=4, help='Normalized X Y WIDTH HEIGHT from runtime log')
    parser.add_argument('--stage-sync', action='store_true')
    parser.add_argument('--detailed', action='store_true', help='Also time encoder modules; requires stage-sync')
    parser.add_argument('--map', type=Path, default=DEFAULT_REGISTERED_MAP)
    parser.add_argument('--views', type=Path, default=DEFAULT_SPLAT_VIEWS)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--recreate-models', action='store_true',
                        help='Rebuild the standalone pipeline before runs after the first')
    parser.add_argument('--idle-seconds', type=float, default=0., help='Idle before each timed pipeline run')
    parser.add_argument('--cache-policy', choices=('retain', 'trim', 'alternate'), default='retain')
    args = parser.parse_args()
    if args.output.exists() or args.runs < 1 or any(n < 32 for n in args.sizes):
        parser.error('Use a new output path, positive runs, and sizes >= 32')
    if args.detailed and not args.stage_sync:
        parser.error('--detailed requires --stage-sync')
    if args.mode == 'features' and args.roi is None:
        parser.error('features mode requires --roi (use 0 0 1 1 for a pre-cropped image)')
    if args.recreate_models and args.mode != 'pipeline':
        parser.error('--recreate-models requires --mode pipeline')
    if args.idle_seconds < 0 or (args.cache_policy != 'retain' and args.mode != 'pipeline'):
        parser.error('Use nonnegative idle time; cache policy requires pipeline mode')
    data = args.photo.read_bytes()
    image = decode_photo(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'pipeline':
        import registered_aliked_live as registered
        import numpy as np
        def new_engine():
            return registered.RegisteredALIKEDLiveLocalizer(args.map, args.views,
                device=args.device, max_keypoints=2000, top_k=args.top_k,
                resize=1024, fundamental_threshold=3., ransac_threshold=4.,
                max_reprojection_error=3., retrieval_mode='dino-patch')
        engine = new_engine()
        torch, extractor, device = engine.torch, engine.extractor, engine.device
    else:
        torch, extractor, _, device = _load_models(2000, .01, args.device)
    probe = ExtractionProbe(torch, extractor, device, args.stage_sync, args.detailed)
    write_row(args.output, dict(event='configuration', photo=str(args.photo.resolve()),
        sha256=hashlib.sha256(data).hexdigest(), image_size=list(image.shape[:2][::-1]),
        device=str(device), torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
        runs=args.runs, sizes=args.sizes, mode=args.mode, roi=args.roi,
        stage_sync=args.stage_sync, detailed=args.detailed,
        trace_active=__import__('sys').gettrace() is not None,
        recreate_models=args.recreate_models,
        idle_seconds=args.idle_seconds, cache_policy=args.cache_policy,
        note='Timings only; no position accuracy claim. Each size keeps 2000-point budget.'))
    totals = {}
    if args.mode == 'features':
        cropped = crop_live_object(image, args.roi)
        if cropped is None:
            raise ValueError('Empty ROI')
        crop, bounds = cropped
        write_row(args.output, dict(event='crop', bounds=list(bounds), crop_size=list(crop.shape[:2][::-1])))
        for size in args.sizes:
            totals[size] = []
            for run in range(args.runs):
                synchronize(torch, device)
                before = time.perf_counter()
                tensor = _image_tensor(torch, crop, device)
                synchronize(torch, device)
                conversion_ms = (time.perf_counter()-before)*1000
                with torch.inference_mode():
                    _, row = probe.run(tensor, size)
                row.update(event='extraction', run=run, first_run=run==0, conversion_ms=conversion_ms)
                totals[size].append(row['total_ms'])
                write_row(args.output, row)
                del tensor
    else:
        original_extract = registered._extract
        def profiled_extract(model, tensor, resize):
            features, row = probe.run(tensor, resize)
            write_row(args.output, dict(row, event='extraction', run=current_run))
            return features
        registered._extract = profiled_extract
        try:
            matrix = approximate_camera_matrix(image.shape[1], image.shape[0], 72.)
            for size in args.sizes:
                engine.query_resize = size
                totals[size] = []
                for current_run in range(args.runs):
                    if args.recreate_models and current_run:
                        import gc
                        synchronize(torch, device)
                        del engine, probe, extractor
                        gc.collect()
                        build_started = time.perf_counter()
                        engine = new_engine()
                        torch, extractor, device = engine.torch, engine.extractor, engine.device
                        probe = ExtractionProbe(torch, extractor, device, args.stage_sync, args.detailed)
                        engine.query_resize = size
                        synchronize(torch, device)
                        write_row(args.output, dict(event='model_recreated', run=current_run,
                            build_ms=(time.perf_counter()-build_started)*1000,
                            memory=memory_snapshot(torch, device)))
                    synchronize(torch, device)
                    cache_before = memory_snapshot(torch, device)
                    trim_cache = args.cache_policy == 'trim' or (args.cache_policy == 'alternate' and current_run % 2 == 0)
                    if trim_cache:
                        if device.type != 'mps':
                            raise ValueError('Cache trimming experiment requires MPS')
                        torch.mps.empty_cache()
                    cache_after = memory_snapshot(torch, device)
                    time.sleep(args.idle_seconds)
                    write_row(args.output, dict(event='before_pipeline', run=current_run,
                        trim_cache=trim_cache, cache_before=cache_before, cache_after=cache_after,
                        after_idle=memory_snapshot(torch, device)))
                    cv2.setRNGSeed(0)
                    synchronize(torch, device)
                    before = time.perf_counter()
                    result, _ = localize_photo(engine, image, matrix, np.zeros(5))
                    synchronize(torch, device)
                    elapsed = (time.perf_counter()-before)*1000
                    totals[size].append(elapsed)
                    write_row(args.output, dict(event='pipeline', run=current_run,
                        resize=size, total_ms=elapsed, result=result.as_dict(),
                        diagnostics=engine.last_diagnostics,
                        memory=memory_snapshot(torch, device)))
        finally:
            registered._extract = original_extract
    write_row(args.output, dict(event='summary', median_ms={str(k):statistics.median(v) for k,v in totals.items()},
        all_runs_ms={str(k):v for k,v in totals.items()}, includes_first_run=True))


if __name__ == '__main__':
    main()
