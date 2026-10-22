"""Reuse an ALIKED dense map and compute descriptors only for requested points.

The installed ALIKED detector and coordinate conversion are retained. A dense
retry computes only descriptors missing from the first spatially selected set.
No weights or installed package files are modified.
"""
import time

import numpy as np

from guided_matching import spatial_indices


class ALIKEDQuery:
    def __init__(self, torch, extractor, tensor, resize):
        from lightglue.utils import ImagePreprocessor

        self.torch = torch
        self.extractor = extractor
        if tensor.dim() == 3:
            tensor = tensor[None]
        if tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError('ALIKEDQuery expects one RGB image')
        started = time.perf_counter()
        shape = tensor.shape[-2:][::-1]
        preprocess = {} if resize <= 0 else {'resize': resize}
        image, scales = ImagePreprocessor(
            **{**extractor.preprocess_conf, **preprocess})(tensor)
        preprocessed = time.perf_counter()
        self.feature_map, score_map = extractor.extract_dense_map(image)
        keypoints, scores, _ = extractor.dkd(score_map)
        self.normalized = keypoints[0]
        self.scores = scores[0]
        h, w = image.shape[-2:]
        wh = torch.tensor([w-1, h-1], device=image.device)
        pixels = wh * (self.normalized + 1) / 2.
        self.points = (pixels + .5) / scales - .5
        # The spatial selector uses the same original-image coordinates as the
        # previous post-extraction selector. This also completes detector work.
        self.points_numpy = self.points.detach().cpu().numpy()
        self.scores_numpy = self.scores.detach().cpu().numpy()
        self.image_size = torch.tensor(shape)[None].to(image).float()
        self.count = len(self.points_numpy)
        self.ready = np.zeros(self.count, bool)
        self.descriptors = None
        self.timings_ms = dict(
            preprocess=(preprocessed-started)*1000,
            dense_detection=(time.perf_counter()-preprocessed)*1000,
            descriptor=0.)
        self.descriptor_points_computed = 0

    def features(self, budget=None):
        indices = (np.arange(self.count, dtype=np.int64) if budget is None else
                   spatial_indices(self.points_numpy, self.scores_numpy, budget))
        missing = indices[~self.ready[indices]]
        if len(missing):
            started = time.perf_counter()
            selected = self.torch.as_tensor(missing, device=self.points.device,
                                           dtype=self.torch.long)
            descriptors, _ = self.extractor.desc_head(
                self.feature_map, [self.normalized.index_select(0, selected)])
            values = descriptors[0]
            if self.descriptors is None:
                self.descriptors = values.new_empty((self.count, values.shape[-1]))
            self.descriptors.index_copy_(0, selected, values)
            self.ready[missing] = True
            self.descriptor_points_computed += len(missing)
            self.timings_ms['descriptor'] += (time.perf_counter()-started)*1000
        selected = self.torch.as_tensor(indices, device=self.points.device,
                                       dtype=self.torch.long)
        # Empty images are handled by the localizer before matching.
        descriptors = (self.points.new_empty((0, self.feature_map.shape[1]))
                       if self.descriptors is None else
                       self.descriptors.index_select(0, selected))
        return dict(keypoints=self.points.index_select(0, selected)[None],
                    keypoint_scores=self.scores.index_select(0, selected)[None],
                    descriptors=descriptors[None], image_size=self.image_size)
