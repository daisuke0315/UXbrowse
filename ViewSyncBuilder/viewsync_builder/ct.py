"""Single-frame DICOM CT geometry and patient-axis MPR."""
from itertools import product
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from scipy.ndimage import map_coordinates


def validate_series(datasets):
    if len(datasets) < 2:
        raise ValueError('A CT volume needs at least two single-frame slices')
    first = datasets[0]
    required = ('Modality', 'SeriesInstanceUID', 'FrameOfReferenceUID', 'ImageOrientationPatient',
                'ImagePositionPatient', 'PixelSpacing', 'Rows', 'Columns')
    for d in datasets:
        if any(not hasattr(d, key) for key in required):
            raise ValueError('Missing DICOM coordinate metadata')
        if d.Modality != 'CT' or int(getattr(d, 'NumberOfFrames', 1)) != 1:
            raise ValueError('Only single-frame CT is supported')
        if (d.SeriesInstanceUID != first.SeriesInstanceUID or
                d.FrameOfReferenceUID != first.FrameOfReferenceUID or
                (d.Rows, d.Columns) != (first.Rows, first.Columns)):
            raise ValueError('Mixed/inconsistent CT series')
        if (not np.allclose(d.ImageOrientationPatient, first.ImageOrientationPatient, atol=1e-5, rtol=0) or
                not np.allclose(d.PixelSpacing, first.PixelSpacing, atol=1e-5, rtol=0)):
            raise ValueError('Inconsistent CT orientation or pixel spacing')
    orientation = np.asarray(first.ImageOrientationPatient, float)
    spacing = np.asarray(first.PixelSpacing, float)
    if (orientation.shape != (6,) or spacing.shape != (2,) or
            not np.isfinite(orientation).all() or not np.isfinite(spacing).all() or np.min(spacing) <= 0):
        raise ValueError('Invalid DICOM orientation/spacing')
    row, column = orientation[:3], orientation[3:]
    if not np.allclose([np.linalg.norm(row), np.linalg.norm(column), row@column], [1, 1, 0], atol=1e-5):
        raise ValueError('DICOM directions must be orthonormal')
    normal = np.cross(row, column)
    positions = np.array([d.ImagePositionPatient for d in datasets], float)
    if positions.shape != (len(datasets), 3) or not np.isfinite(positions).all():
        raise ValueError('Invalid ImagePositionPatient')
    indices = np.argsort(positions @ normal)
    ordered = [datasets[int(i)] for i in indices]
    positions = positions[indices]
    differences = np.diff(positions, axis=0)
    distances = differences @ normal
    if np.min(distances) <= 1e-6:
        raise ValueError('Duplicate CT slice positions')
    declared = [float(d.SpacingBetweenSlices) for d in ordered if hasattr(d, 'SpacingBetweenSlices')]
    if declared:
        if (len(declared) != len(ordered) or not np.isfinite(declared).all() or
                min(declared) <= 0 or not np.allclose(declared, declared[0], atol=1e-5, rtol=0)):
            raise ValueError('Inconsistent declared CT slice spacing')
        base = declared[0]
    else:
        near_minimum = np.isclose(distances, distances.min(), atol=.02, rtol=1e-3)
        if np.count_nonzero(near_minimum) <= len(distances)/2:
            raise ValueError('Cannot determine an unambiguous CT slice spacing')
        base = float(np.median(distances[near_minimum]))
    multiples = np.rint(distances/base)
    if np.min(multiples) < 1 or not np.allclose(distances, multiples*base, atol=.02, rtol=1e-3):
        raise ValueError('Noninteger or irregular CT slice spacing')
    step = np.median(differences/multiples[:, None], axis=0)
    if not np.allclose(differences, multiples[:, None]*step, atol=.02, rtol=1e-3):
        raise ValueError('Inconsistent CT in-plane slice displacement')
    grid = np.concatenate(([0], np.cumsum(multiples))).astype(int)
    if not np.allclose(positions, positions[0]+grid[:, None]*step, atol=.02, rtol=0):
        raise ValueError('CT positions drift from the regular slice grid')
    affine = np.eye(4)
    affine[:3, 0], affine[:3, 1] = row*spacing[1], column*spacing[0]
    affine[:3, 2], affine[:3, 3] = step, positions[0]
    return ordered, affine


def _sample_indices(ijk, dimensions):
    """Indices with nonzero weights in linear Z sampling, including boundary checks."""
    ijk = np.asarray(ijk, float)
    shape = (3,) + (1,)*(ijk.ndim-1)
    inside = np.all((ijk >= -1e-6) & (ijk <= (np.asarray(dimensions)-1).reshape(shape)+1e-6), axis=0)
    z = ijk[2]
    z = np.where(np.isclose(z, np.rint(z), atol=1e-6, rtol=0), np.rint(z), z)
    return np.floor(z).astype(int), np.ceil(z).astype(int), inside


def point_sampling(metadata, xyz):
    if not metadata:
        return dict(status='unavailable', interpolated_indices=[])
    affine = np.asarray(metadata['affine'], float)
    ijk = np.linalg.solve(affine[:3, :3], np.asarray(xyz, float)-affine[:3, 3])
    lo, hi, inside = _sample_indices(ijk, metadata['dimensions'])
    generated = {p['index'] for p in metadata.get('regularization', {}).get('interpolated_slices', [])}
    affected = sorted(generated.intersection((int(lo), int(hi)))) if inside else []
    return dict(status=('interpolated' if affected else 'original') if inside else 'outside',
                series_uid=metadata['series_uid'], voxel_z=float(ijk[2]),
                interpolated_indices=affected)


def annotate_points(points, metadata):
    return [{**p, 'ct_sampling': point_sampling(metadata, p['xyz'])} for p in points]


class CTVolume:
    def __init__(self, voxels, affine, metadata):
        self.voxels = np.asarray(voxels, np.float32)
        self.affine, self.metadata = np.asarray(affine, float), metadata
        self.inverse = np.linalg.inv(self.affine)
        z, y, x = self.voxels.shape
        corners = np.array([self.voxel_to_lps(p) for p in product((0, x-1), (0, y-1), (0, z-1))])
        self.bounds = np.array([corners.min(0), corners.max(0)])

    @classmethod
    def series(cls, folder):
        groups = {}
        for path in sorted(Path(folder).rglob('*')):
            if not path.is_file():
                continue
            try:
                d = pydicom.dcmread(path, stop_before_pixels=True)
            except pydicom.errors.InvalidDicomError:
                continue
            if getattr(d, 'Modality', '') == 'CT':
                key = str(getattr(d, 'SeriesInstanceUID', 'missing'))
                groups.setdefault(key, []).append((path, d))
        return groups

    @classmethod
    def load(cls, folder, series_uid=None):
        groups = cls.series(folder)
        if series_uid is None:
            if len(groups) != 1:
                raise ValueError('Select exactly one CT series')
            series_uid = next(iter(groups))
        if series_uid not in groups:
            raise ValueError('CT series was not found')
        paths = {id(d): path for path, d in groups[series_uid]}
        ordered, affine = validate_series([d for _, d in groups[series_uid]])
        normal = np.cross(affine[:3, 0], affine[:3, 1])
        normal /= np.linalg.norm(normal)
        step_mm = float(affine[:3, 2] @ normal)
        source_indices = np.rint([(np.asarray(d.ImagePositionPatient)-affine[:3, 3])@normal/step_mm
                                 for d in ordered]).astype(int)
        depth = int(source_indices[-1])+1
        first = ordered[0]
        # Reserve the regular grid once; do not hold a second copy of the entire CT.
        voxels = np.empty((depth, int(first.Rows), int(first.Columns)), np.float32)
        for header in ordered:
            d = pydicom.dcmread(paths[id(header)])
            try:
                values = d.pixel_array.astype(np.float32)
            except Exception as error:
                raise ValueError('Cannot decode CT Pixel Data; check transfer syntax support') from error
            slope, intercept = float(getattr(d, 'RescaleSlope', 1)), float(getattr(d, 'RescaleIntercept', 0))
            if not np.isfinite([slope, intercept]).all() or slope == 0 or values.shape != voxels.shape[1:]:
                raise ValueError('Invalid CT rescale or pixel dimensions')
            index = int(round((np.asarray(header.ImagePositionPatient)-affine[:3, 3])@normal/step_mm))
            voxels[index] = values*slope+intercept
        generated = []
        for n, (left, right) in enumerate(zip(source_indices[:-1], source_indices[1:])):
            for index in range(int(left)+1, int(right)):
                weight = (index-left)/(right-left)
                voxels[index] = (1-weight)*voxels[left]+weight*voxels[right]
                generated.append(dict(index=index, position_lps_mm=(affine[:3, 3]+index*affine[:3, 2]).tolist(),
                                      source_indices=[int(left), int(right)],
                                      source_sop_instance_uids=[str(ordered[n].SOPInstanceUID), str(ordered[n+1].SOPInstanceUID)],
                                      weight_right=float(weight), gap_mm=float((right-left)*step_mm)))
        metadata = dict(series_uid=series_uid, frame_of_reference_uid=str(first.FrameOfReferenceUID),
                        coordinate_system='DICOM_LPS', unit='mm', affine=affine.tolist(),
                        slice_positions=[(affine[:3, 3]+i*affine[:3, 2]).tolist() for i in range(depth)],
                        dimensions=[int(first.Columns), int(first.Rows), depth],
                        regularization=dict(method='linear_HU_between_original_slices',
                            original_slices=len(ordered), output_slices=depth, spacing_mm=step_mm,
                            spacing_source='SpacingBetweenSlices' if hasattr(first, 'SpacingBetweenSlices') else 'ImagePositionPatient',
                            source_slice_indices=source_indices.tolist(),
                            source_slice_positions=[list(map(float, d.ImagePositionPatient)) for d in ordered],
                            source_sop_instance_uids=[str(d.SOPInstanceUID) for d in ordered],
                            interpolated_slices=generated))
        return cls(voxels, affine, metadata)

    def point_sampling(self, xyz):
        return point_sampling(self.metadata, xyz)

    def voxel_to_lps(self, ijk):
        return self.affine[:3, :3] @ np.asarray(ijk) + self.affine[:3, 3]

    def lps_to_voxel(self, xyz):
        return self.inverse[:3, :3] @ np.asarray(xyz) + self.inverse[:3, 3]

    def mpr(self, plane, position, size=512, level=40, width=400):
        if plane not in ('axial', 'coronal', 'sagittal') or not 16 <= size <= 1024 or width <= 0:
            raise ValueError('Invalid MPR parameters')
        if not np.isfinite([position, level, width]).all():
            raise ValueError('MPR parameters must be finite')
        fixed, horizontal, vertical = {'axial': (2, 0, 1), 'coronal': (1, 0, 2), 'sagittal': (0, 1, 2)}[plane]
        if not self.bounds[0, fixed] <= position <= self.bounds[1, fixed]:
            raise ValueError('MPR position outside CT bounds')
        step = max(np.ptp(self.bounds[:, horizontal]), np.ptp(self.bounds[:, vertical])) / (size-1)
        origin = self.bounds.mean(0)
        origin[fixed] = position
        origin[horizontal] -= step*(size-1)/2
        origin[vertical] -= step*(size-1)/2
        u, v = np.zeros(3), np.zeros(3)
        u[horizontal], v[vertical] = step, step
        yy, xx = np.mgrid[:size, :size]
        xyz = origin[:, None, None] + u[:, None, None]*xx + v[:, None, None]*yy
        ijk = np.einsum('ij,jhw->ihw', self.inverse[:3, :3], xyz-self.affine[:3, 3, None, None])
        hu = map_coordinates(self.voxels, ijk[[2, 1, 0]], order=1, mode='constant', cval=-1024)
        lo, hi, inside = _sample_indices(ijk, [self.voxels.shape[2], self.voxels.shape[1], self.voxels.shape[0]])
        generated = [p['index'] for p in self.metadata.get('regularization', {}).get('interpolated_slices', [])]
        affected = inside & (np.isin(lo, generated) | np.isin(hi, generated))
        image = Image.fromarray((np.clip((hu-level)/width+.5, 0, 1)*255).astype(np.uint8))
        return image, dict(origin=origin.tolist(), u=u.tolist(), v=v.tolist(),
                           width=size, height=size, coordinate_system='DICOM_LPS', unit='mm',
                           interpolation_affected_pixels=int(np.count_nonzero(affected)))
