import unittest
import tempfile
from pathlib import Path
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage

from viewsync_builder.ct import CTVolume, validate_series


def write_series(folder, positions=(0, 1, 3, 4), spacing_between=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, z in enumerate(positions):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = f'1.2.3.4.{index+1}'
        d = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
        d.SOPClassUID, d.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        d.Modality, d.SeriesInstanceUID, d.FrameOfReferenceUID = 'CT', '1.2.3', '1.2.4'
        d.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        d.ImagePositionPatient = [0, 0, z]
        d.PixelSpacing = [1, 1]
        d.SliceThickness = 1
        if spacing_between is not None:
            d.SpacingBetweenSlices = spacing_between
        d.Rows, d.Columns, d.SamplesPerPixel = 4, 5, 1
        d.PhotometricInterpretation = 'MONOCHROME2'
        d.BitsAllocated, d.BitsStored, d.HighBit, d.PixelRepresentation = 16, 16, 15, 1
        # Interpolate after rescaling: endpoint slopes differ.
        d.RescaleSlope, d.RescaleIntercept = (2 if z == 3 else 1), -100
        d.PixelData = np.full((4, 5), z*10, dtype='<i2').tobytes()
        path = folder/f'{index:03d}.dcm'
        d.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def slices():
    result = []
    for z in (4., 0., 2.):
        d = Dataset()
        d.Modality = 'CT'
        d.SeriesInstanceUID = '1.2.3'
        d.FrameOfReferenceUID = '1.2.4'
        d.ImageOrientationPatient = [0, 1, 0, -1, 0, 0]
        d.ImagePositionPatient = [10, 20, z]
        d.PixelSpacing = [.5, .8]
        d.Rows, d.Columns = 4, 5
        result.append(d)
    return result


class CTTests(unittest.TestCase):
    def test_oblique_direction_and_pixel_spacing_order(self):
        ordered, affine = validate_series(slices())
        self.assertEqual([float(d.ImagePositionPatient[2]) for d in ordered], [0, 2, 4])
        volume = CTVolume(np.zeros((3, 4, 5)), affine, {})
        np.testing.assert_allclose(volume.voxel_to_lps([2, 3, 1]), [8.5, 21.6, 2])
        np.testing.assert_allclose(volume.lps_to_voxel([8.5, 21.6, 2]), [2, 3, 1])
        image, mapping = volume.mpr('axial', 2, size=32)
        self.assertEqual(image.size, (32, 32))
        self.assertEqual(mapping['origin'][2], 2)
        self.assertEqual(mapping['u'][2], 0)

    def test_noninteger_duplicate_or_mixed_slices_are_rejected(self):
        for change in ('noninteger', 'duplicate', 'series', 'orientation', 'frame', 'spacing'):
            data = slices()
            if change == 'noninteger': data[0].ImagePositionPatient[2] = 5
            if change == 'duplicate': data[0].ImagePositionPatient[2] = 0
            if change == 'series': data[0].SeriesInstanceUID = '1.2.99'
            if change == 'frame': data[0].FrameOfReferenceUID = '1.2.99'
            if change == 'orientation': data[0].ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            if change == 'spacing': data[0].PixelSpacing = [0, 1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_series(data)

    def test_missing_slices_are_interpolated_in_hu_and_originals_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_series(temporary)
            originals = [p.read_bytes() for p in paths]
            volume = CTVolume.load(temporary)
            self.assertEqual(volume.voxels.shape, (5, 4, 5))
            np.testing.assert_array_equal(volume.voxels[:, 0, 0], [-100, -90, -65, -40, -60])
            self.assertEqual(originals, [p.read_bytes() for p in paths])
            plan = volume.metadata['regularization']
            self.assertEqual(plan['source_slice_indices'], [0, 1, 3, 4])
            self.assertEqual(plan['spacing_source'], 'ImagePositionPatient')
            self.assertEqual(len(plan['interpolated_slices']), 1)
            gap = plan['interpolated_slices'][0]
            self.assertEqual(gap['index'], 2)
            self.assertEqual(gap['position_lps_mm'], [0, 0, 2])
            self.assertEqual(gap['source_indices'], [1, 3])
            self.assertEqual(gap['weight_right'], .5)
            self.assertEqual(len(volume.metadata['slice_positions']), 5)

    def test_explicit_spacing_supports_consecutive_gaps_but_ambiguous_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_series(temporary, (0, 3, 4), spacing_between=1)
            volume = CTVolume.load(temporary)
            self.assertEqual(volume.voxels.shape[0], 5)
            self.assertEqual(volume.metadata['regularization']['spacing_source'], 'SpacingBetweenSlices')
            self.assertEqual([g['index'] for g in volume.metadata['regularization']['interpolated_slices']], [1, 2])
        with tempfile.TemporaryDirectory() as temporary:
            write_series(temporary, (0, 1, 3))
            with self.assertRaisesRegex(ValueError, 'spacing'):
                CTVolume.load(temporary)

    def test_sampling_marks_interpolated_neighbors_and_mpr_reports_influence(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_series(temporary)
            volume = CTVolume.load(temporary)
            self.assertEqual(volume.point_sampling([2, 2, 1])['status'], 'original')
            self.assertEqual(volume.point_sampling([2, 2, 1.25])['status'], 'interpolated')
            self.assertEqual(volume.point_sampling([2, 2, 2])['interpolated_indices'], [2])
            self.assertEqual(volume.point_sampling([2, 2, 3])['status'], 'original')
            self.assertEqual(volume.point_sampling([2, 2, 5])['status'], 'outside')
            _, original = volume.mpr('axial', 1, size=32)
            _, interpolated = volume.mpr('axial', 2, size=32)
            _, side = volume.mpr('sagittal', 2, size=32)
            self.assertEqual(original['interpolation_affected_pixels'], 0)
            self.assertGreater(interpolated['interpolation_affected_pixels'], 0)
            self.assertGreater(side['interpolation_affected_pixels'], 0)

    def test_tilted_gap_keeps_the_patient_coordinate_grid(self):
        data = []
        c = 2**-.5
        step = np.array([-c+.1, 0, c+.1])
        for index in (0, 1, 3, 4):
            d = slices()[0]
            d.ImageOrientationPatient = [c, 0, c, 0, 1, 0]
            d.ImagePositionPatient = (np.array([10, 20, 30])+index*step).tolist()
            data.append(d)
        _, affine = validate_series(data)
        np.testing.assert_allclose(affine[:3, 2], step)
        np.testing.assert_allclose(affine[:3, 3], [10, 20, 30])

    def test_slice_thickness_does_not_invent_missing_slices(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_series(temporary, (0, 2, 4))
            volume = CTVolume.load(temporary)
            self.assertEqual(volume.voxels.shape[0], 3)
            self.assertEqual(volume.metadata['regularization']['spacing_mm'], 2)
            self.assertEqual(volume.metadata['regularization']['interpolated_slices'], [])
