import tempfile
import unittest
from pathlib import Path
from PIL import Image

from viewsync_builder.project import Project, exclusive_project


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_import_preserves_original_and_normalizes_exif_once(self):
        source = self.root / 'photo.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (40, 20), 'orange').save(source, exif=exif)
        original = source.read_bytes()
        project = Project.create(self.root / 'project', '研究')
        record = project.import_image(source)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual((record['width'], record['height']), (20, 40))
        self.assertEqual(len(Project.open(project.root).manifest['images']), 1)
        self.assertEqual(project.import_image(source)['id'], record['id'])
        self.assertEqual((project.root / record['original']).read_bytes(), original)

    def test_project_creation_never_overwrites_existing_folder(self):
        Project.create(self.root / 'project', 'first')
        with self.assertRaises(FileExistsError):
            Project.create(self.root / 'project', 'second')

    def test_two_jobs_cannot_write_the_same_project(self):
        project = Project.create(self.root/'project', 'test')
        with exclusive_project(project.root):
            with self.assertRaisesRegex(ValueError, 'running'):
                with exclusive_project(project.root):
                    self.fail('A second job acquired the same project')
        with exclusive_project(project.root):
            pass

    def test_evaluation_points_and_duplicate_observations_are_validated(self):
        project = Project.create(self.root / 'project', 'test')
        point = dict(id='g1', role='evaluation', xyz=[1, 2, 3], enabled=True,
                     observations=[dict(image_id='missing', uv=[5, 6])])
        with self.assertRaises(ValueError):
            project.set_controls([point])


if __name__ == '__main__':
    unittest.main()
