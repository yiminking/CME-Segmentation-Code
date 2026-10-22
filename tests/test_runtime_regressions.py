"""Optional runtime regression tests; unavailable scientific dependencies skip tests."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def available(*modules):
    return all(importlib.util.find_spec(name) is not None for name in modules)


class RuntimeRegressionTests(unittest.TestCase):
    @unittest.skipUnless(available('torch', 'torchvision', 'numpy', 'PIL', 'safetensors'), 'Model dependencies unavailable')
    def test_lora_import_does_not_require_working_directory_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = f'import sys; sys.path.insert(0, {str(ROOT)!r}); from src.lora import LoRA_sam'
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code],
                                    cwd=tmp, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(available('sunpy', 'astropy', 'numpy', 'scipy', 'PIL'), 'FITS dependencies unavailable')
    def test_uppercase_fits_extension_produces_png(self):
        import numpy as np
        from astropy.io import fits
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inputs, outputs = base / 'input/event', base / 'output'
            inputs.mkdir(parents=True)
            header = fits.Header({'INSTRUME': 'LASCO', 'DETECTOR': 'C2', 'TELESCOP': 'SOHO',
                                  'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN',
                                  'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec', 'CRPIX1': 512.5,
                                  'CRPIX2': 512.5, 'CRVAL1': 0.0, 'CRVAL2': 0.0,
                                  'CDELT1': 11.9, 'CDELT2': 11.9, 'HGLN_OBS': 0.0,
                                  'HGLT_OBS': 0.0, 'DSUN_OBS': 149597870700.0,
                                  'RSUN_REF': 695700000.0, 'RSUN_OBS': 959.63})
            header['DATE-OBS'] = '2023-07-29T12:00:00'
            fits.writeto(inputs / '000.fts', np.full((1024, 1024), 1e-9), header)
            header['DATE-OBS'] = '2023-07-29T12:12:00'
            fits.writeto(inputs / '001.FTS', np.full((1024, 1024), 1.1e-9), header)
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'preprocess.py'),
                                     '--input', str(inputs.parent), '--output', str(outputs)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = outputs / 'event/running_001.png'
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.size, (1024, 1024))
                self.assertEqual(image.mode, 'RGB')

    @unittest.skipUnless(available('cv2', 'numpy', 'PIL'), 'Image conversion dependencies unavailable')
    def test_unicode_paths_and_matching_proposal_background(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / '\u6d4b\u8bd5'
            images, masks, proposals = base / 'images', base / 'masks', base / 'proposals'
            images.mkdir(parents=True)
            masks.mkdir()
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[4:15, 5:17] = 255
            Image.fromarray(mask).save(masks / 'mask_frame.png')
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(images / 'running_frame.png')
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'png_to_labelme.py'),
                                     '--masks', str(masks), '--images', str(images), '--output', str(proposals)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = proposals / 'running_frame.json'
            data = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual((output.parent / data['imagePath']).resolve(), (images / 'running_frame.png').resolve())
            self.assertEqual(len(data['shapes']), 1)
            self.assertIsNone(data['shapes'][0]['group_id'])


if __name__ == '__main__':
    unittest.main()
