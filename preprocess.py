import os
import numpy as np
from PIL import Image
import sunpy.map
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.map.maputils import all_coordinates_from_map
from scipy.ndimage import median_filter

def apply_sun_center_mask(image_data, smap, factor=2.4):
    """Construct a central solar mask using SunPy coordinates and set masked pixels to zero."""
    coords = all_coordinates_from_map(smap)
    center = SkyCoord(0 * u.deg, 0 * u.deg, frame=smap.coordinate_frame)
    radii = np.sqrt((coords.Tx - center.Tx)**2 + (coords.Ty - center.Ty)**2)
    mask_inner = radii < smap.rsun_obs * factor

    masked_data = np.copy(image_data)
    masked_data[mask_inner] = 0
    return masked_data


def process_diff_image(diff_data, smap, log_offset=14, clip_min=1, clip_max=5):
    """Apply logarithmic scaling, offset, clipping, normalization, median filtering and a central mask."""
    eps = 1e-12

    img_log = np.log10(diff_data + eps) + log_offset
    img_log = np.nan_to_num(img_log, nan=clip_min)
    img_clipped = np.clip(img_log, clip_min, clip_max)

    img_norm = (img_clipped - clip_min) * 255 / (clip_max - clip_min)
    img_uint8 = np.clip(img_norm, 0, 255).astype(np.uint8)

    img_filtered = median_filter(img_uint8, size=3)
    img_masked = apply_sun_center_mask(img_filtered, smap)
    return img_masked


def process_fits_files(root_folder, output_root):
    os.makedirs(output_root, exist_ok=True)

    for root, dirs, files in os.walk(root_folder):
        fits_files = sorted([f for f in files if f.lower().endswith('.fts')])
        if len(fits_files) < 2:
            continue

        previous_map = None

        for fts_file in fits_files:
            fts_path = os.path.join(root, fts_file)

            # Inspect image dimensions
            with fits.open(fts_path, memmap=False) as hdul:
                nx = hdul[0].header.get('NAXIS1', 0)
                ny = hdul[0].header.get('NAXIS2', 0)

            if nx == 512 and ny == 512:
                print(f"Skipping 512x512 image: {fts_file}")
                continue

            current_map = sunpy.map.Map(fts_path)

            if previous_map is None:
                previous_map = current_map
                continue

            # Compute the running difference and flip vertically
            diff = current_map.data - previous_map.data
            diff_flipped = np.flipud(diff)

            # Process the difference image
            img_processed = process_diff_image(diff_flipped, current_map)
            rgb_image = np.stack([img_processed]*3, axis=-1)

            # Construct the output path while preserving the directory structure
            rel_path = os.path.relpath(root, root_folder)
            output_dir = os.path.join(output_root, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            png_path = os.path.join(output_dir, f"running_{os.path.splitext(fts_file)[0]}.png")
            Image.fromarray(rgb_image).save(png_path)
            print(f"Saved PNG: {png_path}")

            previous_map = current_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess selected Level-1 LASCO C2 .fts sequences.")
    parser.add_argument("--input", required=True, help="Event folders containing time-ordered .fts files")
    parser.add_argument("--output", default="./processed_png")
    args = parser.parse_args()
    if not os.path.isdir(args.input):
        parser.error("Input directory does not exist")
    process_fits_files(args.input, args.output)
