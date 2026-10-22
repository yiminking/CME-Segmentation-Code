"""Initial polygon proposals.
This is not the final manually refined annotation or JSON-to-mask export.
"""
import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np

def mask_to_polygons(mask):
    """
    Convert a binary mask into polygon coordinates for LabelMe.
    """
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        if len(contour) < 3:  # A polygon requires at least three vertices
            continue
        contour = contour.squeeze(1).tolist()
        polygons.append(contour)
    return polygons


def process_all_masks(mask_root, image_root, output_root):
    mask_root, image_root, output_root = map(Path, (mask_root, image_root, output_root))
    count = 0
    for mask_path in sorted(mask_root.rglob("*")):
        if mask_path.suffix.lower() != ".png":
            continue
        rel = mask_path.relative_to(mask_root)
        image_name = "running_" + rel.name[len("mask_"):] if rel.name.startswith("mask_") else rel.name
        image_path = image_root / rel.parent / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Matching running-difference image not found: {image_path}")
        # Decode from bytes to support Unicode paths on Windows as well.
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None or image is None:
            raise ValueError(f"Cannot decode image/mask pair: {mask_path}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Image/mask dimensions differ: {mask_path}")
        h, w = mask.shape
        target = output_root / rel.parent / Path(image_name).with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "5.3.1", "flags": {},
            "shapes": [{"label": "cme", "points": pts, "group_id": None,
                        "shape_type": "polygon", "flags": {}} for pts in mask_to_polygons(mask)],
            "imagePath": os.path.relpath(image_path.resolve(), target.parent.resolve()).replace(os.sep, "/"),
            "imageData": None, "imageHeight": h, "imageWidth": w,
        }
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
        print(f"Saved proposal: {target}")
    if not count:
        raise ValueError("No mask PNG files found")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert predicted masks to initial LabelMe polygons.")
    parser.add_argument("--masks", default="./predicted_masks")
    parser.add_argument("--images", default="./processed_png")
    parser.add_argument("--output", default="./proposals")
    args = parser.parse_args()
    if not Path(args.masks).is_dir() or not Path(args.images).is_dir():
        parser.error("Both mask and image directories must exist")
    process_all_masks(args.masks, args.images, args.output)
