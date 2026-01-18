# Script that identifies which image has the best view of the stack.
# This image can then be used by another script to estimate density.

import numpy as np
import os
from PIL import Image
import argparse
import json

def get_image_with_most_visible_pixels(folder_path):
    max_visible_pixels = -1
    best_image_filename = None
    best_mask = None

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".png")):
            continue

        filepath = os.path.join(folder_path, filename)

        try:
            with Image.open(filepath).convert("RGBA") as img:
                alpha = img.getchannel("A")
                alpha_data = alpha.load()
                width, height = img.size

                visible_count = sum(
                    1
                    for x in range(width)
                    for y in range(height)
                    if alpha_data[x, y] > 127
                )

                if visible_count > max_visible_pixels:
                    max_visible_pixels = visible_count
                    best_image_filename = filename
                    best_mask = np.array(alpha)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    return best_mask, best_image_filename


def find_nadir_image(folder):

    best_mask, best_name = get_image_with_most_visible_pixels(
        os.path.join(folder, "obj_seg")
    )

    # Prefer full-res if available, but fall back to images.
    image_dirs = []
    if os.path.isdir(os.path.join(folder, "images_full_resolution")):
        image_dirs.append(os.path.join(folder, "images_full_resolution"))
    image_dirs.append(os.path.join(folder, "images"))

    # Optional mapping from processed names (e.g. frame_00001.jpg) to original /
    # full-res names (e.g. IMG_....jpg).
    name_mapping = {}
    mapping_path = os.path.join(folder, "mapping_to_fullres_names.json")
    if os.path.isfile(mapping_path):
        try:
            with open(mapping_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                name_mapping = loaded
        except Exception as e:
            print(f"Failed to read mapping file {mapping_path}: {e}")

    base = os.path.splitext(best_name)[0]

    # Try to find a mapped filename (handle both mapping directions).
    mapped_name = None
    if name_mapping:
        for k in (best_name, base + ".jpg", base + ".png"):
            if k in name_mapping:
                mapped_name = name_mapping[k]
                break
        if mapped_name is None:
            for k, v in name_mapping.items():
                if v in (best_name, base + ".jpg", base + ".png") or os.path.splitext(
                    str(v)
                )[0] == base:
                    mapped_name = k
                    break

    # Search order: mapped name (if any), then raw best_name, then common stems.
    search_names = [n for n in (mapped_name, best_name, base) if n]

    nadir_path = None
    for d in image_dirs:
        for nm in search_names:
            stem, ext = os.path.splitext(nm)
            candidates = [nm] if ext else [stem + ".png", stem + ".jpg", stem + ".jpeg"]
            for cand in candidates:
                p = os.path.join(d, cand)
                if os.path.isfile(p):
                    nadir_path = p
                    break
            if nadir_path:
                break
        if nadir_path:
            break

    if nadir_path is None:
        raise FileNotFoundError(
            f"Could not find nadir image for mask '{best_name}'. "
            f"Tried mapped name '{mapped_name}' in {image_dirs}."
        )

    nadir_image = Image.open(nadir_path)

    best_mask = (best_mask).astype(np.uint8)  # Normalize if necessary
    mask_image = Image.fromarray(best_mask)
    mask_image = mask_image.resize(nadir_image.size, Image.BILINEAR)

    mask_image_np = np.array(mask_image) / 255.0
    mask = mask_image_np > 0.5

    coords = np.argwhere(mask)

    (y_min, x_min), (y_max, x_max) = coords.min(0), coords.max(0)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Compute 5% padding
    pad_x = int(0.05 * bbox_width)
    pad_y = int(0.05 * bbox_height)

    x_min = max(0, x_min + pad_x)
    x_max = min(nadir_image.width, x_max - pad_x)
    y_min = max(0, y_min + pad_y)
    y_max = min(nadir_image.height, y_max - pad_y)

    print("Cropping", (x_min, y_min, x_max, y_max), "from", np.array(nadir_image).shape)
    cropped_nadir = nadir_image.crop((x_min, y_min, x_max, y_max))

    # Save cropped image
    cropped_nadir.save(os.path.join(folder, "nadir.png"))
    print("Saved nadir to", os.path.join(folder, "nadir.png"))

    return np.array(cropped_nadir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply density network")
    parser.add_argument("folder", default="data/pasta", type=str, help="folder name")
    args = parser.parse_args()

    best_mask, best_image_filename = get_image_with_most_visible_pixels(
        os.path.join(args.folder, "obj_seg")
    )
    print(best_image_filename)

    find_nadir_image(args.folder)
