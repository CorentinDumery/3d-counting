import os
import json
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def load_and_resize(image_path, target_size=None):
    """Load an image and resize it to a common size if needed."""
    img = Image.open(image_path).convert("RGB")  # Convert to standard format

    if target_size:
        img = img.resize(target_size, Image.BILINEAR)  # Resize to match reference
    return np.array(img)

from skimage.metrics import structural_similarity as ssim

def compute_best_match(img_from, to_images, to_filenames, threshold=0.95):
    """Finds the most similar image using Structural Similarity (SSIM)."""
    best_match = None
    best_score = -1

    for img_to, filename in zip(to_images, to_filenames):
        # Resize img_to to match img_from
        img_to_resized = Image.fromarray(img_to).resize(img_from.shape[:2][::-1], Image.BILINEAR)
        img_to_resized = np.array(img_to_resized)

        # Determine a safe `win_size`
        min_dim = min(img_from.shape[:2])  # Smallest dimension (height or width)
        win_size = max(7, min_dim - 1)  # Ensure win_size is smaller than image size

        # Compute SSIM with a safe window size
        score = ssim(img_from, img_to_resized, channel_axis=2, win_size=win_size)

        if score > best_score:
            best_match = filename
            best_score = score

        if best_score >= threshold:  # Stop early if perfect match is found
            break

    return best_match, best_score


def find_similar_images(folder_from, folder_to, output_json="mapping.json", threshold=0.95):
    """Finds visually similar images between two folders and saves the mapping."""
    
    # Ensure both folders exist
    if not os.path.isdir(folder_from) or not os.path.isdir(folder_to):
        raise FileNotFoundError("One or both input folders do not exist.")

    # Load images in folder_to for comparison
    to_filenames = sorted(os.listdir(folder_to))  # Preserve order
    to_images = [load_and_resize(os.path.join(folder_to, fname)) for fname in to_filenames]

    # Process images in folder_from
    mapping = {}
    from_filenames = sorted(os.listdir(folder_from))  # Preserve order
    for i, img_name in enumerate(tqdm(from_filenames, desc="Matching images")):
        img_path = os.path.join(folder_from, img_name)
        img_from = load_and_resize(img_path)

        best_match, best_score = compute_best_match(img_from, to_images, to_filenames, threshold)

        if best_match:
            mapping[img_name] = best_match
        else:
            raise ValueError(f"No visually similar image found for {img_name} in {folder_to}.")

    # Save mapping as JSON
    with open(output_json, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"✅ Mapping saved to {output_json}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find visually similar images between two folders and save mapping.")
    parser.add_argument("folder_from", help="Path to the source folder containing images to match.")
    parser.add_argument("folder_to", help="Path to the target folder containing images to compare.")
    parser.add_argument("--output", default="mapping.json", help="Output JSON filename (default: mapping.json)")
    parser.add_argument("--threshold", type=float, default=0.99, help="SSIM threshold for matching (default: 0.95)")

    args = parser.parse_args()
    find_similar_images(args.folder_from, args.folder_to, args.output, args.threshold)
