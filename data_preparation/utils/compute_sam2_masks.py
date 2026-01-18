
import hydra
hydra.initialize(config_path="../../ext/sam2/sam2/configs/sam2.1")

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import shutil


# https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb


device = torch.device("cuda")
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "weights/sam2.1_hiera_large.pt"
model_cfg = "sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def copy_and_rename_images(source_folder, destination_folder, shift=0):
    os.makedirs(destination_folder, exist_ok=True)

    images = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.bmp'))])
    if shift:
        images = images[shift:] + images[:shift]
    for index, image in enumerate(images):
        source_path = os.path.join(source_folder, image)
        new_file_name = f"{index + 1}.jpg"  # TODO SEE IMPORTANT NOTE
        destination_path = os.path.join(destination_folder, new_file_name)
        shutil.copy2(source_path, destination_path)

    print(f"Copied and renamed {len(images)} images from '{source_folder}' to '{destination_folder}'.")


if __name__ == "__main__":
    # IMPORTANT NOTE: SAM2 ONLY EXPECTS JPG, SO WE RENAME PNGs TO JPGs.... and it still works
    import argparse
    parser = argparse.ArgumentParser(description="Scale camera transforms using 3D point correspondences and real-world distance.")
    parser.add_argument('--data', type=str, required=True, help="Path to the input folder.")
    parser.add_argument('--shift', type=int, default=-1, help="optional heatmap as input.")
    parser.add_argument('--type', type=str, choices=["base", "heatmap", "box", "objects"], default="heatmap", help="Choose from 'heatmap', 'box', or 'objects'.")

    args = parser.parse_args()

    SHIFT = 2
    if args.shift != -1:
        SHIFT = args.shift

    copy_and_rename_images(args.data + "/images", args.data + "/images_sam2", SHIFT)

    video_dir = args.data + "/images_sam2/"

    image = Image.open(args.data + "/images_sam2/1.jpg")
    image = np.array(image.convert("RGB"))

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    initial_frame_names = [
        p for p in os.listdir(args.data + "/images")
        if os.path.splitext(p)[-1] in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    initial_frame_names.sort()

    if SHIFT:
        initial_frame_names = initial_frame_names[SHIFT:] + initial_frame_names[:SHIFT]


    inference_state = predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    ###################################################
    def _get_points_and_labels_for_first_frame():
        if args.type in ["base", "box", "objects"]:
            # Heatmap not provided, ask user to select points
            points = []
            labels = []

            def onclick(event):
                # Ignore clicks outside the axes
                if event.xdata is None or event.ydata is None:
                    return
                x, y = int(event.xdata), int(event.ydata)
                if event.button == 1:  # Left click - positive (green)
                    points.append([x, y])
                    labels.append(1)
                    plt.plot(x, y, "go")
                elif event.button == 3:  # Right click - negative (red)
                    points.append([x, y])
                    labels.append(0)
                    plt.plot(x, y, "ro")
                plt.draw()

            fig, ax = plt.subplots()
            ax.imshow(image)
            fig.canvas.mpl_connect("button_press_event", onclick)
            plt.title("Select points (left=positive, right=negative). Close window when done.")
            plt.show()

            points = np.array(points, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
            print("Points:", points)
            print("Labels:", labels)
            return points, labels

        # Heatmap mode: auto-select points from heatmaps
        heatmap_box = Image.open(args.data + "/selected_sam_heatmap_box.png").convert("L")
        heatmap_obj = Image.open(args.data + "/selected_sam_heatmap_obj.png").convert("L")

        image_size = (image.shape[1], image.shape[0])  # (width, height)
        heatmap_box = heatmap_box.resize(image_size, Image.BILINEAR)
        heatmap_obj = heatmap_obj.resize(image_size, Image.BILINEAR)

        heatmap_box_np = np.array(heatmap_box) / 255.0  # Normalize to [0, 1]
        heatmap_obj_np = np.array(heatmap_obj) / 255.0

        points = []
        labels = []
        from scipy.spatial.distance import cdist

        def sample_points_from_heatmap(heatmap, label, num_points=4, min_distance=20):
            # Get all pixels with intensity > 0.5
            yx_coords = np.column_stack(np.where(heatmap > 0.5))
            if len(yx_coords) == 0:
                return

            intensities = heatmap[yx_coords[:, 0], yx_coords[:, 1]]
            sorted_indices = np.argsort(-intensities)  # Descending order
            sorted_coords = yx_coords[sorted_indices]
            intensities = intensities[sorted_indices]

            selected = [sorted_coords[0]]
            for _ in range(1, min(num_points, len(sorted_coords))):
                distances = cdist(sorted_coords, np.array(selected))
                valid_mask = np.all(distances >= min_distance, axis=1)
                if not np.any(valid_mask):
                    break
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[np.argmax(intensities[valid_indices])]
                selected.append(sorted_coords[best_idx])
                sorted_coords = np.delete(sorted_coords, best_idx, axis=0)
                intensities = np.delete(intensities, best_idx)

            for y, x in selected:
                points.append([x, y])
                labels.append(label)

        sample_points_from_heatmap(heatmap_box_np, label=1, num_points=1)
        sample_points_from_heatmap(heatmap_obj_np, label=1, num_points=5)

        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print("Auto-selected points:", points)
        print("Labels:", labels)
        return points, labels

    ###################################################

    while True:
        predictor.reset_state(inference_state)
        points, labels = _get_points_and_labels_for_first_frame()

        if points is None or len(points) == 0:
            print("No points selected. Please try again.")
            continue

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # Preview the mask on the first frame for user approval
        plt.figure(figsize=(9, 6))
        plt.title(f"Preview mask on frame {ann_frame_idx} (close window to answer in terminal)")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.show()

        approve = input("Approve this mask and propagate to all frames? [y/N]: ").strip().lower()
        if approve in ("y", "yes"):
            break
        print("Mask rejected. Let's try again.")
        plt.close("all")

    # Run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    if args.type == "base": 
        vis_frame_stride = 40
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            plt.show()

    folder_out = args.data + "/images_alpha"
    if args.type == "box":
        folder_out = args.data + "/box_seg"
    if args.type == "objects":
        folder_out = args.data + "/obj_seg"
    os.makedirs(folder_out, exist_ok=True)

    if args.type == "objects":
        best_view = 0
        best_score = -1
        best_mask = None

        for out_frame_idx in range(0, len(frame_names), 1):
            test = 1
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                assert test == 1
                score = np.sum(out_mask)
                if score > best_score:
                    best_view = out_frame_idx
                    best_score = score
                    best_mask = out_mask
                test += 1

        # Find the original name of that image
        import json
        base_folder = args.data.replace("_colmap", "")
        print("base_folder", base_folder)
        import os
        import json

        mapping_path = os.path.join(base_folder + "_colmap", "mapping_to_fullres_names.json")

        # Check if the JSON file exists before attempting to load it
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            original_name = mapping.get(initial_frame_names[best_view], initial_frame_names[best_view])
        else:
            original_name = initial_frame_names[best_view]  # Keep the same name if JSON doesn't exist

        original_image_path = os.path.join(base_folder, "images", original_name)
        nadir_image = Image.open(original_image_path)

        print(best_mask.shape)
        best_mask = (best_mask[0] * 255).astype(np.uint8)  # Normalize if necessary
        mask_image = Image.fromarray(best_mask)
        mask_image = mask_image.resize(nadir_image.size, Image.BILINEAR)

        mask_image_np = np.array(mask_image) / 255.0
        mask = mask_image_np > 0.5

        coords = np.argwhere(mask)
        (y_min, x_min), (y_max, x_max) = coords.min(0), coords.max(0)

        extend = True
        if extend:
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Compute 5% padding
            pad_x = int(0.1 * bbox_width)
            pad_y = int(0.1 * bbox_height)

            x_min = max(0, x_min + pad_x)
            x_max = min(nadir_image.width, x_max - pad_x)
            y_min = max(0, y_min + pad_y)
            y_max = min(nadir_image.height, y_max - pad_y)

        print("Cropping", (x_min, y_min, x_max, y_max), "from", np.array(nadir_image).shape)
        cropped_nadir = nadir_image.crop((x_min, y_min, x_max, y_max))

        # Save cropped image
        cropped_nadir.save(os.path.join(args.data, "nadir.png"))
        print("Saved nadir to", os.path.join(args.data, "nadir.png"))
        
        
    for out_frame_idx in range(0, len(frame_names), 1):
        image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        image_np = np.array(image)

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if out_mask.ndim == 2:
                out_mask = np.expand_dims(out_mask, axis=-1)  # Expand to (H, W, 1)

            alpha_channel = (out_mask[0] * 255).astype(np.uint8)
            image_with_alpha = np.dstack([image_np, alpha_channel])

            #image_np[:, :, 3] = alpha_channel[:, :, 0]  # Apply the mask to the alpha channel
            
            # Convert back to an image
            image_with_alpha = Image.fromarray(image_with_alpha, 'RGBA')
            
            # Save the image with the alpha mask applied
            output_path = os.path.join(folder_out, initial_frame_names[out_frame_idx][:-4] + ".png")
            image_with_alpha.save(output_path)
    
    shutil.rmtree(args.data + "/images_sam2")