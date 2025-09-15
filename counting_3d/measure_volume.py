# Corentin Dumery, 2025
# If this code is useful to you in any way, please consider citing our paper,
# Counting Stacked Objects, ICCV25 (Oral)

import torch
import numpy as np
import json
import argparse
import os
import matplotlib.pyplot as plt

from utils.camera_projection import project_points
from utils.append_to_json import append_to_json


def load_cameras_json(camera_json_path):
    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)

    # Extract intrinsics
    fx, fy = camera_data["fl_x"], camera_data["fl_y"]
    cx, cy = camera_data["cx"], camera_data["cy"]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    camera_matrices = []
    image_paths = []

    camera_data["frames"].sort(key=lambda x: x["file_path"])

    for frame in camera_data["frames"]:
        transform_matrix = np.array(frame["transform_matrix"])

        # Invert the transformation matrix to get world-to-camera transform
        world_to_camera = np.linalg.inv(transform_matrix)

        camera_matrices.append(world_to_camera[:3, :])  # Extract 3x4 extrinsic matrix
        image_paths.append(frame["file_path"])

    # Convert to NumPy arrays
    camera_matrices = np.array(camera_matrices)
    return camera_matrices, intrinsics, image_paths


def generate_voxels(points, grid_size=128):
    """
    Generates the set of voxels from a 128³ grid that contain at least one point.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        grid_size (int, optional): The size of the voxel grid (default is 128³).

    Returns:
        set: A set of tuples representing the occupied voxel coordinates.
    """
    # Normalize points to [0, 1] assuming they are within a known bounding box
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    scaling = 1 / (max_bounds - min_bounds).max()
    normalized_points = (points - min_bounds) * scaling

    # Scale to [0, grid_size) and convert to integer indices
    voxel_indices = (normalized_points * (grid_size - 1)).astype(int)

    # Use a set to store unique voxel coordinates
    occupied_voxels = set(map(tuple, voxel_indices))

    return occupied_voxels, scaling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs for the script.")

    # String argument with a default value
    parser.add_argument(
        "--splatfacto_path",
        type=str,
        default="../my_nerfstudio/outputs/pong19_alpha/splatfacto/2024-10-18_170010/nerfstudio_models/step-000029999.ckpt",
        help='Path to the output .ckpt of splatfacto")',
    )
    parser.add_argument(
        "--cameras_path",
        type=str,
        default=None,
        help='Path to cameras. Used to filter background gaussians.")',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help='Path to save results and visualization.")',
    )

    parser.add_argument("--box-thickness", type=float, required=True)

    args = parser.parse_args()

    loaded_state = torch.load(args.splatfacto_path, map_location="cpu")
    points = loaded_state["pipeline"]["_model.gauss_params.means"].cpu().numpy()
    opacities = loaded_state["pipeline"]["_model.gauss_params.opacities"].cpu().numpy()

    mask = opacities[:, 0] >= 0  # Flatten the (N,1) array to (N,)
    points = points[mask]

    print("Reading means from 3DGS of shape", points.shape)
    print("Points MAX:", np.max(points, axis=0))
    print("Points MIN:", np.min(points, axis=0))

    # Dividing the bounding box into a 100**3 grid,
    # Generate the voxels that contain at least one point and visualize them in 3d with matplotlib
    resolution = 100

    camera_json_path = (
        args.cameras_path
    )  # "data/real_units/unit_cross_masked/transforms.json"

    import os
    import numpy as np
    import glob
    import open3d as o3d

    folder_path = args.save_path
    raw_depth_files = sorted(glob.glob(os.path.join(folder_path, "raw_depth", "*.npy")))
    acc_files = sorted(glob.glob(os.path.join(folder_path, "acc", "*.npy")))
    raw_depths = [np.load(depth) for depth in raw_depth_files]
    accs = [np.load(acc) for acc in acc_files]

    # Load the cameras
    camera_matrices, intrinsics, image_paths = load_cameras_json(camera_json_path)
    image_files = [os.path.join(folder_path, im) for im in image_paths]

    assert len(accs) == len(image_files)
    assert len(raw_depths) == len(image_files)

    # Generate resolution x resolution x resolution grid around BB
    # It is initialized to 0, meaning "0 camera vote to remove this point"
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    scaling = 1 / (max_bounds - min_bounds).max()
    voxel_grid = np.zeros((resolution, resolution, resolution))
    voxel_size = (max_bounds - min_bounds) / resolution
    voxel_half_size = np.max(voxel_size) / 2

    # Function that maps from voxel index to 3D point at center of voxel
    def voxel_to_point(i, j, k):
        """Convert voxel indices (i,j,k) to a 3D coordinate in real-world space."""
        x = min_bounds[0] + (i + 0.5) * (max_bounds[0] - min_bounds[0]) / resolution
        y = min_bounds[1] + (j + 0.5) * (max_bounds[1] - min_bounds[1]) / resolution
        z = min_bounds[2] + (k + 0.5) * (max_bounds[2] - min_bounds[2]) / resolution
        return np.array([x, y, z])

    # Generate 3D voxel points and store their (i, j, k) indices
    voxel_points = []
    voxel_indices = []  # Store original (i, j, k) indices

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                voxel_points.append(voxel_to_point(i, j, k))
                voxel_indices.append((i, j, k))  # Store voxel grid indices

    voxel_points = np.array(voxel_points)
    voxel_indices = np.array(voxel_indices)  # Nx3 array of (i, j, k)

    for cam_idx, (cam_matrix, raw_depth, acc) in enumerate(
        zip(camera_matrices, raw_depths, accs)
    ):

        # Project all remaining voxels onto camera
        projected_2d, depths = project_points(voxel_points, cam_matrix, intrinsics)

        # Filter valid projections within the image
        img_h, img_w = raw_depth.shape
        valid_mask = (
            (projected_2d[:, 0] >= 0)
            & (projected_2d[:, 0] < img_w)
            & (projected_2d[:, 1] >= 0)
            & (projected_2d[:, 1] < img_h)
        )

        projected_2d = projected_2d[valid_mask].astype(int)
        depths = depths[valid_mask]
        valid_voxel_indices = voxel_indices[valid_mask]

        count = 0
        for (px, py), voxel_depth, (i, j, k) in zip(
            projected_2d, depths, valid_voxel_indices
        ):

            pixel_acc = acc[py, px]
            pixel_depth = raw_depth[py, px]

            # If voxel projects to a pixel with low accuracy, increase vote
            if pixel_acc < 0.95:
                voxel_grid[i, j, k] += 1
                count += 1

            # If voxel depth is closer than pixel depth (using threshold), increase vote
            elif -voxel_depth < pixel_depth:  # - voxel_half_size:
                voxel_grid[i, j, k] += 1

        print("Added", count, "/", resolution**3)

    # Remove all voxels that have at least 5
    voxel_grid[voxel_grid >= 5] = -1

    if args.box_thickness > 0:
        from scipy.ndimage import binary_dilation

        # Using:
        # voxel_size = (max_bounds - min_bounds) / resolution # one value for each 3D axis
        # We want to carve out the thickness of the box on all sides except the top.
        # The thickness (args.box_thickness) is defined as a % of the total bounding box size.

        voxel_size = (max_bounds - min_bounds) / resolution  # Shape: (3,)

        # Compute thickness in world units (X-axis based)
        thickness_world = (
            max_bounds[0] - min_bounds[0]
        ) * args.box_thickness  # Scalar value

        # Convert thickness to voxel units
        thickness_voxels = int(
            np.round(thickness_world / voxel_size[0])
        )  # Ensure integer value
        print("Thickness:", thickness_voxels)
        if thickness_voxels >= 1:

            voxel_size = (max_bounds - min_bounds) / resolution  # Shape: (3,)

            # Compute thickness in world units (X-axis based)
            thickness_world = (
                max_bounds[0] - min_bounds[0]
            ) * args.box_thickness  # Scalar value

            # Convert thickness to voxel units
            thickness_voxels = int(
                np.round(thickness_world / voxel_size[0])
            )  # Ensure integer value

            inside_voxels = voxel_grid >= 0  # Voxels inside the reconstructed box
            outside_voxels = voxel_grid == -1  # Voxels outside the reconstruction

            # roll: shift by -1; ^: XOR, &: AND
            # **Compute transitions along X and Y (side boundaries)**
            x_transition = (
                np.roll(inside_voxels, shift=-1, axis=0) ^ inside_voxels
            )  # Left/right boundary
            y_transition = (
                np.roll(inside_voxels, shift=-1, axis=1) ^ inside_voxels
            )  # Front/back boundary

            # **Compute bottom transition along Z (only bottom, not top)**
            # Shift OUTSIDE up by 1, then AND
            z_transition = (
                np.roll(outside_voxels, shift=-1, axis=2) & inside_voxels
            )  # Bottom boundary

            # **Create full transition mask (combining all transitions)**
            transition_mask = x_transition | y_transition | z_transition

            # **Dilate the mask to create the thickness shell**
            struct_element = np.ones(
                (thickness_voxels, thickness_voxels, thickness_voxels)
            )  # Dilation kernel
            dilated_mask = binary_dilation(transition_mask, structure=struct_element)

            # **Apply the carving: Only carve where the mask is dilated**
            voxel_grid[dilated_mask] = -2

    # Measure fraction of voxels remaining
    remaining_voxels = np.sum(voxel_grid >= 0)
    total_voxels = np.prod(voxel_grid.shape)
    fraction_remaining = remaining_voxels / total_voxels

    # Compute final volume in real-world scale
    volume = remaining_voxels * np.prod(voxel_size)

    print(f"Fraction of voxels remaining: {fraction_remaining:.4f}")
    print(f"Estimated object volume: {volume:.4f} m3")
    print("Estimated Volume (cm3 or ml):", volume * 1000000)

    append_to_json(args.save_path + "/results.json", "volume_in_cm3", volume * 1000000)
    append_to_json(args.save_path + "/results.json", "volume_in_m3", volume)
