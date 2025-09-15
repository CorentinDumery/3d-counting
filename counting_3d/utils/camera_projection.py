import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def project_points(points, cam_matrix, intrinsics):
    """
    Projects 3D points onto a 2D image plane using camera extrinsics and intrinsics.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        cam_matrix (np.ndarray): 3x4 camera extrinsic matrix.
        intrinsics (np.ndarray): 3x3 intrinsic camera matrix.

    Returns:
        np.ndarray: Nx2 array of 2D projected points.
    """
    # Convert points to homogeneous coordinates (Nx4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    # Transform world to camera
    points_camera = (cam_matrix @ points_h.T).T

    # Perspective divide
    x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]

    # Avoid division by zero
    z[z == 0] = 1e-6

    # Apply intrinsic transformation before distortion correction
    x_proj = (-intrinsics[0, 0] * x / z) + intrinsics[0, 2]
    y_proj = (intrinsics[1, 1] * y / z) + intrinsics[1, 2]

    # Stack projected points
    projected = np.vstack((x_proj, y_proj)).T

    depth = z.reshape(-1, 1)
    return projected, depth


def visualize_projections(projected_points, image_path=None):
    """
    Visualizes the 2D projected points onto a blank canvas or an image.

    Args:
        projected_points (np.ndarray): Nx2 array of 2D points.
        image_path (str, optional): Path to an image for overlay.
    """
    if image_path:
        try:
            img = Image.open(image_path)
            img_w, img_h = img.size
        except Exception as e:
            print(f"⚠️ Warning: Could not load image ({e}). Using a blank canvas.")
            img = None
            img_w, img_h = 800, 800  # Default blank canvas size
    else:
        img = None
        img_w, img_h = 800, 800

    # Create a figure
    plt.figure(figsize=(8, 8))

    if img is not None:
        plt.imshow(img)  # Show the image if provided
    else:
        plt.xlim(0, img_w)
        plt.ylim(img_h, 0)
        plt.gca().set_facecolor("white")

    # Plot the projected points
    x_vals, y_vals = projected_points[:, 0], projected_points[:, 1]
    plt.scatter(x_vals, y_vals, color="red", s=8, label="Projected Points")

    plt.title("Projected Points on First Camera View")
    plt.axis("off")
    plt.legend()
    plt.show()
