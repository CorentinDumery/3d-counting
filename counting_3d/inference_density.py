import os
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from gamma_model import DinoRegression
from utils.append_to_json import append_to_json
from find_nadir_image import find_nadir_image
from utils.compute_depth import compute_depth, make_model

dino_arch = "dinov2_vitb14"
feats_dim = 768 if dino_arch == "dinov2_vitb14" else 1024
encoded_image_size = 32
image_size = 14 * encoded_image_size

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply density network")

    parser.add_argument(
        "--exp-name", default="1_0", type=str, metavar="PATH", help="experiment name"
    )

    parser.add_argument(
        "--input-folder",
        default="",
        type=str,
        metavar="PATH",
        help="image for inference",
    )  # required=True,
    parser.add_argument(
        "--output-folder",
        default="",
        type=str,
        metavar="PATH",
        help="image for inference",
    )  # required=True,
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = find_nadir_image(args.input_folder)

    depth_model = make_model()
    depth_uint8 = compute_depth(depth_model, img)
    depth_map = depth_uint8

    if args.output_folder != "":
        plt.imsave(args.output_folder + "/estimated_depth.png", depth_map)
        plt.imsave(args.output_folder + "/nadir.png", img)

    config = {
        "device": device,
        "MODEL_CONFIG": {
            "feats_dim": feats_dim,
            "encoded_image_size": encoded_image_size,
        },
    }
    dino_arch = "dinov2_vitb14"
    model = DinoRegression(dino_arch, config)  # Initialize with any required arguments

    print(f"LOADING density_net_{args.exp_name}.pth")
    model_path = os.path.join("weights", f"density_net_{args.exp_name}.pth")
    model.load_state_dict(torch.load(model_path))

    model.eval()

    depth_tensor = torch.tensor(depth_map).unsqueeze(0).unsqueeze(0).half().to(device)

    def crop_tensor(tensor, crop_fraction=0.3):
        _, _, h, w = tensor.shape  # Assuming tensor has shape (C, H, W)
        crop_h = int(h * crop_fraction)
        crop_w = int(w * crop_fraction)
        cropped_tensor = tensor[:, :, crop_h : h - crop_h, crop_w : w - crop_w]
        return cropped_tensor

    depth_tensor = crop_tensor(depth_tensor)

    depth_resized = F.interpolate(
        depth_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    # Crop the image in the center
    _, _, H, W = depth_tensor.shape
    crop_size = image_size
    start_x = (W - crop_size) // 2
    start_y = (H - crop_size) // 2
    depth_cropped = depth_tensor[
        :, :, start_y : start_y + crop_size, start_x : start_x + crop_size
    ]

    if H < image_size or W < image_size:
        # Cannot just crop, fallback to resize
        depth_cropped = depth_resized
    # Convert to 3 channels by copying the single channel
    depth_resized_3ch = depth_resized.repeat(
        1, 3, 1, 1
    )  # Shape: [1, 3, image_size, image_size]
    depth_cropped_3ch = depth_cropped.repeat(
        1, 3, 1, 1
    )  # Shape: [1, 3, crop_size, crop_size]

    depth_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    depth = depth_resized_3ch
    depth /= depth.max()

    print("Range of depth:", depth.min().item(), "to", depth.max().item())
    assert depth.shape == (1, 3, image_size, image_size), f"got shape {depth.shape}"
    normalized_depth_tensor = depth_normalize(depth[0]).unsqueeze(
        0
    )  # Apply on a single image

    # normalized_depth_tensor = depth_normalize(depth)  # Apply on a single image
    if args.output_folder != "":
        plt.imsave(
            args.output_folder + "/estimated_depth_cropped.png",
            depth[0, 0].cpu().numpy(),
        )

    out = model(normalized_depth_tensor.float())

    print("Predicted density:", out.item())

    if args.output_folder != "":
        append_to_json(
            args.output_folder + "/results.json",
            "volume_usage_" + args.exp_name,
            out.item(),
        )
