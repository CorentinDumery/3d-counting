# Just a regular pipeline, except ns-eval also saves depth and accumulation maps.

from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from dataclasses import dataclass, field
from nerfstudio.configs.base_config import InstantiateConfig

from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from pathlib import Path
import torch
import os


@dataclass
class CountingPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: CountingPipeline)


class CountingPipeline(VanillaPipeline):
    config: CountingPipelineConfig

    def __init__(self, config: CountingPipelineConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Your custom setup here
        print("Using CountingPipeline")

    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        metrics_dict = super().get_average_eval_image_metrics()

        self.eval()

        idx = 0

        import torchvision.utils as vutils
        import numpy as np

        for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
            outputs = self.model.get_outputs_for_camera(camera=camera)
            # height, width = camera.height, camera.width
            # metrics_dict, depth_dict = self.model.get_depth_metrics_and_images(outputs, batch)
            metrics_dict, depth_dict = self.get_depth_metrics_and_images(outputs, batch)

            if output_path is not None:
                os.makedirs(os.path.join(output_path, "acc"), exist_ok=True)
                os.makedirs(os.path.join(output_path, "raw_depth"), exist_ok=True)

                print("Saving depth+acc on image", idx)

                key = "depth"
                predicted_depth = depth_dict[key]  # [H, W, C] order
                min_depth = predicted_depth.min()
                max_depth = predicted_depth.max()
                normalized_depth = (predicted_depth - min_depth) / (
                    max_depth - min_depth + 1e-8
                )  # Normalize to [0, 1]

                # Convert to 8-bit for saving
                depth_image = (
                    (normalized_depth * 1).clamp(0, 1).to(torch.float)
                )  # Explicitly cast to uint8
                depth_image = depth_image.squeeze().unsqueeze(0)

                key = "raw_depth"
                raw_depth = depth_dict[key].squeeze().cpu().numpy()
                np.save(output_path / f"raw_depth/raw_depth_{idx:04d}", raw_depth)
                vutils.save_image(
                    outputs["accumulation"].squeeze().unsqueeze(0).cpu(),
                    output_path / f"acc/acc_{idx:04d}.png",
                )

                np.save(
                    output_path / f"acc/acc_{idx:04d}",
                    outputs["accumulation"].squeeze().cpu().numpy(),
                )
            idx += 1

        return metrics_dict

    def get_depth_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Computes depth metrics and returns depth visualization.

        Args:
            batch: Batch of data containing ground truth.
            outputs: Outputs of the model containing predicted depth.

        Returns:
            A dictionary of depth-related metrics and depth images.
        """

        # Ground truth depth (if available)
        gt_depth = batch.get(
            "depth", None
        )  # Some datasets may not provide ground truth depth
        predicted_depth = outputs["depth"]

        # Normalize predicted depth for visualization
        min_depth = predicted_depth.min()
        max_depth = predicted_depth.max()
        normalized_depth = (predicted_depth - min_depth) / (
            max_depth - min_depth + 1e-8
        )  # Avoid divide by zero

        # Convert to [1, 1, H, W] format for metric calculations
        predicted_depth = predicted_depth[None, None, :, :]
        if gt_depth is not None:
            gt_depth = gt_depth[None, None, :, :]

        # Compute depth error metrics if GT depth is available
        metrics_dict = {}
        if gt_depth is not None:
            abs_rel = torch.mean(
                torch.abs(predicted_depth - gt_depth) / (gt_depth + 1e-8)
            )  # Absolute relative error
            rmse = torch.sqrt(
                torch.mean((predicted_depth - gt_depth) ** 2)
            )  # Root Mean Squared Error (RMSE)

            metrics_dict["abs_rel"] = float(abs_rel.item())
            metrics_dict["rmse"] = float(rmse.item())

        # Convert depth to an image format
        depth_image = (
            (normalized_depth * 255).clamp(0, 255).byte()
        )  # Convert to 8-bit grayscale

        depth_dict = {"depth": depth_image, "raw_depth": predicted_depth}

        return metrics_dict, depth_dict
