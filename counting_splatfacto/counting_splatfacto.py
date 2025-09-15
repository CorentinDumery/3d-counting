"""
CountingSplatfacto Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import (
    NerfactoModel,
    NerfactoModelConfig,
)  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from nerfstudio.viewer.viewer_elements import (
    ViewerButton,
    ViewerSlider,
    ViewerClick,
    ViewerControl,
)


from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel

import numpy as np
import torch


@dataclass
class CountingSplatfactoModelConfig(SplatfactoModelConfig):
    """CountingSplatfacto Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: CountingSplatfactoModel)


class CountingSplatfactoModel(SplatfactoModel):
    """CountingSplatfacto Model."""

    config: CountingSplatfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from nerfstudio.viewer.viewer_elements import (
            ViewerButton,
            ViewerSlider,
            ViewerClick,
            ViewerControl,
        )

        self.click_gaussian = ViewerButton(name="Click", cb_hook=self._click_gaussian)
        self.viewer_control = ViewerControl()

        self.previous_click = None

    def _click_gaussian(self, button: ViewerButton):
        """Start listening for click-based 3D point specification.
        Refer to garfield_interaction.py for more details."""

        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.click_gaussian.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Refer to garfield_interaction.py for more details."""

        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf

        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        # project it into coordinates with matrix
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.eval()
        with torch.no_grad():
            outputs = self.get_outputs(cam.to(self.device))
            print("pix_y, pix_x:", pix_y, pix_x)
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()
        self.train()

        self.click_location = np.array(click.origin) + np.array(click.direction) * (
            depth / z_dir
        )
        print(f"Clicked in: {self.click_location}")

        import trimesh
        from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # type: ignore
        # """
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/click",
            mesh=sphere_mesh,
            position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location,
        )
        # """

        if self.previous_click is not None:
            dist = np.linalg.norm(self.previous_click - self.click_location)
            print("Distance from previous click:", dist)

        self.previous_click = self.click_location
