import torch
import torch.nn as nn

import sys

sys.path.append("ext/dinov2")
from dinov2.eval.linear import create_linear_input
from dinov2.eval.linear import LinearClassifier
from dinov2.eval.utils import ModelWithIntermediateLayers
from functools import partial


class DinoRegression(nn.Module):  # With regression head
    def __init__(self, type, exp_config):
        super().__init__()

        self.config = exp_config
        self.device = exp_config["device"]

        # Load pretrained DINO model
        model = torch.hub.load("facebookresearch/dinov2", type, pretrained=True).to(
            self.device
        )

        autocast_ctx = partial(
            torch.cuda.amp.autocast, enabled=True, dtype=torch.float16
        )

        # Backbone DinoV2 model
        self.feature_model = ModelWithIntermediateLayers(
            model, n_last_blocks=1, autocast_ctx=autocast_ctx
        ).to(self.device)

        with torch.no_grad():
            sample_input = torch.randn(1, 3, 14 * 32, 14 * 32).to(
                self.device
            )  # TODO use the config not hardcoded 14 32
            sample_output = self.feature_model(sample_input)

        # Get output dimension and define a regression layer
        out_dim = create_linear_input(
            sample_output, use_n_blocks=1, use_avgpool=True
        ).shape[1]

        use_simple_linear = False
        if use_simple_linear:
            out_dim = 256 * 768

            # Fully connected layer for regression output
            self.regressor = nn.Linear(out_dim, 1).to(self.device)
            torch.nn.init.xavier_uniform_(self.regressor.weight)
        else:
            layers = [
                nn.Conv2d(
                    self.config["MODEL_CONFIG"]["feats_dim"],
                    512,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),  # (8, 8, 512)
                nn.ReLU(),
                nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # (4, 4, 256)
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # (2, 2, 128)
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=2),  # (1, 1, 64)
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            ]

            layers.append(nn.Linear(64, 1))
            layers.append(nn.Sigmoid())

            self.regressor = nn.Sequential(*layers).to(self.device)

        self.print_model_size()

    def forward(self, x, volume=None):
        with torch.no_grad():
            features = self.feature_model(x)
            # First indexing on the layer output (forward can return several of them in dinov2),
            # then taking first tensor and discarding the second, which is some classification tokens
            # features: tuple of length 1
            # features[0]: tuple of length 2
            # features[0][0]: tensor of shape [B, 1024, 768]
            # features[0][1]: tensor of shape [B, 768]

        feats = features[0][0].clone()

        feats = feats.permute(0, 2, 1)  # B, H*W, D -> B, D, H*W
        feats = feats.view(
            (
                -1,
                self.config["MODEL_CONFIG"]["feats_dim"],
                self.config["MODEL_CONFIG"]["encoded_image_size"],
                self.config["MODEL_CONFIG"]["encoded_image_size"],
            )
        )
        return self.regressor(feats)

    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of weights: {total_params}")
        regressor_params = sum(p.numel() for p in self.regressor.parameters())
        print(f"regressor number of weights: {regressor_params}")
