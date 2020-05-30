import torch
import torch.nn as nn


class RPN(nn.Module):
    """
    Region  Proposal  Network 
    """

    def __init__(self, num_anchores):
        self.num_anchores = num_anchores

        self.intermidiate_layer = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.class_layer = nn.Conv2d(
            512, num_anchores, kernel_size=1, padding=1)
        self.reg_layer = nn.Conv2d(
            512, 4 * num_anchores, kernel_size=1, padding=1)

    def compute_anchors(self):
        pass

    def forward(self, feature_maps):
        """
        Args:
            feature_maps list of 4 elements with shapes:
                torch.Size([1, 256, 256, 256])
                torch.Size([1, 256, 128, 128])
                torch.Size([1, 256, 64, 64])
                torch.Size([1, 256, 32, 32])
        """
        res = []
        for x in feature_maps:
            intermidiate = self.intermidiate_layer(x)
            class_pred = self.class_layer(intermidiate)
            reg_pred = self.reg_layer(intermidiate)
