import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        in_channels = [256, 512, 1024, 2048]
        self.latteral_convs = [
            nn.Conv2d(c, channels, kernel_size=1) for c in in_channels]
        self.final_convs = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in in_channels]

    def forward(self, x):
        output = []
        for tensor, conv in zip(x, self.latteral_convs):
            output.append(conv(tensor))

        for i in reversed(range(len(self.latteral_convs)-1)):
            output[i] = output[i] + F.interpolate(output[i+1], scale_factor=2)

        for i in range(len(output)):
            output[i] = self.final_convs[i](output[i])

        return output
