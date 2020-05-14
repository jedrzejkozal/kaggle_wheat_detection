import torch
import torchvision.models as models

models.resnet101(pretrained=True)


class Resnet101(models.resnet.ResNet):
    def __init__(self, pretrained=False):
        super().__init__(models.resnet.Bottleneck, [3, 4, 23, 3])

        if pretrained:
            state_dict = models.utils.load_state_dict_from_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                                                               progress=False)
            super().load_state_dict(state_dict)

    def forward(self, x):
        outputs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)

        return outputs
