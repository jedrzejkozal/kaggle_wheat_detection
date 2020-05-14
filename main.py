import torch

from dataset import *
from transforms import *
import nets.backbones.resnet101
import nets.feature_pyramid


trans = Compose([ToTensor()])
dataset = WheatDataset('./data', transforms=trans)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

backbone = nets.backbones.resnet101.Resnet101(pretrained=True)
pyramid = nets.feature_pyramid.FeaturePyramidNetwork()

for img, bbox, label in dataloader:
    out = backbone(img)
    for o in out:
        print(o.shape)
    print()
    out = pyramid(out)
    for o in out:
        print(o.shape)
    print()

    break
