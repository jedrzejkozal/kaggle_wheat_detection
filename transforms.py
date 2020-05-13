import torch
import torchvision.transforms.functional as F


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, img, bbox, label):
        img = F.to_tensor(img)
        bbox = torch.Tensor(bbox)
        label = torch.Tensor(label)
        return img, bbox, label


class Compose:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, img, bbox, label):
        for t in self.transforms_list:
            img, bbox, label = t(img, bbox, label)
        return img, bbox, label
