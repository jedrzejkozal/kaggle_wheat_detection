import torch

from dataset import *
from transforms import *


trans = Compose([ToTensor()])
dataset = WheatDataset('./data', transforms=trans)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

for img, bbox, label in dataloader:
    print(img.size)
    print(bbox)
    print(label)
