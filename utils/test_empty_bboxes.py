from dataset import *
from torchvision_training import get_norm_stats, get_train_transforms


mean, std = get_norm_stats(False)
train_transforms = get_train_transforms(0.15, 0.15, mean, std)

train_dataset = WheatDataset(
    './data/train', './data/train.csv', transforms=train_transforms)

for i in range(len(train_dataset)):
    _, bbox, label = train_dataset[i]
    if bbox == []:
        print(i)
        break
