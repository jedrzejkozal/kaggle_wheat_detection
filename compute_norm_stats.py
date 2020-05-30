import torch

from dataset import *
from transforms import *


def main():
    trans = Compose([ToTensor()])
    dataset = WheatDataset('./data', transforms=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    sum = 0.0
    count = 0.0
    for img, _, _ in dataloader:
        sum += img.sum((2, 3))
        count += img[0, 0].numel()
    mean = sum / count
    print('mean = ', mean)

    sum = 0.0
    count = 0.0
    for img, _, _ in dataloader:
        img = (img - mean.unsqueeze(2).unsqueeze(3))**2
        sum += img.sum((2, 3))
        count += img[0, 0].numel()
    std = sum / count
    print('std = ', std)

    # results:
    # mean = torch.tensor([[0.3153, 0.3173, 0.2146]])
    # std = tensor([[0.0602, 0.0567, 0.0376]])


if __name__ == '__main__':
    main()
