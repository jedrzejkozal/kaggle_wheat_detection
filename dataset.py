import os
import csv
import pathlib
import torch
import torch.utils.data
import collections
import PIL.Image as Image


class WheatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, transforms=None):
        self.dataset_root = pathlib.Path(dataset_root)
        self.train_filenames = os.listdir(self.dataset_root / 'train')
        self.transforms = transforms

        self.classes = dict()
        self.boxes = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        with open(self.dataset_root / 'train.csv') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if row == 'image_id,width,height,bbox,source\n':
                    continue
                filename, _, _, bbox, class_name = row
                if class_name not in self.classes:  # 0 is background
                    self.classes[class_name] = len(self.classes)+1
                self.boxes[filename + '.jpg'].append(eval(bbox))
                self.labels[filename + '.jpg'].append(self.classes[class_name])

    def __len__(self):
        return len(self.train_filenames)

    def __getitem__(self, index):
        filename = self.train_filenames[index]
        bbox = self.boxes[filename]
        label = self.labels[filename]

        filename = pathlib.Path(filename)
        img = Image.open(self.dataset_root / 'train' / filename)

        if self.transforms is not None:
            img, bbox, label = self.transforms(img, bbox, label)

        return img, bbox, label
