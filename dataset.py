import os
import csv
import pathlib
import torch
import torch.utils.data
import collections
import PIL.Image as Image


class WheatDataset(torch.utils.data.Dataset):
    def __init__(self, files_path, csv_path, transforms=None):
        self.files_path = pathlib.Path(files_path)
        csv_path = pathlib.Path(csv_path)
        self.train_filenames = []
        self.transforms = transforms

        self.classes = dict()
        self.boxes = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        with open(csv_path) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if row[0] == 'image_id':
                    continue
                filename, _, _, bbox, class_name = row
                filename += '.jpg'
                if len(self.train_filenames) == 0 or self.train_filenames[-1] != filename:
                    self.train_filenames.append(filename)
                if class_name not in self.classes:  # 0 is background
                    self.classes[class_name] = len(self.classes) + 1
                self.boxes[filename].append(eval(bbox))
                self.labels[filename].append(self.classes[class_name])

    def __len__(self):
        return len(self.train_filenames)

    def __getitem__(self, index):
        filename = self.train_filenames[index]
        bbox = self.boxes[filename]
        label = self.labels[filename]

        filename = pathlib.Path(filename)
        img = Image.open(self.files_path / filename)

        if self.transforms is not None:
            img, bbox, label = self.transforms(img, bbox, label)

        return img, bbox, label
