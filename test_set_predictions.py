import torch
import os
import pathlib
import cv2
import csv
import albumentations as A
from albumentations.pytorch import ToTensor
from matplotlib import pyplot as plt


def get_test_set_predictions(model, images_dir, mean, std, device):
    filenames = os.listdir(images_dir)
    root = pathlib.Path(images_dir)
    transforms = A.Compose([A.Normalize(mean=mean, std=std), ToTensor()])
    model.eval()
    results = {}

    with torch.no_grad():
        for filename in filenames:
            filepath = root / filename
            img = cv2.imread(str(filepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augumented = transforms(image=img)
            output = model([augumented['image'].to(device)])
            results[filename] = output
    return results


def save_test_set_predictions(predictions, csv_filename):
    with open(csv_filename, 'w+') as f:
        csv_writer = csv.writer(f)
        for filename, img_pred in predictions.items():
            pred_str = ""
            for pred in img_pred:
                confs, bboxes = pred['scores'], pred['boxes']
                for conf, bbox in zip(confs, bboxes):
                    bbox = [int(b) for b in bbox]
                    pred_str += " {} {} {} {} {}".format(conf, *bbox)
            pred_str = pred_str[1:]
            filename, _ = filename.split('.')
            csv_writer.writerow((filename, pred_str))


def show_test_set_predictions(filename, images_dir):
    images_dir = pathlib.Path(images_dir)

    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        for filename, predictions in csv_reader:
            filename += '.jpg'
            filepath = str(images_dir / filename)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for (x_min, y_min, x_max, y_max) in predictions_iterator(predictions):
                img = add_bbox(
                    img, (x_min, y_min, x_max - x_min, y_max - y_min))
            plt.figure(figsize=(12, 12))
            plt.imshow(img)
            plt.show()


def predictions_iterator(predictions):
    pred_iter = iter(predictions.split(' '))

    while True:
        _ = next(pred_iter)  # prediction confidence
        x_min = int(next(pred_iter))
        y_min = int(next(pred_iter))
        x_max = int(next(pred_iter))
        y_max = int(next(pred_iter))
        yield (x_min, y_min, x_max, y_max)


def add_bbox(img, bbox, color=(255, 0, 0), thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(
        x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)
    return img
