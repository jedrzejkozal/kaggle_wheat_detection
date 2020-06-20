import torch
import os
import pathlib
import cv2
import csv
import albumentations as A
from albumentations.pytorch import ToTensor


def get_test_set_predictions(model, images_dir, mean, std):
    filenames = os.listdir(images_dir)
    root = pathlib.Path(images_dir)
    transforms = A.Compose([A.Normalize(mean=mean, std=std), ToTensor()])
    model.eval()
    results = {}

    for filename in filenames:
        filepath = root / filename
        img = cv2.imread(str(filepath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augumented = transforms(image=img)
        output = model([augumented['image']])
        results[filename] = output
    return results


def save_test_set_predictions(predictions, filename):
    with open(filename, 'w+') as f:
        csv_writer = csv.writer(f)
        for filename, img_pred in predictions.items():
            pred_str = ""
            for pred in img_pred:
                confs, bboxes = pred['scores'], pred['boxes']
                for conf, bbox in zip(confs, bboxes):
                    bbox = [int(b) for b in bbox]
                    pred_str += " {} {} {} {} {}".format(conf, *bbox)
            pred_str = pred_str[1:]
            csv_writer.writerow((filename, pred_str))
