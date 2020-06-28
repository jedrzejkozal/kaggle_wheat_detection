import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import functools
import time
import argparse
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor

from torch.utils.tensorboard import SummaryWriter

import metrics.mAP
from test_set_predictions import *
from dataset import *


def main():
    args = parse_args()
    writer = SummaryWriter()
    store_hyperparams(args, writer)

    device = torch.device(args.device)
    train_dataset, train_dataset_val, val_dataset, num_classes, mean, std = get_datasets(
        args)
    train_dataloader, train_dataloader_val, val_dataloader = get_dataloaders(
        device, train_dataset, train_dataset_val, val_dataset, args)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False, num_classes=num_classes)
    initilize_weights(model)
    model = model.to(device)

    optimizer = get_optimizer(args, model.parameters())

    train(model, (train_dataloader, train_dataloader_val, val_dataloader),
          optimizer, args.epochs, num_classes, writer, args.test_images_dir)

    torch.save(model, 'faster_rcnn.pt')
    preds = get_test_set_predictions(model, args.test_images_dir, mean, std)
    save_test_set_predictions(preds, 'submission.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster RCNN')

    parser.add_argument('--images-dir', type=str, default='./data/train')
    parser.add_argument('--test-images-dir', type=str, default='./data/test')
    parser.add_argument('--train-csv-path', type=str,
                        default='./data/train.csv')
    parser.add_argument('--val-csv-path', type=str, default='./data/val.csv')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--min-area', type=float, default=0.15)
    parser.add_argument('--min-visibility', type=float, default=0.15)
    parser.add_argument('--use-imagenet-norm-stats',
                        action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=0)

    return parser.parse_args()


def store_hyperparams(args, summary_writer):
    hyperparams = {}
    for arg in vars(args):
        hyperparams[arg] = getattr(args, arg)

    summary_writer.add_hparams(hyperparams, {})
    summary_writer.flush()


def get_datasets(args):
    mean, std = get_norm_stats(args.use_imagenet_norm_stats)
    train_transforms = get_train_transforms(
        args.min_area, args.min_visibility, mean, std)
    val_transforms = get_val_transforms(
        args.min_area, args.min_visibility, mean, std)
    train_dataset = WheatDataset(
        args.images_dir, args.train_csv_path, transforms=train_transforms)
    train_dataset_val = WheatDataset(
        args.images_dir, args.train_csv_path, transforms=val_transforms)
    val_dataset = WheatDataset(
        args.images_dir, args.val_csv_path, transforms=val_transforms)
    num_classes = len(train_dataset.classes) + 1  # include background
    print('train samples = ', len(train_dataset))
    print('val samples = ', len(val_dataset))
    print('num classes = ', num_classes)
    return train_dataset, train_dataset_val, val_dataset, num_classes, mean, std


def get_norm_stats(use_imagenet_norm_stats):
    if use_imagenet_norm_stats:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        return (0.3153, 0.3173, 0.2146), (0.0602, 0.0567, 0.0376)


def get_train_transforms(min_area, min_visibility, mean, std):
    return A.Compose([
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomSizedCrop((700, 700), 1024, 1024, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensor()
    ],
        bbox_params=A.BboxParams(format='coco', min_area=min_area,
                                 min_visibility=min_visibility, label_fields=['labels']))


def get_val_transforms(min_area, min_visibility, mean, std):
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensor()
    ],
        bbox_params=A.BboxParams(format='coco', min_area=min_area,
                                 min_visibility=min_visibility, label_fields=['labels']))


def collate(samples, device=None):
    images = []
    targets = []
    for img, bbox, label in samples:
        if bbox == []:
            continue
        images.append(img.to(device))
        if len(bbox) == 0:
            bbox = [[]]
        bbox = torch.Tensor(bbox)

        bbox = xmin_ymin_width_height_to_xmin_ymin_xmax_ymax(bbox)
        targets.append({
            'boxes': bbox.to(device),
            'labels': torch.Tensor(label).long().to(device)
        })
    return images, targets


def xmin_ymin_width_height_to_xmin_ymin_xmax_ymax(box):
    return torch.cat([box[:, :2], box[:, :2] + box[:, 2:]], dim=1)


def get_dataloaders(device, train_dataset, train_dataset_val, val_dataset, args):
    collate_fn = functools.partial(collate, device=device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    train_dataloader_val = torch.utils.data.DataLoader(
        train_dataset_val, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    return train_dataloader, train_dataloader_val, val_dataloader


def initilize_weights(model):
    def init(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight)
    model.apply(init)


def get_optimizer(args, model_params):
    optimizer_str = args.optimizer.upper()
    if optimizer_str == 'ADAM':
        return optim.Adam(model_params, args.lr)
    elif optimizer_str == 'SGD':
        return optim.SGD(model_params, args.lr, momentum=args.momentum)


def train(model, dataloaders, optimizer, epochs, num_classes, summary_writer, test_images_dir):
    train_dataloader, train_dataloader_val, val_dataloader = dataloaders
    for epoch in range(epochs):
        start = time.time()
        model.train()
        for img, target in train_dataloader:
            output = model(img, target)
            loss = sum(l for l in output.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        duration = time.time() - start
        print("epoch {}/{} duration {}: "
              "loss_classifier={:.4f}, "
              "loss_box_reg={:.4f}, "
              "loss_objectness={:.4f}, "
              "loss_rpn_box_reg={:.4f}"
              .format(epoch + 1,
                      epochs,
                      time.strftime('%M:%S', time.gmtime(duration)),
                      output['loss_classifier'].item(),
                      output['loss_box_reg'].item(),
                      output['loss_objectness'].item(),
                      output['loss_rpn_box_reg'].item()))

        save_metrics(model, train_dataloader_val,
                     val_dataloader, summary_writer, epoch, num_classes)


def save_metrics(model, train_dataloader, val_dataloader, summary_writer, epoch, num_classes):
    model.eval()
    with torch.no_grad():
        mAP = {'train': 0., 'val': 0.}

        mAP_calc = metrics.mAP.mAP(
            num_classes, thresholds=np.arange(0.5, 0.755, 0.05))
        for img, target in train_dataloader:
            output = model(img)
            mAP_calc.add_predictions(output, target)

        mAP['train'] = mAP_calc.get_value()

        for img, target in val_dataloader:
            output = model(img)
            mAP_calc.add_predictions(output, target)

        mAP['val'] = mAP_calc.get_value()
        summary_writer.add_scalars('mAP', mAP, epoch)


if __name__ == '__main__':
    main()
