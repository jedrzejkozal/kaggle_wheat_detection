import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import functools
import argparse

from torch.utils.tensorboard import SummaryWriter

from dataset import *
from transforms import *


def main():
    args = parse_args()
    writer = SummaryWriter()
    store_hyperparams(args, writer)

    train(args, writer)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster RCNN')

    parser.add_argument('--images-dir', type=str, default='./data/train')
    parser.add_argument('--train-csv-path', type=str,
                        default='./data/train.csv')
    parser.add_argument('--val-csv-path', type=str, default='./data/val.csv')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--device', type=str, default="cuda")

    return parser.parse_args()


def store_hyperparams(args, summary_writer):
    hyperparams = {}
    for arg in vars(args):
        hyperparams[arg] = getattr(args, arg)

    summary_writer.add_hparams(hyperparams, {})
    summary_writer.flush()


def get_optimizer(args, model_params):
    optimizer_str = args.optimizer.upper()
    if optimizer_str == 'ADAM':
        return optim.Adam(model_params, args.lr)
    elif optimizer_str == 'SGD':
        return optim.SGD(model_params, args.lr, momentum=args.momentum)


def train(args, summary_writer):
    trans = Compose([ToTensor()])
    train_dataset = WheatDataset(
        args.images_dir, args.train_csv_path, transforms=trans)
    val_dataset = WheatDataset(
        args.images_dir, args.val_csv_path, transforms=trans)
    num_classes = len(train_dataset.classes) + 1  # include background
    print('train samples = ', len(train_dataset))
    print('val samples = ', len(val_dataset))
    print('num classes = ', num_classes)

    device = torch.device(args.device)
    collate_fn = functools.partial(collate, device=device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False, num_classes=num_classes)
    model = model.to(device)

    optimizer = get_optimizer(args, model.parameters())

    for epoch in range(args.epochs):
        model.train()
        for img, target in train_dataloader:
            optimizer.zero_grad()

            output = model(img, target)
            loss = output['loss_classifier'] + \
                output['loss_box_reg'] + output['loss_objectness'] + \
                output['loss_rpn_box_reg']
            loss.backward()
            print(
                f"epoch {epoch+1}/{args.epochs}: loss_classifier={output['loss_classifier'].item()}, loss_box_reg={output['loss_box_reg'].item()}, loss_objectness={output['loss_objectness'].item()}, loss_rpn_box_reg={output['loss_rpn_box_reg'].item()}")
            optimizer.step()

            break

        with torch.no_grad():
            loss_values = {'train': 0., 'val': 0.}
            loss_classifier = {'train': 0., 'val': 0.}
            loss_box_reg = {'train': 0., 'val': 0.}
            loss_objectness = {'train': 0., 'val': 0.}
            loss_rpn_box_reg = {'train': 0., 'val': 0.}
            for img, target in train_dataloader:
                output = model(img, target)
                loss_value = output['loss_classifier'] + \
                    output['loss_box_reg'] + output['loss_objectness'] + \
                    output['loss_rpn_box_reg']
                loss_values['train'] += loss_value
                loss_classifier['train'] += output['loss_classifier']
                loss_box_reg['train'] += output['loss_box_reg']
                loss_objectness['train'] += output['loss_objectness']
                loss_rpn_box_reg['train'] += output['loss_rpn_box_reg']
                break

            for img, target in val_dataloader:
                output = model(img, target)
                loss_value = output['loss_classifier'] + \
                    output['loss_box_reg'] + output['loss_objectness'] + \
                    output['loss_rpn_box_reg']
                loss_values['val'] += loss_value
                loss_classifier['train'] += output['loss_classifier']
                loss_box_reg['train'] += output['loss_box_reg']
                loss_objectness['train'] += output['loss_objectness']
                loss_rpn_box_reg['train'] += output['loss_rpn_box_reg']
                break
            summary_writer.add_scalars('Loss', loss_values, epoch)
            summary_writer.add_scalars(
                'Loss classifier', loss_classifier, epoch)
            summary_writer.add_scalars('Loss box reg', loss_box_reg, epoch)
            summary_writer.add_scalars(
                'Loss objectness', loss_objectness, epoch)
            summary_writer.add_scalars(
                'Loss rpn box reg', loss_rpn_box_reg, epoch)


def collate(samples, device=None):
    images = []
    targets = []
    for img, bbox, label in samples:
        images.append(img.to(device))
        bbox = xmin_ymin_width_height_to_xmin_ymin_xmax_ymax(bbox)
        targets.append({
            'boxes': bbox.to(device),
            'labels': label.long().to(device)
        })
    return images, targets


def xmin_ymin_width_height_to_xmin_ymin_xmax_ymax(box):
    if box.ndim < 2:
        return box
    return torch.cat([box[:, :2], box[:, :2] + box[:, 2:]], dim=1)


if __name__ == '__main__':
    main()
