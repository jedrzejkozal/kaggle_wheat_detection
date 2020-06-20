import pytest
import torch
import iou


def test_nonoverlaping_boxes():
    box_a = torch.tensor([[10.0, 20.0, 40.0, 30.0]])
    box_b = torch.tensor([[100.0, 120.0, 110.0, 140.0]])

    assert iou.iou(box_a, box_b) == 0.0


def test_half_overlaping_boxes():
    box_a = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    box_b = torch.tensor([[10.0, 20.0, 20.0, 40.0]])

    assert iou.iou(box_a, box_b) == 0.5


def test_max_overlaping_boxes():
    box_a = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    box_b = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    assert iou.iou(box_a, box_b) == 1.0


def test_multiple_boxes():
    box_a = torch.tensor([[10.0, 20.0, 40.0, 30.0], [20.0, 30.0, 40.0, 40.0], [
                         100.0, 120.0, 110.0, 140.0]])
    box_b = torch.tensor(
        [[10.0, 20.0, 40.0, 30.0], [20.0, 30.0, 40.0, 40.0]])

    assert (iou.iou(box_a, box_b) == torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])).all()
