import pytest
import numpy as np
import torch

from mAP import *


def test_mAP_1_img_matching_boxes():
    num_classes = 2  # including background (background is 0)
    model_output = [{
        'boxes': torch.Tensor([[100, 100, 200, 200], [200, 200, 300, 300]]),
        'labels': torch.Tensor([1, 1]),
        'scores': torch.Tensor([1.0, 1.0])
    }]
    targets = [{
        'boxes': torch.Tensor([[100, 100, 200, 200], [200, 200, 300, 300]]),
        'labels': torch.Tensor([1, 1])
    }]
    mAP_calc = mAP(num_classes, thresholds=np.arange(0.5, 0.755, 0.05))

    mAP_calc.add_predictions(model_output, targets)
    value = mAP_calc.get_value()
    assert value == 1.0


def test_mAP_1_img_matching_boxes_one_pred_class_is_different_then_gt():
    num_classes = 3  # including background (background is 0)
    model_output = [{
        'boxes': torch.Tensor([[100, 100, 200, 200], [200, 200, 300, 300]]),
        'labels': torch.Tensor([1, 2]),
        'scores': torch.Tensor([1.0, 1.0])
    }]
    targets = [{
        'boxes': torch.Tensor([[100, 100, 200, 200], [200, 200, 300, 300]]),
        'labels': torch.Tensor([1, 1])
    }]
    mAP_calc = mAP(num_classes, thresholds=np.arange(0.5, 0.755, 0.05))

    mAP_calc.add_predictions(model_output, targets)
    value = mAP_calc.get_value()
    assert value == 0.3333333333333333


def test_nonoverlaping_boxes_mAP_is_0():
    num_classes = 2  # including background (background is 0)
    model_output = [{
        'boxes': torch.Tensor([[300, 300, 400, 400], [400, 400, 500, 500], [500, 500, 600, 600]]),
        'labels': torch.Tensor([1, 1, 1]),
        'scores': torch.Tensor([1.0, 1.0, 1.0])
    }]
    targets = [{
        'boxes': torch.Tensor([[100, 100, 200, 200], [200, 200, 300, 300]]),
        'labels': torch.Tensor([1, 1])
    }]

    mAP_calc = mAP(num_classes, thresholds=np.arange(0.5, 0.755, 0.05))

    mAP_calc.add_predictions(model_output, targets)
    value = mAP_calc.get_value()
    assert value == 0.


def test_2_images_partial_overlaping():
    num_classes = 2  # including background (background is 0)
    model_output = [
        {
            'boxes': torch.Tensor([[100, 100, 200, 200]]),
            'labels': torch.Tensor([1]),
            'scores': torch.Tensor([1.0])
        },
        {
            'boxes': torch.Tensor([[200, 200, 300, 300]]),
            'labels': torch.Tensor([1]),
            'scores': torch.Tensor([1.0])
        },
    ]
    targets = [
        {
            'boxes': torch.Tensor([[149, 100, 200, 200]]),
            'labels': torch.Tensor([1])
        },
        {
            'boxes': torch.Tensor([[200, 249, 300, 300]]),
            'labels': torch.Tensor([1])
        },
    ]

    mAP_calc = mAP(num_classes, thresholds=np.arange(0.5, 0.755, 0.05))

    mAP_calc.add_predictions(model_output, targets)
    value = mAP_calc.get_value()
    assert value == 1 / 6.
