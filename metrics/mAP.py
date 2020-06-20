import torch
import numpy as np
import collections

# from metrics.iou import iou
from iou import iou


class mAP:
    def __init__(self, num_classes, thresholds):
        self.classes = np.arange(1, num_classes)
        self.thresholds = thresholds
        self.num_thresholds = float(len(thresholds))
        self.AP_sum = 0.0
        self.num_imgs = 0

    def add_predictions(self, predictions, targets):
        for pred, target in zip(predictions, targets):
            pred_boxes, pred_lables = pred['boxes'], pred['labels']
            # if len(pred_boxes) == 0:
            #     # TODO add all gt as FN
            #     print('no preds')
            #     continue
            target_boxes, target_labels = target['boxes'], target['labels']

            if len(target_boxes) == 0 and len(pred_boxes) > 0:
                # mAP for this image is 0 according to kaggle evaluation tab
                self.num_imgs += 1

            AP_for_thresholds = []
            counts = collections.defaultdict(lambda: (0, 0, 0))
            for c in self.classes:
                class_pred_boxes = pred_boxes[pred_lables == c]
                class_target_boxes = target_boxes[target_labels == c]
                if len(class_pred_boxes) == 0:
                    FN = len(class_target_boxes)
                    for t in self.thresholds:
                        old_TP, old_FP, old_FN = counts[t]
                        counts[t] = (old_TP, old_FP, old_FN + FN)
                    continue
                if len(class_target_boxes) == 0:
                    FP = len(class_pred_boxes)
                    for t in self.thresholds:
                        old_TP, old_FP, old_FN = counts[t]
                        counts[t] = (old_TP, old_FP + FP, old_FN)
                    continue

                self.update_matching_counts(
                    class_pred_boxes, class_target_boxes, counts)

            AP_for_thresholds = [TP / float(TP + FP + FN)
                                 for TP, FP, FN in counts.values()]
            self.AP_sum += sum(AP_for_thresholds) / self.num_thresholds
            self.num_imgs += 1

    def update_matching_counts(self, pred_boxes, target_boxes, counts):
        boxes_iou = iou(pred_boxes, target_boxes)
        pred_max_iou, _ = torch.max(boxes_iou, dim=1)
        gt_max_iou, _ = torch.max(boxes_iou, dim=0)

        for threshold in self.thresholds:
            matching_preds = pred_max_iou > threshold
            TP = sum(matching_preds)
            FP = len(matching_preds) - TP
            matching_gt = gt_max_iou > threshold
            FN = len(matching_gt) - sum(matching_gt)

            old_TP, old_FP, old_FN = counts[threshold]
            counts[threshold] = (old_TP + TP, old_FP + FP, old_FN + FN)

    def get_value(self):
        mAP_value = self.AP_sum / self.num_imgs
        self.AP_sum = 0.0
        self.num_imgs = 0
        return mAP_value
