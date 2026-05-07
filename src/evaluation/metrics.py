import numpy as np


def compute_iou(pred, gt):
    # squeeze remove dimensões singleton (alguns PNGs vêm como (H, W, 1)).
    pred = np.squeeze(pred) > 0
    gt = np.squeeze(gt) > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 0.0

    return intersection / union


def compute_mean_iou(preds, gts):
    ious = [compute_iou(p, g) for p, g in zip(preds, gts)]
    return float(np.mean(ious))
