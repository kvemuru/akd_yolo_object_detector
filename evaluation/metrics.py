"""
Evaluation Metrics for Object Detection

Implements mAP and other detection metrics.
"""

import tensorflow as tf
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def compute_ap(recalls, precisions):
    """
    Compute Average Precision from recall-precision curve.
    
    Args:
        recalls: List of recall values
        precisions: List of precision values
    
    Returns:
        Average precision value
    """
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Compute AP
    indices = recalls[1:] > recalls[:-1]
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    
    return ap


def compute_map(predictions, ground_truth, iou_threshold=0.5, num_classes=20):
    """
    Compute mean Average Precision.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        ground_truth: List of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
    
    Returns:
        mAP value and per-class AP
    """
    aps = []
    
    for class_id in range(num_classes):
        # Get predictions and ground truth for this class
        class_predictions = []
        class_ground_truth = []
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            # Filter predictions for this class
            pred_mask = pred['labels'] == class_id
            class_predictions.append({
                'image_id': img_idx,
                'boxes': pred['boxes'][pred_mask],
                'scores': pred['scores'][pred_mask]
            })
            
            # Filter ground truth for this class
            gt_mask = gt['labels'] == class_id
            class_ground_truth.append({
                'image_id': img_idx,
                'boxes': gt['boxes'][gt_mask]
            })
        
        # Compute AP for this class
        ap = compute_class_ap(class_predictions, class_ground_truth, iou_threshold)
        aps.append(ap)
    
    # Compute mAP
    valid_aps = [ap for ap in aps if ap is not None]
    mAP = np.mean(valid_aps) if valid_aps else 0.0
    
    return mAP, aps


def compute_class_ap(predictions, ground_truth, iou_threshold):
    """
    Compute AP for a single class.
    
    Args:
        predictions: List of predictions for the class
        ground_truth: List of ground truth for the class
    
    Returns:
        AP value or None if no predictions/ground truth
    """
    # Collect all predictions with their image and confidence
    all_predictions = []
    for img_pred in predictions:
        for box, score in zip(img_pred['boxes'], img_pred['scores']):
            all_predictions.append({
                'image_id': img_pred['image_id'],
                'box': box,
                'score': score
            })
    
    if not all_predictions:
        return None
    
    # Sort by confidence
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Count ground truth boxes per image
    gt_counts = {}
    for img_gt in ground_truth:
        gt_counts[img_gt['image_id']] = len(img_gt['boxes'])
    
    total_gt = sum(gt_counts.values())
    if total_gt == 0:
        return None
    
    # Track which ground truth boxes have been matched
    matched_gt = defaultdict(set)
    
    tp = []
    fp = []
    
    for pred in all_predictions:
        img_id = pred['image_id']
        pred_box = pred['box']
        
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(ground_truth[img_id]['boxes']):
            if gt_idx in matched_gt[img_id]:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched_gt[img_id].add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    return compute_ap(recalls.tolist(), precisions.tolist())


class DetectionMetrics:
    """Object detection metrics tracker."""
    
    def __init__(self, num_classes=20, iou_thresholds=[0.5]):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.ground_truth = []
    
    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """
        Update metrics with new predictions.
        
        Args:
            pred_boxes: Predicted boxes (N, 4)
            pred_scores: Prediction scores (N,)
            pred_labels: Prediction labels (N,)
            gt_boxes: Ground truth boxes (M, 4)
            gt_labels: Ground truth labels (M,)
        """
        self.predictions.append({
            'boxes': np.array(pred_boxes),
            'scores': np.array(pred_scores),
            'labels': np.array(pred_labels)
        })
        
        self.ground_truth.append({
            'boxes': np.array(gt_boxes),
            'labels': np.array(gt_labels)
        })
    
    def compute(self):
        """Compute all metrics."""
        results = {}
        
        for iou_thresh in self.iou_thresholds:
            mAP, class_aps = compute_map(
                self.predictions,
                self.ground_truth,
                iou_threshold=iou_thresh,
                num_classes=self.num_classes
            )
            results[f'mAP@{iou_thresh}'] = mAP
            results[f'AP_per_class@{iou_thresh}'] = class_aps
        
        return results
    
    def get_summary(self):
        """Get formatted summary string."""
        results = self.compute()
        
        summary = "Detection Metrics Summary\n"
        summary += "=" * 40 + "\n"
        
        for metric, value in results.items():
            if not metric.startswith('AP_per_class'):
                summary += f"{metric}: {value:.4f}\n"
        
        return summary


if __name__ == '__main__':
    print("=" * 50)
    print("Object Detection Metrics")
    print("=" * 50)
    
    # Test IoU computation
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = compute_iou(box1, box2)
    print(f"IoU test: {iou:.4f} (expected ~0.14)")
    
    # Test AP computation
    recalls = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    precisions = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    ap = compute_ap(recalls, precisions)
    print(f"AP test: {ap:.4f}")
