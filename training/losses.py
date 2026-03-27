"""
Loss Functions for YOLO Object Detection

Implements YOLO v2 loss with box regression, objectness, and classification.
"""

import tensorflow as tf


class YOLOLossV2(tf.keras.losses.Loss):
    """
    Complete YOLO v2 loss implementation.
    
    Combines:
    - Localization loss (bounding box coordinates)
    - Objectness loss (confidence of object presence)
    - Classification loss (class probabilities)
    """
    
    def __init__(
        self,
        grid_size=(7, 7),
        num_classes=20,
        num_anchors=5,
        anchor_boxes=None,
        lambda_coord=5.0,
        lambda_noobj=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        # Default anchors if not provided
        if anchor_boxes is None:
            self.anchor_boxes = tf.constant([
                [1.12, 1.85],
                [2.00, 2.83],
                [3.19, 3.60],
                [4.55, 5.26],
                [5.49, 5.82]
            ], dtype=tf.float32)
        else:
            self.anchor_boxes = tf.constant(anchor_boxes, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        """
        Compute YOLO loss.
        
        Args:
            y_true: Ground truth (batch, h, w, anchors*(4+1+classes))
            y_pred: Predictions (same shape)
        
        Returns:
            Total loss
        """
        batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)
        
        # Parse predictions
        # predictions: [bx, by, bw, bh, objectness, class_probs]
        pred_boxes = y_pred[..., :self.num_anchors * 4]
        pred_obj = y_pred[..., self.num_anchors * 4:self.num_anchors * 5]
        pred_cls = y_pred[..., self.num_anchors * 5:]
        
        # Parse ground truth
        true_boxes = y_true[..., :self.num_anchors * 4]
        true_obj = y_true[..., self.num_anchors * 4:self.num_anchors * 5]
        true_cls = y_true[..., self.num_anchors * 5:]
        
        # Reshape for computation
        pred_boxes = tf.reshape(pred_boxes, 
            [-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 4])
        pred_obj = tf.reshape(pred_obj,
            [-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 1])
        pred_cls = tf.reshape(pred_cls,
            [-1, self.grid_size[0], self.grid_size[1], self.num_anchors, self.num_classes])
        
        true_boxes = tf.reshape(true_boxes,
            [-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 4])
        true_obj = tf.reshape(true_obj,
            [-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 1])
        true_cls = tf.reshape(true_cls,
            [-1, self.grid_size[0], self.grid_size[1], self.num_anchors, self.num_classes])
        
        # Compute masks
        obj_mask = tf.cast(true_obj[..., 0] > 0.5, tf.float32)
        noobj_mask = 1.0 - obj_mask
        
        # Localization loss (only for objects)
        box_loss = tf.reduce_sum(
            obj_mask * tf.reduce_sum(tf.square(true_boxes - pred_boxes), axis=-1)
        )
        
        # Objectness loss
        obj_loss = tf.reduce_sum(
            obj_mask * tf.square(true_obj - tf.sigmoid(pred_obj))
        )
        noobj_loss = tf.reduce_sum(
            noobj_mask * tf.square(true_obj - tf.sigmoid(pred_obj))
        )
        
        # Classification loss (only for objects)
        cls_loss = tf.reduce_sum(
            obj_mask * tf.reduce_sum(tf.square(true_cls - tf.sigmoid(pred_cls)), axis=-1)
        )
        
        # Total loss
        total_loss = (
            self.lambda_coord * box_loss +
            obj_loss +
            self.lambda_noobj * noobj_loss +
            cls_loss
        ) / batch_size
        
        return total_loss


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for handling class imbalance.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        focal_loss = alpha_t * focal_weight * bce
        
        return tf.reduce_mean(focal_loss)


class IoULoss(tf.keras.losses.Loss):
    """
    IoU-based loss for bounding box regression.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        # Convert to corner format
        # Assuming y_true/y_pred are in center format [cx, cy, w, h]
        true_x1 = y_true[..., 0] - y_true[..., 2] / 2
        true_y1 = y_true[..., 1] - y_true[..., 3] / 2
        true_x2 = y_true[..., 0] + y_true[..., 2] / 2
        true_y2 = y_true[..., 1] + y_true[..., 3] / 2
        
        pred_x1 = y_pred[..., 0] - y_pred[..., 2] / 2
        pred_y1 = y_pred[..., 1] - y_pred[..., 3] / 2
        pred_x2 = y_pred[..., 0] + y_pred[..., 2] / 2
        pred_y2 = y_pred[..., 1] + y_pred[..., 3] / 2
        
        # Compute intersection
        inter_x1 = tf.maximum(true_x1, pred_x1)
        inter_y1 = tf.maximum(true_y1, pred_y1)
        inter_x2 = tf.minimum(true_x2, pred_x2)
        inter_y2 = tf.minimum(true_y2, pred_y2)
        
        inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)
        
        # Compute union
        true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        union_area = true_area + pred_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        # IoU loss (1 - IoU)
        loss = 1.0 - iou
        
        return tf.reduce_mean(loss)


def yolo_loss(y_true, y_pred, grid_size=(7, 7), num_classes=20, 
              num_anchors=5, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Complete YOLO loss function.
    
    Args:
        y_true: Ground truth
        y_pred: Model predictions
        grid_size: YOLO grid dimensions
        num_classes: Number of classes
        num_anchors: Number of anchors
        lambda_coord: Weight for coordinate loss
        lambda_noobj: Weight for no-object loss
    
    Returns:
        Total YOLO loss
    """
    loss = YOLOLossV2(
        grid_size=grid_size,
        num_classes=num_classes,
        num_anchors=num_anchors,
        lambda_coord=lambda_coord,
        lambda_noobj=lambda_noobj
    )
    return loss(y_true, y_pred)


if __name__ == '__main__':
    # Test loss function
    batch_size = 4
    grid_h, grid_w = 7, 7
    num_classes = 20
    num_anchors = 5
    
    y_true = tf.random.normal((batch_size, grid_h, grid_w, num_anchors * (4 + 1 + num_classes)))
    y_pred = tf.random.normal((batch_size, grid_h, grid_w, num_anchors * (4 + 1 + num_classes)))
    
    yolo_loss_fn = YOLOLossV2(
        grid_size=(grid_h, grid_w),
        num_classes=num_classes,
        num_anchors=num_anchors
    )
    
    loss_value = yolo_loss_fn(y_true, y_pred)
    print(f"YOLO Loss: {loss_value.numpy():.4f}")
    
    iou_loss_fn = IoULoss()
    iou_loss = iou_loss_fn(y_true[..., :4], y_pred[..., :4])
    print(f"IoU Loss: {iou_loss.numpy():.4f}")
