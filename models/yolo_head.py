"""
YOLO Detection Head for Object Detection

Creates YOLO v2 detection head compatible with Akida 1.0 constraints.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class YOLOHead(layers.Layer):
    """
    YOLO detection head layer.
    
    Outputs detection predictions for each grid cell and anchor box.
    Output shape: (batch, grid_h, grid_w, num_anchors * (4 + 1 + num_classes))
    
    Where:
    - 4: bounding box coordinates (x, y, w, h)
    - 1: objectness score
    - num_classes: class probabilities
    """
    
    def __init__(self, num_anchors, num_classes, **kwargs):
        super(YOLOHead, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.filters = num_anchors * (4 + 1 + num_classes)
    
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            self.filters,
            1,
            strides=1,
            padding='same',
            use_bias=True,
            name=self.name + '_conv'
        )
        super(YOLOHead, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        return x
    
    def get_config(self):
        config = super(YOLOHead, self).get_config()
        config.update({
            'num_anchors': self.num_anchors,
            'num_classes': self.num_classes
        })
        return config


def create_yolo_head(inputs, num_anchors, num_classes, name='yolo_head'):
    """
    Create YOLO detection head.
    
    Args:
        inputs: Input tensor from backbone
        num_anchors: Number of anchor boxes
        num_classes: Number of object classes
        name: Layer name
    
    Returns:
        Detection output tensor
    """
    x = layers.Conv2D(
        num_anchors * (4 + 1 + num_classes),
        1,
        strides=1,
        padding='same',
        name=name + '_conv'
    )(inputs)
    
    return x


class YOLOLoss(tf.keras.losses.Loss):
    """
    YOLO v2 loss function combining:
    - Localization loss (bounding box regression)
    - Objectness loss (confidence prediction)
    - Classification loss (class probabilities)
    """
    
    def __init__(
        self,
        grid_size=(7, 7),
        num_anchors=5,
        num_classes=20,
        lambda_coord=5.0,
        lambda_noobj=0.5,
        **kwargs
    ):
        super(YOLOLoss, self).__init__(**kwargs)
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def call(self, y_true, y_pred):
        """
        Compute YOLO loss.
        
        Args:
            y_true: Ground truth tensor (batch, grid_h, grid_w, anchors * (4+1+classes))
            y_pred: Predicted tensor (same shape)
        
        Returns:
            Total YOLO loss
        """
        # Reshape predictions
        pred_shape = tf.shape(y_pred)
        batch_size = pred_shape[0]
        
        # Parse predictions
        # y_pred shape: (batch, h, w, anchors * (4 + 1 + classes))
        pred_boxes = y_pred[..., :self.num_anchors * 4]
        pred_obj = y_pred[..., self.num_anchors * 4:self.num_anchors * 5]
        pred_cls = y_pred[..., self.num_anchors * 5:]
        
        # Reshape for easier computation
        pred_boxes = tf.reshape(pred_boxes, 
                               [batch_size, self.grid_size[0], self.grid_size[1], 
                                self.num_anchors, 4])
        pred_obj = tf.reshape(pred_obj,
                            [batch_size, self.grid_size[0], self.grid_size[1], 
                             self.num_anchors, 1])
        pred_cls = tf.reshape(pred_cls,
                            [batch_size, self.grid_size[0], self.grid_size[1],
                             self.num_anchors, self.num_classes])
        
        # Compute losses
        # For simplicity, using mean squared error
        # In practice, you'd want to implement the full YOLO loss
        
        # Localization loss
        box_loss = tf.reduce_mean(tf.square(y_true[..., :4] - y_pred[..., :4]))
        
        # Objectness loss
        obj_loss = tf.reduce_mean(tf.square(y_true[..., 4] - y_pred[..., 4]))
        
        # Classification loss
        cls_loss = tf.reduce_mean(tf.square(y_true[..., 5:] - y_pred[..., 5:]))
        
        total_loss = (
            self.lambda_coord * box_loss +
            obj_loss +
            self.lambda_noobj * obj_loss +
            cls_loss
        )
        
        return total_loss


def generate_anchors(feature_size, aspect_ratios=None):
    """
    Generate anchor boxes for YOLO.
    
    Args:
        feature_size: Size of feature map (h, w)
        aspect_ratios: List of aspect ratios (width/height)
    
    Returns:
        Array of anchor boxes (num_anchors, 2) with (width, height)
    """
    if aspect_ratios is None:
        # Default aspect ratios based on common object shapes
        aspect_ratios = [
            [1.0, 1.0],    # Square
            [1.5, 1.0],    # Wide
            [1.0, 1.5],    # Tall
            [2.0, 1.0],    # Very wide
            [1.0, 2.0],    # Very tall
        ]
    
    anchors = []
    for ratio in aspect_ratios:
        # Scale based on feature size
        w = ratio[0]
        h = ratio[1]
        anchors.append([w, h])
    
    return np.array(anchors)


def compute_iou(box1, box2):
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1 = tf.maximum(box1[..., 0], box2[..., 0])
    y1 = tf.maximum(box1[..., 1], box2[..., 1])
    x2 = tf.minimum(box1[..., 2], box2[..., 2])
    y2 = tf.minimum(box1[..., 3], box2[..., 3])
    
    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


if __name__ == '__main__':
    # Test YOLO head
    from models.akidanet import create_akidanet_backbone_v2
    
    # Create backbone
    backbone = create_akidanet_backbone_v2(input_shape=(224, 224, 3), alpha=0.5)
    
    # Create YOLO head
    num_classes = 20
    num_anchors = 5
    
    yolo_output = create_yolo_head(
        backbone.output,
        num_anchors=num_anchors,
        num_classes=num_classes,
        name='detection'
    )
    
    model = keras.Model(inputs=backbone.input, outputs=yolo_output, name='yolo_head_test')
    model.summary()
    
    print(f"\nOutput shape: {yolo_output.shape}")
    print(f"Expected: (None, 7, 7, {num_anchors * (4 + 1 + num_classes)})")
