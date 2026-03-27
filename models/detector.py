"""
Combined Object Detector Model

Integrates AkidaNet backbone with YOLO detection head.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from models.akidanet import create_akidanet_backbone_v2
from models.yolo_head import create_yolo_head


class YOLODetector(keras.Model):
    """
    YOLO-based object detector compatible with Akida 1.0.
    
    Architecture:
    - AkidaNet backbone (SeparableConvs)
    - YOLO detection head
    - 8-bit quantization ready
    """
    
    def __init__(
        self,
        input_shape=(224, 224, 3),
        num_classes=20,
        num_anchors=5,
        alpha=0.5,
        grid_size=(7, 7),
        anchors=None,
        **kwargs
    ):
        super(YOLODetector, self).__init__(**kwargs)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.alpha = alpha
        self.grid_size = grid_size
        self.anchors = anchors
        
        # Default anchors if not provided
        if self.anchors is None:
            self.anchors = np.array([
                [1.12, 1.85],
                [2.00, 2.83],
                [3.19, 3.60],
                [4.55, 5.26],
                [5.49, 5.82]
            ])
        
        # Build model
        self.backbone = create_akidanet_backbone_v2(input_shape, alpha)
        self.detection_head = self._build_detection_head()
        
        # Register call for proper tracking
        self.built = True
    
    def _build_detection_head(self):
        """Build YOLO detection head layers."""
        outputs = create_yolo_head(
            self.backbone.output,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes,
            name='yolo_head'
        )
        return outputs
    
    def call(self, inputs, training=None, mask=None):
        """
        Forward pass.
        
        Args:
            inputs: Input images (batch, h, w, 3)
            training: Training mode flag
        
        Returns:
            Detection output (batch, h, w, anchors*(4+1+classes))
        """
        backbone_features = self.backbone(inputs, training=training)
        detections = self.detection_head(backbone_features, training=training)
        return detections
    
    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'num_anchors': self.num_anchors,
            'alpha': self.alpha,
            'grid_size': self.grid_size,
            'anchors': self.anchors.tolist() if self.anchors is not None else None
        }
    
    @classmethod
    def from_config(cls, config):
        config['anchors'] = np.array(config['anchors']) if config['anchors'] else None
        return cls(**config)


def create_detector(
    input_shape=(224, 224, 3),
    num_classes=20,
    num_anchors=5,
    alpha=0.5,
    weights_path=None
):
    """
    Create YOLO detector model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of object classes
        num_anchors: Number of anchor boxes
        alpha: Network width multiplier
        weights_path: Optional path to pretrained weights
    
    Returns:
        Compiled YOLO detector model
    """
    # Create backbone
    backbone = create_akidanet_backbone_v2(input_shape, alpha)
    
    # Create YOLO head
    yolo_output = create_yolo_head(
        backbone.output,
        num_anchors=num_anchors,
        num_classes=num_classes,
        name='detection'
    )
    
    # Create model
    model = keras.Model(
        inputs=backbone.input,
        outputs=yolo_output,
        name='akida_yolo_detector'
    )
    
    # Load weights if provided
    if weights_path:
        model.load_weights(weights_path)
    
    return model


def decode_predictions(y_pred, anchors, num_classes, image_shape=(224, 224), 
                      confidence_threshold=0.5, iou_threshold=0.5):
    """
    Decode YOLO predictions to bounding boxes.
    
    Args:
        y_pred: Model output (batch, grid_h, grid_w, anchors*(4+1+classes))
        anchors: Anchor boxes (num_anchors, 2)
        num_classes: Number of classes
        image_shape: Original image shape
        confidence_threshold: Minimum confidence to keep
        iou_threshold: IoU threshold for NMS
    
    Returns:
        List of detections per image, each with boxes, scores, classes
    """
    batch_size = tf.shape(y_pred)[0]
    grid_h, grid_w = tf.shape(y_pred)[1], tf.shape(y_pred)[2]
    
    # Parse predictions
    # Shape: (batch, h, w, anchors, 4+1+classes)
    pred_boxes = y_pred[..., :4]  # (batch, h, w, anchors, 4)
    pred_obj = y_pred[..., 4:5]  # (batch, h, w, anchors, 1)
    pred_cls = y_pred[..., 5:]   # (batch, h, w, anchors, classes)
    
    # Get class predictions
    pred_class = tf.argmax(pred_cls, axis=-1)  # (batch, h, w, anchors)
    pred_class_prob = tf.reduce_max(pred_cls, axis=-1)  # (batch, h, w, anchors)
    
    # Compute final scores
    scores = pred_obj * pred_class_prob  # (batch, h, w, anchors)
    
    # Filter by confidence
    mask = scores > confidence_threshold
    boxes_list = []
    scores_list = []
    classes_list = []
    
    for i in range(batch_size):
        mask_i = mask[i]
        boxes = pred_boxes[i][mask_i]
        scores_i = scores[i][mask_i]
        classes = pred_class[i][mask_i]
        
        if tf.shape(boxes)[0] == 0:
            boxes_list.append(tf.zeros((0, 4)))
            scores_list.append(tf.zeros((0,)))
            classes_list.append(tf.zeros((0,), dtype=tf.int32))
            continue
        
        # Apply NMS
        selected_indices = tf.image.non_max_suppression(
            boxes, scores_i,
            max_output_size=100,
            iou_threshold=iou_threshold
        )
        
        boxes_filtered = tf.gather(boxes, selected_indices)
        scores_filtered = tf.gather(scores_i, selected_indices)
        classes_filtered = tf.gather(classes, selected_indices)
        
        boxes_list.append(boxes_filtered)
        scores_list.append(scores_filtered)
        classes_list.append(classes_filtered)
    
    return boxes_list, scores_list, classes_list


def count_model_params(model):
    """Count model parameters."""
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    return trainable, non_trainable, trainable + non_trainable


if __name__ == '__main__':
    # Create and test detector
    model = create_detector(
        input_shape=(224, 224, 3),
        num_classes=20,
        num_anchors=5,
        alpha=0.5
    )
    
    model.summary()
    
    trainable, non_trainable, total = count_model_params(model)
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Total parameters: {total:,}")
    
    # Test forward pass
    dummy_input = tf.random.normal((2, 224, 224, 3))
    output = model(dummy_input, training=False)
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: (2, 7, 7, {5 * (4 + 1 + 20)})")
