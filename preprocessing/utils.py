"""
Data Preprocessing Utilities for Object Detection

Provides image preprocessing, augmentation, and bounding box handling.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


# VOC 2007/2012 class names
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

VOC_CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}
VOC_IDX_TO_CLASS = {idx: name for idx, name in enumerate(VOC_CLASSES)}


def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess image.
    
    Args:
        image_path: Path to image file
        target_size: Target resize size (width, height)
    
    Returns:
        Preprocessed image tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32)
    return image


def preprocess_for_training(image, bbox, labels, target_size=(224, 224)):
    """
    Preprocess image and boxes for training.
    
    Args:
        image: Input image tensor
        bbox: Bounding boxes (N, 4) normalized [0, 1]
        labels: Class labels (N,)
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image, boxes, labels
    """
    # Resize image
    image = tf.image.resize(image, target_size)
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    return image, bbox, labels


def preprocess_for_inference(image, target_size=(224, 224)):
    """
    Preprocess image for inference.
    
    Args:
        image: Input image (PIL Image, numpy array, or tensor)
        target_size: Target size
    
    Returns:
        Preprocessed image tensor
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if isinstance(image, np.ndarray):
        image = tf.constant(image)
    
    if image.dtype != tf.float32:
        image = tf.cast(image, tf.float32)
    
    image = tf.image.resize(image, target_size)
    image = image / 255.0
    
    return image


def random_flip(image, bbox, labels, p=0.5):
    """
    Randomly flip image horizontally.
    
    Args:
        image: Image tensor
        bbox: Bounding boxes (N, 4) normalized
        labels: Class labels
    
    Returns:
        Flipped/unflipped image, boxes, labels
    """
    if tf.random.uniform([]) < p:
        image = tf.image.flip_left_right(image)
        bbox = tf.stack([
            1.0 - bbox[:, 2],  # x2 -> 1 - x2
            bbox[:, 1],         # y1 unchanged
            1.0 - bbox[:, 0],  # x1 -> 1 - x1
            bbox[:, 3]          # y2 unchanged
        ], axis=-1)
    return image, bbox, labels


def random_crop(image, bbox, labels, target_size=(224, 224)):
    """
    Randomly crop image with bounding boxes.
    
    Args:
        image: Image tensor
        bbox: Bounding boxes (N, 4) normalized
        labels: Class labels
        target_size: Target crop size
    
    Returns:
        Cropped image, boxes, labels
    """
    # Calculate crop box
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    crop_h, crop_w = target_size
    
    # Random crop position
    max_offset_h = tf.maximum(h - crop_h, 0)
    max_offset_w = tf.maximum(w - crop_w, 0)
    
    offset_h = tf.random.uniform([], 0, max_offset_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, max_offset_w + 1, dtype=tf.int32)
    
    # Crop image
    image = image[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w]
    
    # Adjust boxes
    scale_h = tf.cast(crop_h, tf.float32) / tf.cast(h, tf.float32)
    scale_w = tf.cast(crop_w, tf.float32) / tf.cast(w, tf.float32)
    
    # Normalize and adjust
    bbox = bbox * tf.stack([scale_w, scale_h, scale_w, scale_h])
    
    # Filter boxes that are mostly inside crop
    bbox_min = bbox[:, :2]
    bbox_max = bbox[:, 2:]
    
    min_coord = offset_w / tf.cast(w, tf.float32) * tf.ones_like(bbox[:, :2])
    
    bbox = bbox - tf.concat([min_coord, min_coord], axis=-1)
    
    # Keep boxes within [0, 1] range
    bbox = tf.clip_by_value(bbox, 0.0, 1.0)
    
    # Filter boxes that have significant overlap with crop
    box_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    keep_mask = box_area > 0.01
    
    bbox = tf.boolean_mask(bbox, keep_mask)
    labels = tf.boolean_mask(labels, keep_mask)
    
    # Resize to target
    image = tf.image.resize(image, target_size)
    
    return image, bbox, labels


def normalize_boxes(boxes, image_shape):
    """
    Normalize boxes to [0, 1] range.
    
    Args:
        boxes: Boxes in pixel coordinates [x1, y1, x2, y2]
        image_shape: (height, width)
    
    Returns:
        Normalized boxes
    """
    h, w = image_shape[0], image_shape[1]
    return boxes / np.array([w, h, w, h])


def denormalize_boxes(boxes, image_shape):
    """
    Convert normalized boxes back to pixel coordinates.
    
    Args:
        boxes: Normalized boxes [x1, y1, x2, y2] in [0, 1]
        image_shape: (height, width)
    
    Returns:
        Boxes in pixel coordinates
    """
    h, w = image_shape[0], image_shape[1]
    return boxes * np.array([w, h, w, h])


def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1: (N, 4) boxes [x1, y1, x2, y2]
        box2: (M, 4) boxes
    
    Returns:
        (N, M) IoU matrix
    """
    # Expand dimensions
    box1 = tf.expand_dims(box1, -2)  # (N, 1, 4)
    box2 = tf.expand_dims(box2, 0)     # (1, M, 4)
    
    # Compute intersection
    inter_x1 = tf.maximum(box1[..., 0], box2[..., 0])
    inter_y1 = tf.maximum(box1[..., 1], box2[..., 1])
    inter_x2 = tf.minimum(box1[..., 2], box2[..., 2])
    inter_y2 = tf.minimum(box1[..., 3], box2[..., 3])
    
    inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)
    
    # Compute union
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    union_area = tf.expand_dims(area1, -1) + tf.expand_dims(area2, 0) - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


def draw_boxes(image, boxes, labels, scores=None, class_names=VOC_CLASSES):
    """
    Draw bounding boxes on image.
    
    Args:
        image: Image array (h, w, 3)
        boxes: (N, 4) in [x1, y1, x2, y2] format
        labels: (N,) class indices
        scores: (N,) confidence scores (optional)
        class_names: List of class names
    
    Returns:
        Image with boxes drawn
    """
    if isinstance(image, tf.Tensor):
        image = image.numpy().astype(np.uint8)
    
    image = image.copy()
    h, w = image.shape[:2]
    
    colors = np.random.randint(0, 255, (len(class_names), 3), dtype=np.uint8)
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        
        color = tuple(map(int, colors[label]))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        class_name = class_names[label]
        text = class_name
        if scores is not None:
            text = f"{class_name}: {scores[i]:.2f}"
        
        cv2.putText(image, text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


class VOCDataGenerator:
    """
    Data generator for VOC format annotations.
    """
    
    def __init__(self, image_paths, annotations, batch_size=32, 
                 target_size=(224, 224), shuffle=True, augment=False):
        self.image_paths = image_paths
        self.annotations = annotations  # List of dicts with 'boxes' and 'labels'
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_samples = len(image_paths)
        
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        batch_boxes = []
        batch_labels = []
        
        for i in batch_indices:
            image = load_image(self.image_paths[i], self.target_size)
            
            ann = self.annotations[i]
            boxes = ann['boxes']
            labels = ann['labels']
            
            if self.augment:
                image, boxes, labels = random_flip(image, boxes, labels)
            
            images.append(image)
            batch_boxes.append(boxes)
            batch_labels.append(labels)
        
        return tf.stack(images), (batch_boxes, batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    # Test utilities
    print("VOC Classes:", VOC_CLASSES)
    print("Number of classes:", len(VOC_CLASSES))
    
    # Test box normalization
    boxes = np.array([[100, 100, 200, 200], [50, 50, 150, 150]])
    normalized = normalize_boxes(boxes, (480, 640))
    print("\nOriginal boxes:", boxes)
    print("Normalized boxes:", normalized)
