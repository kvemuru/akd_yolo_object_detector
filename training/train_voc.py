"""
Real Data Training for YOLO Object Detector

Downloads PASCAL-VOC dataset and trains the detector.
"""

import os
import urllib.request
import tarfile
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yaml


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
VOC_CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}


def download_voc_dataset(data_dir='./data'):
    """Download and extract VOC dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    voc_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2007')
    if os.path.exists(voc_dir):
        print(f"VOC dataset already exists at {voc_dir}")
        return voc_dir
    
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
    tar_path = os.path.join(data_dir, 'voc_trainval.tar')
    
    print(f"Downloading VOC dataset from {url}...")
    print("This may take a few minutes...")
    
    urllib.request.urlretrieve(url, tar_path)
    print("Extracting...")
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(data_dir)
    
    os.remove(tar_path)
    print(f"VOC dataset extracted to {voc_dir}")
    
    return voc_dir


def parse_voc_annotation(xml_path):
    """Parse VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    boxes = []
    labels = []
    
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue
        
        class_name = obj.find('name').text
        if class_name not in VOC_CLASS_TO_IDX:
            continue
        
        label = VOC_CLASS_TO_IDX[class_name]
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    
    return boxes, labels, (width, height)


def load_voc_data(data_dir='./data', split='train'):
    """
    Load VOC dataset.
    
    Args:
        data_dir: Directory containing VOCdevkit
        split: 'train' or 'val'
    
    Returns:
        List of (image_path, boxes, labels)
    """
    voc_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2007')
    image_dir = os.path.join(voc_dir, 'JPEGImages')
    annotation_dir = os.path.join(voc_dir, 'Annotations')
    
    if split == 'train':
        ids_file = os.path.join(voc_dir, 'ImageSets', 'Main', 'train.txt')
    else:
        ids_file = os.path.join(voc_dir, 'ImageSets', 'Main', 'val.txt')
    
    if not os.path.exists(ids_file):
        ids_file = os.path.join(voc_dir, 'ImageSets', 'Main', 'trainval.txt')
    
    with open(ids_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    data = []
    for img_id in image_ids:
        img_path = os.path.join(image_dir, f'{img_id}.jpg')
        ann_path = os.path.join(annotation_dir, f'{img_id}.xml')
        
        if os.path.exists(img_path) and os.path.exists(ann_path):
            boxes, labels, size = parse_voc_annotation(ann_path)
            if len(boxes) > 0:
                data.append((img_path, boxes, labels, size))
    
    return data


class VOCDataset:
    """VOC dataset for YOLO training."""
    
    def __init__(self, data, grid_size=(7, 7), num_classes=20, num_anchors=5,
                 batch_size=8, augment=True, target_size=(224, 224)):
        self.data = data
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.augment = augment
        self.target_size = target_size
        self.anchors = np.array([
            [1.12, 1.85], [2.00, 2.83], [3.19, 3.60],
            [4.55, 5.26], [5.49, 5.82]
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, boxes, labels, orig_size = self.data[idx]
        
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        orig_h, orig_w = orig_size
        
        if self.augment:
            if tf.random.uniform([]) > 0.5:
                image = tf.image.flip_left_right(image)
                boxes = np.array(boxes).copy()
                boxes[:, [0, 2]] = orig_w - boxes[:, [2, 0]]
        
        image = tf.image.resize(image, self.target_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        gh, gw = self.grid_size
        cell_h = self.target_size[0] / gh
        cell_w = self.target_size[1] / gw
        
        label = np.zeros((gh, gw, self.num_anchors, 5 + self.num_classes), dtype=np.float32)
        
        for box, label_idx in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / orig_w * self.target_size[1]
            cy = (y1 + y2) / 2 / orig_h * self.target_size[0]
            w = (x2 - x1) / orig_w * self.target_size[1]
            h = (y2 - y1) / orig_h * self.target_size[0]
            
            grid_x = min(int(cx / cell_w), gw - 1)
            grid_y = min(int(cy / cell_h), gh - 1)
            
            best_iou = 0
            best_anchor = 0
            for a, anchor in enumerate(self.anchors):
                anchor_w, anchor_h = anchor
                anchor_w_px = anchor_w / gw * self.target_size[1]
                anchor_h_px = anchor_h / gh * self.target_size[0]
                
                iou_w = min(w, anchor_w_px)
                iou_h = min(h, anchor_h_px)
                iou = (iou_w * iou_h) / (w * h + anchor_w_px * anchor_h_px - iou_w * iou_h + 1e-6)
                
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = a
            
            label[grid_y, grid_x, best_anchor, 0] = cx / cell_w - grid_x
            label[grid_y, grid_x, best_anchor, 1] = cy / cell_h - grid_y
            label[grid_y, grid_x, best_anchor, 2] = np.log(w / (self.target_size[1] / gw) + 1e-6)
            label[grid_y, grid_x, best_anchor, 3] = np.log(h / (self.target_size[0] / gh) + 1e-6)
            label[grid_y, grid_x, best_anchor, 4] = 1.0
            label[grid_y, grid_x, best_anchor, 5 + label_idx] = 1.0
        
        label = tf.reshape(label, (gh, gw, self.num_anchors * (5 + self.num_classes)))
        return tf.expand_dims(image, 0), tf.expand_dims(label, 0)


def create_dataset(data, grid_size, num_classes, num_anchors, batch_size, augment=True):
    """Create tf.data.Dataset from VOC data."""
    dataset = VOCDataset(data, grid_size, num_classes, num_anchors, 
                        batch_size, augment)
    
    def generator():
        for i in range(len(dataset)):
            yield dataset[i]
    
    output_signature = (
        tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1, grid_size[0], grid_size[1], num_anchors * (5 + num_classes)), dtype=tf.float32)
    )
    
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


def train_with_real_data(config, epochs=5):
    """Train model with real VOC data."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.detector import create_detector, count_model_params
    
    print("=" * 60)
    print("TRAINING WITH VOC DATASET")
    print("=" * 60)
    
    print("\n[1/4] Downloading VOC dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    voc_dir = download_voc_dataset(data_dir)
    
    print("\n[2/4] Loading training data...")
    train_data = load_voc_data(data_dir, 'train')
    print(f"Loaded {len(train_data)} training images")
    
    print("\n[3/4] Loading validation data...")
    val_data = load_voc_data(data_dir, 'val')
    print(f"Loaded {len(val_data)} validation images")
    
    print("\n[4/4] Creating model...")
    model = create_detector(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors'],
        alpha=config['model']['alpha']
    )
    
    trainable, non_trainable, total = count_model_params(model)
    print(f"Model parameters: {total:,}")
    
    grid_size = tuple(config['model']['grid_size'])
    num_classes = config['model']['num_classes']
    num_anchors = config['model']['num_anchors']
    batch_size = min(config['training']['batch_size'], 8)
    
    train_ds = create_dataset(train_data, grid_size, num_classes, num_anchors, batch_size)
    val_ds = create_dataset(val_data, grid_size, num_classes, num_anchors, batch_size, augment=False)
    
    train_ds = train_ds.repeat()
    val_ds = val_ds.repeat()
    
    optimizer = keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    loss_fn = keras.losses.MeanSquaredError()
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        steps = 0
        
        for images, labels in train_ds.take(50):
            with tf.GradientTape() as tape:
                preds = model(images, training=True)
                loss = loss_fn(labels, preds)
            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            epoch_loss += loss.numpy()
            steps += 1
        
        avg_loss = epoch_loss / steps
        history['loss'].append(avg_loss)
        
        val_loss = 0
        val_steps = 0
        for images, labels in val_ds.take(20):
            preds = model(images, training=False)
            val_loss += loss_fn(labels, preds).numpy()
            val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    return model, history


if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['batch_size'] = 8
    model, history = train_with_real_data(config, epochs=5)
