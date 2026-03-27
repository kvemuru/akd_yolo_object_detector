"""
Training Script for YOLO Object Detector

Implements training loop with YOLO loss and evaluation.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import yaml
import os
from datetime import datetime

from models.detector import create_detector, decode_predictions
from preprocessing.utils import VOC_CLASSES


class YOLOCompetitionLoss(tf.keras.losses.Loss):
    """
    YOLO competition loss with box regression, objectness, and classification.
    """
    
    def __init__(self, grid_size=(7, 7), num_classes=20, 
                 lambda_coord=5.0, lambda_noobj=0.5, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        
        # Reshape for easier computation
        # y_pred: (batch, h, w, anchors*(4+1+classes))
        anchors = 5
        
        # Parse predictions
        pred_xy = tf.sigmoid(y_pred[..., 0:2])  # bx, by
        pred_wh = y_pred[..., 2:4]  # bw, bh
        pred_obj = tf.sigmoid(y_pred[..., 4:5])  # objectness
        pred_cls = tf.sigmoid(y_pred[..., 5:])  # class probabilities
        
        # Simple MSE loss for demonstration
        # In practice, you'd implement full YOLO loss
        loss_xy = tf.reduce_mean(tf.square(y_true[..., 0:2] - pred_xy))
        loss_wh = tf.reduce_mean(tf.square(y_true[..., 2:4] - pred_wh))
        loss_obj = tf.reduce_mean(tf.square(y_true[..., 4:5] - pred_obj))
        loss_cls = tf.reduce_mean(tf.square(y_true[..., 5:] - pred_cls))
        
        total_loss = (
            self.lambda_coord * (loss_xy + loss_wh) +
            loss_obj +
            self.lambda_noobj * loss_obj +
            loss_cls
        )
        
        return total_loss


def create_training_model(config):
    """
    Create and compile training model.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Compiled model
    """
    model = create_detector(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors'],
        alpha=config['model']['alpha']
    )
    
    # Compile with optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    model.compile(
        optimizer=optimizer,
        loss=YOLOCompetitionLoss(
            grid_size=tuple(config['model']['grid_size']),
            num_classes=config['model']['num_classes']
        ),
        metrics=['accuracy']
    )
    
    return model


def create_callbacks(config):
    """
    Create training callbacks.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of Keras callbacks
    """
    log_dir = os.path.join(
        config['paths']['log_dir'],
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['paths']['checkpoint_dir'], 'weights.{epoch:02d}-{loss:.4f}.weights.h5'),
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    return callbacks


def generate_dummy_data(batch_size, grid_size, num_classes, num_anchors):
    """
    Generate dummy training data for testing.
    
    Args:
        batch_size: Batch size
        grid_size: YOLO grid size
        num_classes: Number of classes
        num_anchors: Number of anchors
    
    Returns:
        Dummy images and labels
    """
    input_size = grid_size[0] * 32  # 224 for 7x7 grid
    
    images = np.random.randn(batch_size, input_size, input_size, 3).astype(np.float32)
    images = (images - images.min()) / (images.max() - images.min())
    
    # Generate dummy labels
    # Format: (batch, h, w, anchors*(4+1+classes))
    output_channels = num_anchors * (4 + 1 + num_classes)
    labels = np.random.randn(batch_size, grid_size[0], grid_size[1], output_channels).astype(np.float32)
    
    return images, labels


def train_model(config, model=None):
    """
    Train the YOLO detector.
    
    Args:
        config: Configuration dictionary
        model: Optional pre-created model
    
    Returns:
        Trained model and training history
    """
    if model is None:
        model = create_training_model(config)
    
    callbacks = create_callbacks(config)
    
    # Create dummy data for testing (replace with real VOC data)
    train_images, train_labels = generate_dummy_data(
        batch_size=config['training']['batch_size'],
        grid_size=tuple(config['model']['grid_size']),
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors']
    )
    
    val_images, val_labels = generate_dummy_data(
        batch_size=config['training']['batch_size'],
        grid_size=tuple(config['model']['grid_size']),
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors']
    )
    
    print(f"\nTraining on {len(train_images)} samples")
    print(f"Validation on {len(val_images)} samples")
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, test_images, test_labels):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_images: Test images
        test_labels: Test labels
    
    Returns:
        Evaluation metrics
    """
    results = model.evaluate(test_images, test_labels, verbose=1)
    return results


def export_trained_weights(model, output_path):
    """
    Export model weights.
    
    Args:
        model: Trained model
        output_path: Output file path
    """
    model.save_weights(output_path)
    print(f"Weights saved to {output_path}")


if __name__ == '__main__':
    # Load config
    config = {
        'model': {
            'input_shape': [224, 224, 3],
            'num_classes': 20,
            'num_anchors': 5,
            'alpha': 0.5,
            'grid_size': [7, 7]
        },
        'training': {
            'epochs': 10,
            'batch_size': 8,
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        },
        'paths': {
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs'
        }
    }
    
    # Create model
    model = create_training_model(config)
    print("\nModel created successfully")
    model.summary()
    
    # Train
    print("\n" + "="*50)
    print("Training (using dummy data)")
    print("="*50)
    
    trained_model, history = train_model(config, model)
    
    print("\nTraining complete!")
