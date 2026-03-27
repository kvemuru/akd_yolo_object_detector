"""
Model Pruning Utilities for Akida-Compatible Models

Implements magnitude pruning and structured pruning.
"""

import tensorflow as tf
import numpy as np


def compute_weight_sparsity(weights):
    """
    Compute sparsity (fraction of zero weights).
    
    Args:
        weights: Weight tensor or array
    
    Returns:
        Sparsity as fraction
    """
    weights = np.array(weights)
    zero_count = np.sum(weights == 0)
    total_count = weights.size
    return zero_count / total_count


def magnitude_prune_weights(weights, threshold):
    """
    Prune weights below threshold to zero.
    
    Args:
        weights: Weight array
        threshold: Pruning threshold
    
    Returns:
        Pruned weights
    """
    mask = np.abs(weights) >= threshold
    return weights * mask


def compute_pruning_threshold(weights, target_sparsity):
    """
    Compute threshold for target sparsity.
    
    Args:
        weights: Weight array
        target_sparsity: Desired sparsity (0-1)
    
    Returns:
        Threshold value
    """
    abs_weights = np.abs(weights.flatten())
    threshold_idx = int(len(abs_weights) * target_sparsity)
    sorted_weights = np.sort(abs_weights)
    return sorted_weights[threshold_idx]


def prune_model(model, target_sparsity=0.5):
    """
    Prune model weights to target sparsity.
    
    Args:
        model: Keras model
        target_sparsity: Target sparsity (0-1)
    
    Returns:
        Pruned model
    """
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            weights = layer.get_weights()
            pruned_weights = []
            
            for w in weights:
                if len(w.shape) >= 2:  # Only prune weight matrices (not biases)
                    threshold = compute_pruning_threshold(w, target_sparsity)
                    pruned_w = magnitude_prune_weights(w, threshold)
                    pruned_weights.append(pruned_w)
                else:
                    pruned_weights.append(w)
            
            layer.set_weights(pruned_weights)
    
    return model


def iterative_pruning(model, data, target_sparsity=0.7, steps=10):
    """
    Perform iterative magnitude pruning.
    
    Args:
        model: Keras model
        data: Data for fine-tuning between pruning steps
        target_sparsity: Final target sparsity
        steps: Number of pruning steps
    
    Returns:
        Pruned model
    """
    initial_sparsity = 0.0
    sparsity_step = (target_sparsity - initial_sparsity) / steps
    
    current_sparsity = initial_sparsity
    
    for step in range(steps):
        current_sparsity += sparsity_step
        print(f"Pruning step {step + 1}/{steps}, sparsity: {current_sparsity:.2%}")
        
        # Prune
        model = prune_model(model, current_sparsity)
        
        # Fine-tune briefly
        if data is not None:
            model.fit(data, epochs=1, verbose=0)
    
    return model


def structured_prune_filters(model, prune_ratio=0.2):
    """
    Structurally prune filters (entire channels).
    
    Args:
        model: Keras model
        prune_ratio: Fraction of filters to prune per layer
    
    Returns:
        Model with pruned filters
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Compute filter importance (L1 norm)
            weights = layer.get_weights()[0]
            filter_importance = np.sum(np.abs(weights), axis=(0, 1, 2))
            
            # Find least important filters
            num_prune = int(len(filter_importance) * prune_ratio)
            prune_indices = np.argsort(filter_importance)[:num_prune]
            
            # In practice, you'd need to rebuild the model without these filters
            # This is a simplified demonstration
            print(f"Layer {layer.name}: would prune {num_prune} filters")
    
    return model


class MagnitudePruner:
    """Magnitude-based weight pruner."""
    
    def __init__(self, target_sparsity, schedule='linear'):
        self.target_sparsity = target_sparsity
        self.schedule = schedule
        self.current_sparsity = 0.0
    
    def step(self, model, epoch, total_epochs):
        """
        Perform one pruning step.
        
        Args:
            model: Model to prune
            epoch: Current epoch
            total_epochs: Total training epochs
        """
        if self.schedule == 'linear':
            self.current_sparsity = self.target_sparsity * (epoch / total_epochs)
        elif self.schedule == 'cubic':
            progress = epoch / total_epochs
            self.current_sparsity = self.target_sparsity * (progress ** 3)
        
        for layer in model.layers:
            if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                weights = layer.get_weights()
                pruned_weights = []
                
                for w in weights:
                    if len(w.shape) >= 2:
                        threshold = compute_pruning_threshold(w, self.current_sparsity)
                        pruned_w = magnitude_prune_weights(w, threshold)
                        pruned_weights.append(pruned_w)
                    else:
                        pruned_weights.append(w)
                
                layer.set_weights(pruned_weights)
    
    def get_sparsity(self):
        """Get current sparsity level."""
        return self.current_sparsity


def analyze_model_sparsity(model):
    """
    Analyze sparsity across all layers.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with sparsity statistics
    """
    layer_sparsities = []
    total_params = 0
    zero_params = 0
    
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            for w in layer.get_weights():
                w = np.array(w)
                layer_sparsities.append({
                    'layer': layer.name,
                    'shape': w.shape,
                    'sparsity': compute_weight_sparsity(w),
                    'params': w.size,
                    'zeros': np.sum(w == 0)
                })
                total_params += w.size
                zero_params += np.sum(w == 0)
    
    overall_sparsity = zero_params / total_params if total_params > 0 else 0
    
    return {
        'overall_sparsity': overall_sparsity,
        'layer_sparsities': layer_sparsities,
        'total_params': total_params,
        'zero_params': zero_params,
        'nonzero_params': total_params - zero_params
    }


def print_sparsity_report(report):
    """
    Print formatted sparsity report.
    
    Args:
        report: Sparsity analysis report
    """
    print("=" * 60)
    print("SPARSITY REPORT")
    print("=" * 60)
    print(f"Overall Sparsity: {report['overall_sparsity']:.2%}")
    print(f"Total Parameters: {report['total_params']:,}")
    print(f"Zero Parameters: {report['zero_params']:,}")
    print(f"Non-zero Parameters: {report['nonzero_params']:,}")
    print("-" * 60)
    print(f"{'Layer':<30} {'Shape':<20} {'Sparsity':<10}")
    print("-" * 60)
    
    for layer in report['layer_sparsities']:
        shape_str = str(layer['shape']).replace('(', '').replace(')', '')
        print(f"{layer['layer']:<30} {shape_str:<20} {layer['sparsity']:.2%}")
    
    print("=" * 60)


if __name__ == '__main__':
    # Test pruning utilities
    print("Testing pruning utilities...")
    
    # Create simple test model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Analyze initial sparsity
    report = analyze_model_sparsity(model)
    print(f"Initial sparsity: {report['overall_sparsity']:.4%}")
    
    # Prune to 50%
    pruned_model = prune_model(model, target_sparsity=0.5)
    report = analyze_model_sparsity(pruned_model)
    print(f"After 50% pruning: {report['overall_sparsity']:.2%}")
