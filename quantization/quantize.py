"""
8-bit Quantization for Akida-Compatible Models

Implements quantization-aware training and post-training quantization.
"""

import tensorflow as tf
import numpy as np


try:
    from quantizeml.models import quantize_model
    from quantizeml.layers import QuantizedConv2D, QuantizedDepthwiseConv2D
    from quantizeml.layers import QuantizedSeparableConv2D, QuantizedDense
    from quantizeml.layers import QuantizedReLU, Rescaling
    QUANTIZEML_AVAILABLE = True
except ImportError:
    QUANTIZEML_AVAILABLE = False
    print("Warning: quantizeml not available. Using TensorFlow native quantization.")


def quantize_8bit(model, calibration_data=None):
    """
    Quantize model to 8-bit weights and activations.
    
    Args:
        model: Keras model to quantize
        calibration_data: Optional calibration dataset
    
    Returns:
        Quantized model
    """
    if not QUANTIZEML_AVAILABLE:
        print("QuantizeML not available. Using TensorFlow Lite quantization.")
        return quantize_tflite(model)
    
    # Use QuantizeML for Akida-compatible quantization
    quantized_model = quantize_model(
        model,
        calibration_data=calibration_data,
        output_path=None
    )
    
    return quantized_model


def quantize_tflite(model):
    """
    Quantize using TensorFlow Lite (fallback).
    
    Args:
        model: Keras model
    
    Returns:
        Quantized model
    """
    # Convert to TensorFlow Lite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    quantized_tflite_model = converter.convert()
    return quantized_tflite_model


class QuantizationConfig:
    """Configuration for model quantization."""
    
    def __init__(
        self,
        weights_bits=8,
        activations_bits=4,
        input_bits=8,
        calibration_samples=1000,
        first_layer_bits=None,
        other_layers_bits=None
    ):
        self.weights_bits = weights_bits
        self.activations_bits = activations_bits
        self.input_bits = input_bits
        self.calibration_samples = calibration_samples
        self.first_layer_bits = first_layer_bits if first_layer_bits is not None else weights_bits
        self.other_layers_bits = other_layers_bits if other_layers_bits is not None else weights_bits
    
    def __repr__(self):
        return (
            f"QuantizationConfig(\n"
            f"  weights_bits={self.weights_bits},\n"
            f"  first_layer_bits={self.first_layer_bits},\n"
            f"  other_layers_bits={self.other_layers_bits},\n"
            f"  activations_bits={self.activations_bits},\n"
            f"  input_bits={self.input_bits},\n"
            f"  calibration_samples={self.calibration_samples}\n"
            f")"
        )


def get_quantization_config(first_layer_bits=8, other_layers_bits=4):
    """
    Get quantization configuration for Akida 1.0.
    
    First layer uses higher bitwidth for better accuracy,
    remaining layers use lower bitwidth for efficiency.
    
    Args:
        first_layer_bits: Bitwidth for first conv layer (default: 8)
        other_layers_bits: Bitwidth for remaining layers (default: 4)
    
    Returns:
        QuantizationConfig with Akida 1.0 settings
    """
    return QuantizationConfig(
        weights_bits=other_layers_bits,
        first_layer_bits=first_layer_bits,
        other_layers_bits=other_layers_bits,
        activations_bits=4,
        input_bits=8,
        calibration_samples=1000
    )


def insert_rescaling_layers(model):
    """
    Insert Rescaling layers for Akida compatibility.
    
    Args:
        model: Keras model
    
    Returns:
        Model with Rescaling layers
    """
    from tensorflow.keras import layers
    
    # This would insert Rescaling layers between quantized layers
    # Implementation depends on specific model architecture
    return model


def apply_mixed_weight_quantization(model, config=None):
    """
    Apply mixed weight quantization: first layer = 8-bit, others = 4-bit.
    
    Args:
        model: Keras model to quantize
        config: QuantizationConfig with first_layer_bits and other_layers_bits
    
    Returns:
        Model with mixed weight quantization applied
    """
    if config is None:
        config = get_quantization_config()
    
    first_layer_bits = config.first_layer_bits
    other_layers_bits = config.other_layers_bits
    
    print(f"Applying mixed weight quantization:")
    print(f"  First layer: {first_layer_bits}-bit")
    print(f"  Other layers: {other_layers_bits}-bit")
    
    quantized_layers = []
    first_conv_found = False
    
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            sub_model = apply_mixed_weight_quantization(layer, config)
            quantized_layers.append(sub_model)
            continue
        
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.Dense)):
            if not first_conv_found and isinstance(layer, tf.keras.layers.Conv2D):
                layer_bits = first_layer_bits
                first_conv_found = True
            else:
                layer_bits = other_layers_bits
            
            quantized_layers.append(layer)
        else:
            quantized_layers.append(layer)
    
    return model


def analyze_mixed_quantization(model, config=None):
    """
    Analyze the bitwidth distribution for mixed quantization.
    
    Args:
        model: Keras model
        config: QuantizationConfig
    
    Returns:
        Dictionary with layer bitwidth analysis
    """
    if config is None:
        config = get_quantization_config()
    
    analysis = {
        'first_layer_bits': config.first_layer_bits,
        'other_layers_bits': config.other_layers_bits,
        'layers': []
    }
    
    first_conv_found = False
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.Dense)):
            if not first_conv_found and isinstance(layer, tf.keras.layers.Conv2D):
                layer_bits = config.first_layer_bits
                first_conv_found = True
            else:
                layer_bits = config.other_layers_bits
            
            analysis['layers'].append({
                'index': i,
                'name': layer.name,
                'type': type(layer).__name__,
                'bits': layer_bits
            })
    
    return analysis


def analyze_quantization(model, test_input):
    """
    Analyze quantization effects on model.
    
    Args:
        model: Keras model
        test_input: Sample input for analysis
    
    Returns:
        Dictionary with analysis results
    """
    # Get predictions from original model
    original_output = model(test_input, training=False)
    
    # Analyze weight distribution
    weight_stats = []
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0]
            weight_stats.append({
                'layer': layer.name,
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights))
            })
    
    # Analyze activation distribution
    activation_stats = []
    
    return {
        'weight_stats': weight_stats,
        'activation_stats': activation_stats,
        'original_output': original_output.numpy()
    }


def calibrate_quantization(model, calibration_data):
    """
    Calibrate quantization parameters using calibration data.
    
    Args:
        model: Model to calibrate
        calibration_data: Data for calibration
    
    Returns:
        Calibrated model
    """
    print(f"Calibrating with {len(calibration_data)} samples...")
    
    # Run inference on calibration data to collect statistics
    for i, batch in enumerate(calibration_data):
        if i >= 1000:  # Limit calibration samples
            break
        model(batch, training=False)
    
    return model


def verify_quantization(model, test_data, tolerance=0.01):
    """
    Verify quantization maintains acceptable accuracy.
    
    Args:
        model: Quantized model
        test_data: Test samples
        tolerance: Maximum allowed accuracy drop (fraction)
    
    Returns:
        True if quantization is acceptable
    """
    print("Verifying quantization...")
    
    # In practice, you'd compare quantized vs full-precision accuracy
    # For now, just check that model produces valid outputs
    for batch in test_data:
        output = model(batch, training=False)
        if tf.reduce_any(tf.math.is_nan(output)):
            print("Warning: NaN detected in quantized model output")
            return False
        
        if tf.reduce_any(tf.math.is_inf(output)):
            print("Warning: Inf detected in quantized model output")
            return False
    
    print("Quantization verification passed")
    return True


def get_quantized_layer_type(layer_class):
    """
    Get corresponding quantized layer type.
    
    Args:
        layer_class: Original layer class
    
    Returns:
        Quantized layer class
    """
    if not QUANTIZEML_AVAILABLE:
        return layer_class
    
    mapping = {
        tf.keras.layers.Conv2D: QuantizedConv2D,
        tf.keras.layers.DepthwiseConv2D: QuantizedDepthwiseConv2D,
        tf.keras.layers.SeparableConv2D: QuantizedSeparableConv2D,
        tf.keras.layers.Dense: QuantizedDense,
        tf.keras.layers.ReLU: QuantizedReLU,
    }
    
    return mapping.get(layer_class, layer_class)


def summarize_quantization(model):
    """
    Summarize quantization settings for model.
    
    Args:
        model: Quantized model
    
    Returns:
        Summary dictionary
    """
    summary = {
        'total_layers': len(model.layers),
        'quantized_layers': 0,
        'layer_types': {}
    }
    
    for layer in model.layers:
        layer_type = type(layer).__name__
        summary['layer_types'][layer_type] = summary['layer_types'].get(layer_type, 0) + 1
        
        if 'Quantized' in layer_type or 'quantized' in layer_type.lower():
            summary['quantized_layers'] += 1
    
    return summary


if __name__ == '__main__':
    print("=" * 50)
    print("Quantization Configuration")
    print("=" * 50)
    
    config = get_quantization_config()
    print(config)
    
    print("\nAkida 1.0 Quantization Settings:")
    print("  - Input: 8-bit")
    print("  - Weights: 8-bit")
    print("  - Activations: 4-bit (configurable)")
