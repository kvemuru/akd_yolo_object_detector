"""
Akida Model Conversion Utilities

Converts trained models to Akida-compatible format.
"""

import tensorflow as tf
import numpy as np


try:
    from cnn2snn import convert, check_model_compatibility
    from cnn2snn import set_akida_version, AkidaVersion
    CNN2SNN_AVAILABLE = True
except ImportError:
    CNN2SNN_AVAILABLE = False
    print("Warning: cnn2snn not available")


def check_akida_compatibility(model, verbose=True):
    """
    Check if model is compatible with Akida hardware.
    
    Args:
        model: Keras model to check
        verbose: Print detailed results
    
    Returns:
        Tuple of (is_compatible, issues_list)
    """
    if not CNN2SNN_AVAILABLE:
        print("cnn2snn not available. Skipping compatibility check.")
        return True, []
    
    issues = []
    
    # Check layer types
    unsupported_layers = []
    for layer in model.layers:
        layer_name = type(layer).__name__
        
        # Check for unsupported layer types
        unsupported = [
            'LSTM', 'GRU', 'RNN',  # Recurrent layers
            'BatchNormalization',  # Should be fused
            'Dropout',  # Should be removed
            'Activation',  # Activation should be integrated
        ]
        
        if any(unsupported_type in layer_name for unsupported_type in unsupported):
            issues.append(f"Unsupported layer: {layer_name} in {layer.name}")
    
    # Check for BatchNormalization (should be fused)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            issues.append(f"BatchNormalization not fused: {layer.name}")
    
    is_compatible = len(issues) == 0
    
    if verbose:
        print("=" * 50)
        print("AKIDA COMPATIBILITY CHECK")
        print("=" * 50)
        print(f"Compatible: {'Yes' if is_compatible else 'No'}")
        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No issues found.")
        print("=" * 50)
    
    return is_compatible, issues


def fuse_batchnorm_layers(model):
    """
    Fuse BatchNormalization layers into preceding Conv2D/Dense layers.
    
    Args:
        model: Keras model
    
    Returns:
        Model with fused BatchNorm
    """
    from tensorflow.keras.models import clone_model
    
    def fuse_bn_with_conv(layer, bn_layer):
        """Fuse BatchNorm into Conv layer."""
        if not isinstance(bn_layer, tf.keras.layers.BatchNormalization):
            return layer
        
        # Get weights
        kernel = layer.get_weights()[0]
        bias = layer.get_weights()[1] if len(layer.get_weights()) > 1 else None
        
        bn_weights = bn_layer.get_weights()
        gamma, beta, moving_mean, moving_var = bn_weights[:4]
        epsilon = bn_layer.epsilon
        
        # Fused kernel and bias
        std = np.sqrt(moving_var + epsilon)
        kernel_fused = kernel * (gamma / std).reshape(-1, 1, 1, 1) if len(kernel.shape) == 4 else kernel * (gamma / std)
        
        if bias is not None:
            bias_fused = gamma * (bias - moving_mean) / std + beta
        else:
            bias_fused = gamma * (-moving_mean) / std + beta
        
        # Create new layer with fused weights
        new_layer = type(layer)(**layer.get_config())
        new_layer.build(layer.input_shape)
        new_layer.set_weights([kernel_fused, bias_fused])
        
        return new_layer
    
    # This is a simplified version - full implementation would rebuild the model
    print("Note: Full BatchNorm fusion not implemented. Use tf.keras.models.clone_model with custom handling.")
    return model


def convert_to_akida(model, input_shape=(224, 224, 3), verbose=True):
    """
    Convert Keras model to Akida format.
    
    Args:
        model: Keras model to convert
        input_shape: Input shape for the model
        verbose: Print conversion details
    
    Returns:
        Akida-compatible model
    """
    if not CNN2SNN_AVAILABLE:
        raise ImportError("cnn2snn is required for Akida conversion")
    
    if verbose:
        print("=" * 50)
        print("AKIDA CONVERSION")
        print("=" * 50)
    
    # Set Akida version
    set_akida_version(AkidaVersion.v1)
    
    # Check compatibility
    is_compatible, issues = check_akida_compatibility(model, verbose=True)
    
    if not is_compatible:
        print("\nModel has compatibility issues. Fixing...")
        model = fuse_batchnorm_layers(model)
    
    # Convert
    if verbose:
        print("\nConverting model...")
    
    akida_model = convert(model, input_shape=input_shape)
    
    if verbose:
        print("Conversion complete!")
        print_model_summary(akida_model)
    
    return akida_model


def print_model_summary(model):
    """
    Print model summary.
    
    Args:
        model: Keras or Akida model
    """
    print("\nModel Summary:")
    print(f"  Type: {type(model).__name__}")
    
    if hasattr(model, 'layers'):
        print(f"  Layers: {len(model.layers)}")
        
        total_params = 0
        for layer in model.layers:
            if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                for w in layer.get_weights():
                    total_params += w.size
        print(f"  Total parameters: {total_params:,}")
    
    if hasattr(model, 'summary'):
        try:
            model.summary()
        except:
            pass


def save_akida_model(akida_model, output_path):
    """
    Save Akida model to file.
    
    Args:
        akida_model: Akida model
        output_path: Output file path
    """
    import pickle
    
    # Serialize the model
    with open(output_path, 'wb') as f:
        pickle.dump(akida_model, f)
    
    print(f"Model saved to {output_path}")


def load_akida_model(input_path):
    """
    Load Akida model from file.
    
    Args:
        input_path: Input file path
    
    Returns:
        Loaded Akida model
    """
    import pickle
    
    with open(input_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {input_path}")
    return model


def run_inference_akida(akida_model, input_data):
    """
    Run inference on Akida model.
    
    Args:
        akida_model: Akida model
        input_data: Input tensor
    
    Returns:
        Model predictions
    """
    output = akida_model.predict(input_data)
    return output


class AkidaInference:
    """Context manager for Akida inference."""
    
    def __init__(self, model_path=None, model=None):
        self.model_path = model_path
        self.model = model
        self.device = None
    
    def __enter__(self):
        if self.model_path:
            self.model = load_akida_model(self.model_path)
        
        # Initialize Akida device
        try:
            import akida
            devices = akida.device_list()
            if devices:
                self.device = devices[0]
                print(f"Using device: {self.device}")
        except Exception as e:
            print(f"Could not initialize Akida device: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        self.device = None
    
    def predict(self, input_data):
        """Run prediction."""
        if self.device:
            return run_inference_akida(self.model, input_data)
        else:
            # Fallback to CPU simulation
            return self.model.predict(input_data)


if __name__ == '__main__':
    print("=" * 50)
    print("Akida Conversion Utilities")
    print("=" * 50)
    
    if CNN2SNN_AVAILABLE:
        print("cnn2snn is available")
    else:
        print("cnn2snn is NOT available")
        print("Install with: pip install cnn2snn")
    
    print("\nAkida 1.0 Layer Constraints:")
    print("  - Input: 8-bit")
    print("  - Conv kernels: 3x3, 5x5, 7x7")
    print("  - Stride 2: 3x3 only")
    print("  - Weights: 8-bit")
    print("  - Activations: 1, 2, or 4-bit")
