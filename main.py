"""
Main CLI for Akida Object Detector

Command-line interface for training, quantization, and deployment.
"""

import argparse
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import create_detector, count_model_params
from training.train import train_model, create_training_model
from quantization.quantize import get_quantization_config, quantize_8bit
from quantization.prune import prune_model, analyze_model_sparsity, print_sparsity_report
from conversion.to_akida import convert_to_akida, check_akida_compatibility


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using defaults.")
        return get_default_config()


def get_default_config():
    """Get default configuration."""
    return {
        'model': {
            'input_shape': [224, 224, 3],
            'num_classes': 20,
            'num_anchors': 5,
            'alpha': 0.5,
            'grid_size': [7, 7]
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        },
        'pruning': {
            'enabled': True,
            'target_sparsity': 0.7
        },
        'quantization': {
            'enabled': True,
            'weights_bits': 8,
            'activations_bits': 4
        },
        'paths': {
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs',
            'output_dir': './output'
        }
    }


def cmd_train(args, config):
    """Train the model."""
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Create model
    print("\nCreating model...")
    model = create_training_model(config)
    
    # Print model info
    trainable, non_trainable, total = count_model_params(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Train
    print("\nStarting training...")
    trained_model, history = train_model(config, model)
    
    # Save
    output_path = os.path.join(config['paths']['checkpoint_dir'], 'final_model.weights.h5')
    trained_model.save_weights(output_path)
    print(f"\nModel saved to {output_path}")
    
    return trained_model


def cmd_quantize(args, config):
    """Quantize the model."""
    print("=" * 60)
    print("QUANTIZATION")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = create_detector(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors'],
        alpha=config['model']['alpha']
    )
    
    # Print original stats
    print("\nOriginal model statistics:")
    trainable, non_trainable, total = count_model_params(model)
    print(f"  Total parameters: {total:,}")
    
    # Quantize
    print("\nQuantizing model...")
    quantized_model = quantize_8bit(model)
    
    # Save
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    output_path = os.path.join(config['paths']['output_dir'], 'quantized_model.h5')
    quantized_model.save(output_path)
    print(f"\nQuantized model saved to {output_path}")
    
    return quantized_model


def cmd_prune(args, config):
    """Prune the model."""
    print("=" * 60)
    print("PRUNING")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = create_detector(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors'],
        alpha=config['model']['alpha']
    )
    
    # Analyze initial sparsity
    print("\nInitial sparsity:")
    report = analyze_model_sparsity(model)
    print_sparsity_report(report)
    
    # Prune
    target_sparsity = config['pruning'].get('target_sparsity', 0.7)
    print(f"\nPruning to {target_sparsity:.0%} sparsity...")
    pruned_model = prune_model(model, target_sparsity=target_sparsity)
    
    # Analyze final sparsity
    print("\nFinal sparsity:")
    report = analyze_model_sparsity(pruned_model)
    print_sparsity_report(report)
    
    # Save
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    output_path = os.path.join(config['paths']['output_dir'], f'pruned_model_{int(target_sparsity*100)}p.h5')
    pruned_model.save_weights(output_path)
    print(f"\nPruned model saved to {output_path}")
    
    return pruned_model


def cmd_convert(args, config):
    """Convert model to Akida format."""
    print("=" * 60)
    print("AKIDA CONVERSION")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = create_detector(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors'],
        alpha=config['model']['alpha']
    )
    
    # Check compatibility
    print("\nChecking Akida compatibility...")
    is_compatible, issues = check_akida_compatibility(model)
    
    if not is_compatible:
        print("\nModel has compatibility issues. Please fix before conversion.")
        return None
    
    # Convert
    print("\nConverting to Akida format...")
    akida_model = convert_to_akida(
        model,
        input_shape=tuple(config['model']['input_shape'])
    )
    
    # Save
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    output_path = os.path.join(config['paths']['output_dir'], 'akida_model.akida')
    
    from conversion.to_akida import save_akida_model
    save_akida_model(akida_model, output_path)
    
    return akida_model


def cmd_infer(args, config):
    """Run inference on an image."""
    print("=" * 60)
    print("INFERENCE")
    print("=" * 60)
    
    if not args.image:
        print("Error: --image required for inference")
        return
    
    # Load model
    print("\nLoading model...")
    model = create_detector(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        num_anchors=config['model']['num_anchors'],
        alpha=config['model']['alpha']
    )
    
    # Load weights if provided
    if args.weights:
        print(f"Loading weights from {args.weights}")
        model.load_weights(args.weights)
    
    # Load and preprocess image
    from preprocessing.utils import load_image, preprocess_for_inference
    
    print(f"\nLoading image: {args.image}")
    image = load_image(args.image, target_size=tuple(config['model']['input_shape'][:2]))
    image = preprocess_for_inference(image)
    
    # Run inference
    print("\nRunning inference...")
    import tensorflow as tf
    image_batch = tf.expand_dims(image, 0)
    predictions = model(image_batch, training=False)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print("Note: Implement decoding logic to get bounding boxes")
    
    return predictions


def cmd_full_pipeline(args, config):
    """Run the full pipeline: train -> prune -> quantize -> convert."""
    print("=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)
    
    # Train
    print("\n[1/4] Training...")
    model = cmd_train(args, config)
    
    # Prune
    print("\n[2/4] Pruning...")
    pruned_model = cmd_prune(args, config)
    
    # Quantize
    print("\n[3/4] Quantizing...")
    quantized_model = cmd_quantize(args, config)
    
    # Convert
    print("\n[4/4] Converting to Akida...")
    akida_model = cmd_convert(args, config)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Akida Object Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py --mode train --config config.yaml
  
  # Prune a trained model
  python main.py --mode prune --weights checkpoints/final_model.h5
  
  # Quantize a model
  python main.py --mode quantize --weights checkpoints/final_model.h5
  
  # Convert to Akida
  python main.py --mode convert --weights checkpoints/final_model.h5
  
  # Run inference
  python main.py --mode infer --image image.jpg --weights model.h5
  
  # Full pipeline
  python main.py --mode full
"""
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'prune', 'quantize', 'convert', 'infer', 'full'],
        default='train',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        help='Path to model weights'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image for inference'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run selected mode
    if args.mode == 'train':
        cmd_train(args, config)
    elif args.mode == 'prune':
        cmd_prune(args, config)
    elif args.mode == 'quantize':
        cmd_quantize(args, config)
    elif args.mode == 'convert':
        cmd_convert(args, config)
    elif args.mode == 'infer':
        cmd_infer(args, config)
    elif args.mode == 'full':
        cmd_full_pipeline(args, config)


if __name__ == '__main__':
    main()
