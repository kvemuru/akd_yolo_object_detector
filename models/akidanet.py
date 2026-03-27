"""
AkidaNet Backbone for Object Detection

A lightweight CNN backbone compatible with Akida 1.0 hardware constraints.
Uses separable convolutions for efficient inference.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _make_divisible(value, divisor=8, min_value=None):
    """Ensure value is divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def conv_block(inputs, filters, kernel_size=3, stride=1, use_bn=True, activation='relu', name=''):
    """Standard convolution block with optional batch norm and activation."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=not use_bn,
        name=f'{name}_conv' if name else None
    )(inputs)
    
    if use_bn:
        x = layers.BatchNormalization(name=f'{name}_bn' if name else None)(x)
    
    if activation:
        x = layers.Activation(activation, name=f'{name}_act' if name else None)(x)
    
    return x


def separable_conv_block(inputs, filters, kernel_size=3, stride=1, use_bn=True, activation='relu', name=''):
    """Depthwise separable convolution block - Akida compatible."""
    x = layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=stride,
        padding='same',
        use_bias=not use_bn,
        name=f'{name}_sepconv' if name else None
    )(inputs)
    
    if use_bn:
        x = layers.BatchNormalization(name=f'{name}_bn' if name else None)(x)
    
    if activation:
        x = layers.Activation(activation, name=f'{name}_act' if name else None)(x)
    
    return x


def akidanet_backbone(input_shape, alpha=0.5):
    """
    Create AkidaNet backbone for YOLO detector.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        alpha: Width multiplier for network (0.25, 0.5, 1.0)
    
    Returns:
        Keras model and list of layer outputs for skip connections
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Initial feature extraction
    x = conv_block(inputs, 16, kernel_size=3, stride=2, name='conv0')  # 112x112
    x = conv_block(x, 32, kernel_size=3, stride=1, name='conv1')        # 112x112
    x = conv_block(x, int(64 * alpha), kernel_size=3, stride=2, name='conv2')  # 56x56
    
    # Store for YOLO detection (later stages)
    skip_56 = x  # 56x56 feature map
    
    x = separable_conv_block(x, int(64 * alpha), kernel_size=3, stride=1, name='sep4')  # 56x56
    x = separable_conv_block(x, int(128 * alpha), kernel_size=3, stride=2, name='sep4pw')  # 28x28
    
    # Stage 2: 28x28 resolution
    x = separable_conv_block(x, int(128 * alpha), kernel_size=3, stride=1, name='sep5')  # 28x28
    x = separable_conv_block(x, int(128 * alpha), kernel_size=3, stride=1, name='sep6')  # 28x28
    x = separable_conv_block(x, int(128 * alpha), kernel_size=3, stride=1, name='sep7')  # 28x28
    x = separable_conv_block(x, int(128 * alpha), kernel_size=3, stride=1, name='sep8')  # 28x28
    x = separable_conv_block(x, int(256 * alpha), kernel_size=3, stride=2, name='sep8pw')  # 14x14
    
    skip_28 = x  # 28x28 feature map (optional skip)
    
    # Stage 3: 14x14 resolution
    x = separable_conv_block(x, int(256 * alpha), kernel_size=3, stride=1, name='sep9')   # 14x14
    x = separable_conv_block(x, int(256 * alpha), kernel_size=3, stride=1, name='sep10')  # 14x14
    x = separable_conv_block(x, int(256 * alpha), kernel_size=3, stride=1, name='sep11')  # 14x14
    x = separable_conv_block(x, int(256 * alpha), kernel_size=3, stride=1, name='sep12')  # 14x14
    x = separable_conv_block(x, int(256 * alpha), kernel_size=3, stride=1, name='sep13')  # 14x14
    x = separable_conv_block(x, int(256 * alpha), kernel_size=3, stride=1, name='sep14')  # 14x14
    x = separable_conv_block(x, int(512 * alpha), kernel_size=3, stride=2, name='sep14pw')  # 7x7
    
    skip_14 = x  # 14x14 feature map (optional skip)
    
    # Stage 4: 7x7 resolution (detection scale)
    x = separable_conv_block(x, int(512 * alpha), kernel_size=3, stride=1, name='sep15')  # 7x7
    x = separable_conv_block(x, int(512 * alpha), kernel_size=3, stride=1, name='sep16')  # 7x7
    
    # Additional detection layers
    x = separable_conv_block(x, int(1024 * alpha), kernel_size=3, stride=1, name='det1_dw')  # 7x7
    x = separable_conv_block(x, int(1024 * alpha), kernel_size=1, stride=1, name='det1_pw')   # 7x7
    x = separable_conv_block(x, int(1024 * alpha), kernel_size=1, stride=1, name='det2_dw')  # 7x7
    x = separable_conv_block(x, int(1024 * alpha), kernel_size=1, stride=1, name='det2_pw')   # 7x7
    
    model = keras.Model(inputs=inputs, outputs=x, name='akidanet_backbone')
    
    return model, [skip_56, skip_28, skip_14]


def create_akidanet_backbone_v2(input_shape=(224, 224, 3), alpha=0.5):
    """
    Alternative AkidaNet backbone with explicit layer configuration.
    
    Compatible with Akida 1.0 constraints:
    - Max kernel size: 7x7
    - Stride 2 with 3x3 kernels only
    - SeparableConv2D for efficiency
    """
    inputs = keras.Input(shape=input_shape, name='input_image')
    
    # InputConv layer (Akida compatible)
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False, name='conv0')(inputs)
    x = layers.BatchNormalization(name='conv0_bn')(x)
    x = layers.ReLU(name='conv0_relu')(x)  # 112x112
    
    # Stage 1: 112x112
    x = layers.Conv2D(int(32 * alpha), 3, strides=1, padding='same', use_bias=False, name='conv1')(x)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(name='conv1_relu')(x)  # 112x112
    
    x = layers.Conv2D(int(64 * alpha), 3, strides=2, padding='same', use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='conv2_bn')(x)
    x = layers.ReLU(name='conv2_relu')(x)  # 56x56
    
    # Depthwise separable stages
    # Stage 2: 56->28
    x = separable_conv_block(x, int(64 * alpha), 3, stride=1, name='sep4')
    x = separable_conv_block(x, int(128 * alpha), 3, stride=2, name='sep4pw')  # 28x28
    
    # Stage 3: 28->14
    x = separable_conv_block(x, int(128 * alpha), 3, stride=1, name='sep5')
    x = separable_conv_block(x, int(128 * alpha), 3, stride=1, name='sep6')
    x = separable_conv_block(x, int(128 * alpha), 3, stride=1, name='sep7')
    x = separable_conv_block(x, int(128 * alpha), 3, stride=1, name='sep8')
    x = separable_conv_block(x, int(256 * alpha), 3, stride=2, name='sep8pw')  # 14x14
    
    # Stage 4: 14->7
    x = separable_conv_block(x, int(256 * alpha), 3, stride=1, name='sep9')
    x = separable_conv_block(x, int(256 * alpha), 3, stride=1, name='sep10')
    x = separable_conv_block(x, int(256 * alpha), 3, stride=1, name='sep11')
    x = separable_conv_block(x, int(256 * alpha), 3, stride=1, name='sep12')
    x = separable_conv_block(x, int(256 * alpha), 3, stride=1, name='sep13')
    x = separable_conv_block(x, int(256 * alpha), 3, stride=1, name='sep14')
    x = separable_conv_block(x, int(512 * alpha), 3, stride=2, name='sep14pw')  # 7x7
    
    # Detection stage: 7x7
    x = separable_conv_block(x, int(512 * alpha), 3, stride=1, name='det1')
    x = separable_conv_block(x, int(512 * alpha), 3, stride=1, name='det2')
    
    model = keras.Model(inputs=inputs, outputs=x, name='akidanet_backbone_v2')
    return model


def count_parameters(model):
    """Count trainable and non-trainable parameters."""
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    return trainable, non_trainable, trainable + non_trainable


if __name__ == '__main__':
    # Test the backbone
    model = create_akidanet_backbone_v2(input_shape=(224, 224, 3), alpha=0.5)
    model.summary()
    
    trainable, non_trainable, total = count_parameters(model)
    print(f"\nTrainable: {trainable:,}")
    print(f"Non-trainable: {non_trainable:,}")
    print(f"Total: {total:,}")
