import numpy as np
import cv2
from skimage import feature, filters, color, transform
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.metrics import structural_similarity
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import Concatenate, UpSampling2D, Dropout, Dense, GlobalAveragePooling2D, Reshape, Multiply
import os
from utils import load_ground_truth
from smoke_generation import apply_smoke_effect

# Check if OpenCV contrib modules are available
has_ximgproc = True
try:
    _ = cv2.ximgproc
except AttributeError:
    has_ximgproc = False
    print("Warning: cv2.ximgproc not available. Using fallback implementation for guided filter.")

# Path to save/load the model
EDGE_MODEL_PATH = 'models/edge_detection_model.h5'
SMOKE_LEVEL_PATH = 'models/smoke_level_model.h5'

# Path to save/load the smoke-aware model
SMOKE_AWARE_MODEL_PATH = 'models/smoke_aware_edge_model.h5'

def canny_edge(image, sigma=1.0, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection to an image.
    
    Args:
        image: RGB image
        sigma: Gaussian smoothing parameter
        low_threshold: Low threshold for Canny detection
        high_threshold: High threshold for Canny detection
        
    Returns:
        Binary edge map
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray.astype(np.uint8))
    
    # Apply Gaussian filter to reduce noise
    blurred = gaussian(gray, sigma=sigma)
    
    # Apply Canny edge detection
    edges = canny(blurred, low_threshold=low_threshold, high_threshold=high_threshold)
    
    return edges.astype(np.float32)

def sobel_edge(image, threshold=0.1, ksize=3):
    """Apply Sobel edge detection to an image.
    
    Args:
        image: RGB image
        threshold: Threshold for edge detection
        ksize: Kernel size for Sobel operator
        
    Returns:
        Binary edge map
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray.astype(np.uint8))
    
    # Apply Gaussian filter to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate Sobel gradients
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Calculate gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize
    magnitude = magnitude / magnitude.max()
    
    # Thresholding
    edges = (magnitude > threshold).astype(np.float32)
    
    # Apply morphological operations to clean up edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges

def laplacian_edge(image, threshold=0.1, ksize=3):
    """Apply Laplacian edge detection to an image.
    
    Args:
        image: RGB image
        threshold: Threshold for edge detection
        ksize: Kernel size for Laplacian operator
        
    Returns:
        Binary edge map
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray.astype(np.uint8))
    
    # Apply Gaussian filter to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
    
    # Take absolute value
    laplacian = np.abs(laplacian)
    
    # Normalize
    laplacian = laplacian / laplacian.max()
    
    # Thresholding
    edges = (laplacian > threshold).astype(np.float32)
    
    # Clean up edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges

def box_filter(src, radius):
    """Fast box filter implementation."""
    ksize = 2 * radius + 1
    return cv2.blur(src, (ksize, ksize))

def guided_filter_fallback(guide, src, radius, eps):
    """Fallback implementation of guided filter."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    
    # Step 1: Calculate mean and variance of guide image
    mean_I = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    
    corr_I = box_filter(guide * guide, radius)
    corr_Ip = box_filter(guide * src, radius)
    
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    # Step 2: Calculate a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Step 3: Calculate output
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    
    q = mean_a * guide + mean_b
    
    return q

def guided_filter_edge(image, radius=2, eps=0.01, threshold=0.2):
    """Apply guided filter for edge-preserving smoothing and edge detection.
    
    Args:
        image: RGB image
        radius: Filter radius
        eps: Regularization parameter
        threshold: Threshold for edge detection
        
    Returns:
        Binary edge map
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray.astype(np.uint8))
    
    # Apply guided filter
    if has_ximgproc:
        # Using OpenCV's implementation
        guide = cv2.ximgproc.createGuidedFilter(gray, radius=radius, eps=eps)
        filtered = guide.filter(gray)
    else:
        # Using fallback implementation
        filtered = guided_filter_fallback(gray, gray, radius, eps)
    
    # Calculate difference between original and filtered image
    detail = np.abs(gray.astype(np.float32) - filtered.astype(np.float32))
    
    # Normalize
    max_detail = np.max(detail)
    if max_detail > 0:
        detail = detail / max_detail
    else:
        # If there's no detail, all pixels are the same, so no edges
        return np.zeros_like(detail, dtype=np.float32)
    
    # Thresholding
    edges = (detail > threshold).astype(np.float32)
    
    # Clean up edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges

def enhance_smoky_image(image, smoke_level):
    """Enhance a smoky image based on estimated smoke level.
    
    Args:
        image: RGB image affected by smoke
        smoke_level: Integer from 0 (none) to 4 (extreme)
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space for better color manipulation
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE based on smoke level
    clahe = cv2.createCLAHE(
        clipLimit=min(4.0, 1.0 + smoke_level * 0.75),  # Increase clip limit with smoke level
        tileGridSize=(8, 8)
    )
    enhanced_l = clahe.apply(l)
    
    # For heavy or extreme smoke, apply additional processing
    if smoke_level >= 3:  # Heavy or extreme
        # Increase contrast
        enhanced_l = cv2.convertScaleAbs(enhanced_l, alpha=1.1, beta=0)
        
        # Enhance details
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced_l = cv2.filter2D(enhanced_l, -1, kernel)
    
    # Merge channels and convert back to RGB
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Apply denoising for medium to extreme smoke
    if smoke_level >= 2:  # Medium to extreme
        # Increase strength of denoising based on smoke level
        h = int(6 + smoke_level * 3)  # Convert to int
        enhanced_rgb = cv2.fastNlMeansDenoisingColored(
            enhanced_rgb, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
        )
    
    return enhanced_rgb

def create_dense_edge_detection_model(input_shape=(256, 256, 3)):
    """Create a DenseNet-like model for edge detection.
    
    Args:
        input_shape: Input image shape
        
    Returns:
        Keras model for edge detection
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Encoder path with dense connections
    skip_connections = []
    
    # Level 1
    x1 = Conv2D(64, 3, padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    skip_connections.append(x1)
    x = MaxPooling2D(2)(x1)
    
    # Level 2
    x2 = Conv2D(128, 3, padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv2D(128, 3, padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    skip_connections.append(x2)
    x = MaxPooling2D(2)(x2)
    
    # Level 3
    x3 = Conv2D(256, 3, padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.3)(x3)
    x3 = Conv2D(256, 3, padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    skip_connections.append(x3)
    x = MaxPooling2D(2)(x3)
    
    # Bridge
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Decoder path with dense connections
    skip_connections = skip_connections[::-1]  # Reverse for easy indexing
    
    # Level 3
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_connections[0]])
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Level 2
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_connections[1]])
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Level 1
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_connections[2]])
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_edge_detection_model(train_images, train_gt, val_images=None, val_gt=None, 
                              epochs=20, batch_size=8, smoke_augmentation=True):
    """Train the deep edge detection model.
    
    Args:
        train_images: List of training image paths
        train_gt: List of ground truth edge map paths
        val_images: List of validation image paths (optional)
        val_gt: List of validation ground truth edge map paths (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        smoke_augmentation: Whether to apply smoke augmentation during training
        
    Returns:
        Trained model
    """
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from smoke_generation import apply_smoke_effect
    
    # Create model
    model = create_dense_edge_detection_model()
    
    # Load and preprocess data
    X_train = []
    y_train = []
    
    print(f"Loading and processing {len(train_images)} training images...")
    
    for i, (img_path, gt_path) in enumerate(zip(train_images, train_gt)):
        if i % 20 == 0:
            print(f"Processing image {i}/{len(train_images)}")
            
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load and preprocess ground truth
        gt = load_ground_truth(gt_path)
        if gt is None:
            print(f"Failed to load ground truth: {gt_path}")
            continue
            
        # Resize images to model input size
        img_resized = cv2.resize(img, (256, 256))
        gt_resized = cv2.resize(gt, (256, 256))
        
        # Normalize to 0-1 range
        img_norm = img_resized.astype(np.float32) / 255.0
        gt_norm = gt_resized.astype(np.float32) / 255.0
        gt_norm = np.expand_dims(gt_norm, axis=-1)  # Add channel dimension
        
        # Add the original clean image and its ground truth
        X_train.append(img_norm)
        y_train.append(gt_norm)
        
        # If smoke augmentation is enabled, add smoky versions of the image
        if smoke_augmentation:
            # Generate different smoke levels
            smoke_levels = [1, 2, 3, 4]  # light, medium, heavy, extreme
            smoke_methods = ['perlin', 'gaussian', 'texture']
            
            # Add smoky versions with the same ground truth
            for level in smoke_levels:
                for method in smoke_methods[:1]:  # Use just one method to avoid too many images
                    # Apply smoke effect
                    smoky_img = apply_smoke_effect(img, level, method)
                    
                    # Resize and normalize
                    smoky_resized = cv2.resize(smoky_img, (256, 256))
                    smoky_norm = smoky_resized.astype(np.float32) / 255.0
                    
                    # Add to training data
                    X_train.append(smoky_norm)
                    y_train.append(gt_norm)  # Same ground truth
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Created training dataset with {len(X_train)} images")
    
    # Prepare validation data if provided
    validation_data = None
    if val_images and val_gt:
        X_val = []
        y_val = []
        
        print(f"Loading and processing {len(val_images)} validation images...")
        
        for img_path, gt_path in zip(val_images, val_gt):
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load and preprocess ground truth
            gt = load_ground_truth(gt_path)
            if gt is None:
                continue
                
            # Resize and normalize
            img_resized = cv2.resize(img, (256, 256))
            gt_resized = cv2.resize(gt, (256, 256))
            
            img_norm = img_resized.astype(np.float32) / 255.0
            gt_norm = gt_resized.astype(np.float32) / 255.0
            gt_norm = np.expand_dims(gt_norm, axis=-1)
            
            X_val.append(img_norm)
            y_val.append(gt_norm)
            
            # Add some smoky validation samples
            if smoke_augmentation:
                # Add one smoky version for validation
                smoky_img = apply_smoke_effect(img, 2, 'perlin')  # Medium smoke
                smoky_resized = cv2.resize(smoky_img, (256, 256))
                smoky_norm = smoky_resized.astype(np.float32) / 255.0
                
                X_val.append(smoky_norm)
                y_val.append(gt_norm)
        
        if X_val:
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            validation_data = (X_val, y_val)
            print(f"Created validation dataset with {len(X_val)} images")
    
    # Set up data augmentation for clean images
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(EDGE_MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train model
    print(f"Training edge detection model with {len(X_train)} images...")
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        steps_per_epoch=len(X_train) // batch_size
    )
    
    # Load best model weights
    if os.path.exists(EDGE_MODEL_PATH):
        model = load_model(EDGE_MODEL_PATH)
    
    return model, history

def deep_edge_detection(image, model=None, threshold=0.5):
    """Apply deep learning-based edge detection to an image.
    
    Args:
        image: RGB or grayscale image
        model: Pre-trained edge detection model (will load from disk if None)
        threshold: Threshold for edge detection
        
    Returns:
        Binary edge map
    """
    # Load model if not provided
    if model is None:
        if os.path.exists(EDGE_MODEL_PATH):
            model = load_model(EDGE_MODEL_PATH)
        else:
            print("Error: No edge detection model found. Please train the model first.")
            return None
    
    # Preprocess input image
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] > 3:
        # Convert RGBA to RGB
        image = image[:, :, :3]
    
    # Resize to model input size
    original_size = image.shape[:2]
    input_image = cv2.resize(image, (256, 256))
    input_image = input_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    
    # Predict edges
    edge_pred = model.predict(input_image)[0, :, :, 0]
    
    # Resize back to original size
    edge_pred = cv2.resize(edge_pred, (original_size[1], original_size[0]))
    
    # Thresholding for binary edges
    edge_binary = (edge_pred > threshold).astype(np.float32)
    
    return edge_binary

def hybrid_edge_detection(image, deep_model=None):
    """Apply hybrid edge detection using both deep learning and traditional methods.
    
    This approach combines deep learning edge detection with traditional methods for
    improved accuracy. Based on "Hybrid Image Edge Detection Algorithm Based on
    Fractional Differential and Canny Operator" paper.
    
    Args:
        image: RGB image
        deep_model: Pre-trained deep learning model (optional)
        
    Returns:
        Binary edge map
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply deep learning edge detection
    deep_edges = deep_edge_detection(image, model=deep_model)
    
    if deep_edges is None:
        # Fallback to traditional methods if deep model is not available
        return sobel_edge(image, threshold=0.15)
    
    # Apply Canny with optimized parameters based on research
    canny_edges = canny_edge(image, sigma=1.3, low_threshold=30, high_threshold=100)
    
    # Apply multi-scale Sobel for texture details
    sobel_edges = sobel_edge(image, threshold=0.12, ksize=3)
    
    # Combine edges using weighted fusion
    combined_edges = 0.6 * deep_edges + 0.2 * canny_edges + 0.2 * sobel_edges
    
    # Apply non-maximum suppression and hysteresis thresholding to refine edges
    refined_edges = apply_non_maximum_suppression(combined_edges, gray)
    
    return refined_edges

def apply_non_maximum_suppression(edge_map, gray_image):
    """Apply non-maximum suppression to refine edges.
    
    Args:
        edge_map: Edge probability map
        gray_image: Grayscale image for gradient calculation
        
    Returns:
        Refined edge map
    """
    # Calculate gradients
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi
    
    # Quantize the direction into one of 4 directions (0, 45, 90, 135 degrees)
    direction_quantized = np.zeros_like(direction, dtype=np.uint8)
    direction_quantized[(direction >= -22.5) & (direction < 22.5)] = 0  # 0 degrees (horizontal)
    direction_quantized[(direction >= 22.5) & (direction < 67.5)] = 1   # 45 degrees
    direction_quantized[(direction >= 67.5) & (direction < 112.5)] = 2  # 90 degrees (vertical)
    direction_quantized[(direction >= 112.5) & (direction < 157.5)] = 3 # 135 degrees
    direction_quantized[(direction >= -67.5) & (direction < -22.5)] = 1
    direction_quantized[(direction >= -112.5) & (direction < -67.5)] = 2
    direction_quantized[(direction >= -157.5) & (direction < -112.5)] = 3
    direction_quantized[(direction >= 157.5) | (direction < -157.5)] = 0
    
    # Apply non-maximum suppression
    height, width = edge_map.shape
    suppressed = np.zeros((height, width), dtype=np.float32)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Skip if edge strength is already 0
            if edge_map[i, j] == 0:
                continue
                
            # Get the direction
            d = direction_quantized[i, j]
            
            # Check if the current pixel is a local maximum
            if d == 0:  # Horizontal
                if edge_map[i, j] >= edge_map[i, j-1] and edge_map[i, j] >= edge_map[i, j+1]:
                    suppressed[i, j] = edge_map[i, j]
            elif d == 1:  # 45 degrees
                if edge_map[i, j] >= edge_map[i-1, j+1] and edge_map[i, j] >= edge_map[i+1, j-1]:
                    suppressed[i, j] = edge_map[i, j]
            elif d == 2:  # Vertical
                if edge_map[i, j] >= edge_map[i-1, j] and edge_map[i, j] >= edge_map[i+1, j]:
                    suppressed[i, j] = edge_map[i, j]
            elif d == 3:  # 135 degrees
                if edge_map[i, j] >= edge_map[i-1, j-1] and edge_map[i, j] >= edge_map[i+1, j+1]:
                    suppressed[i, j] = edge_map[i, j]
    
    # Apply hysteresis thresholding
    high_threshold = 0.2
    low_threshold = 0.1
    
    strong_edges = (suppressed > high_threshold).astype(np.uint8)
    weak_edges = ((suppressed > low_threshold) & (suppressed <= high_threshold)).astype(np.uint8)
    
    # Use connected components to link weak edges to strong edges
    labeled_strong, num_strong = ndimage.label(strong_edges)
    final_edges = np.copy(strong_edges)
    
    for i in range(1, num_strong+1):
        strong_region = (labeled_strong == i)
        dilated_region = ndimage.binary_dilation(strong_region, structure=np.ones((3, 3)))
        connected_weak = dilated_region & weak_edges
        final_edges = final_edges | connected_weak
    
    return final_edges.astype(np.float32)

def rethink_canny_edge_detection(image, adaptive_thresholds=True):
    """Improved Canny edge detection using adaptive thresholds and FCM in NSST domain.
    
    Based on "Image Edge Detection Based on FCM and Improved Canny Operator in NSST Domain" paper.
    
    Args:
        image: RGB or grayscale image
        adaptive_thresholds: Whether to use adaptive thresholding
        
    Returns:
        Binary edge map
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray.astype(np.uint8))
    
    # Apply bilateral filter for edge-preserving smoothing
    smoothed = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # Compute gradients using Sobel operators with optimal kernel size
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi
    
    # Normalize magnitude
    magnitude_norm = magnitude / magnitude.max()
    
    # Determine thresholds adaptively using Otsu's method if requested
    if adaptive_thresholds:
        # Convert to 8-bit for Otsu
        mag_8bit = (magnitude_norm * 255).astype(np.uint8)
        high_threshold, _ = cv2.threshold(mag_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_threshold = high_threshold / 255.0
        low_threshold = high_threshold * 0.4
    else:
        # Fixed thresholds
        high_threshold = 0.2
        low_threshold = 0.08
    
    # Apply non-maximum suppression and hysteresis thresholding
    # Non-maximum suppression
    height, width = magnitude_norm.shape
    suppressed = np.zeros((height, width), dtype=np.float32)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Skip if edge strength is already 0
            if magnitude_norm[i, j] == 0:
                continue
                
            # Get the gradient direction and round to one of four angles (0, 45, 90, 135)
            angle = direction[i, j]
            if (angle < 0):
                angle += 180
                
            # Check if the current pixel is a local maximum
            if (angle <= 22.5 or angle > 157.5):  # Horizontal (0 degrees)
                if magnitude_norm[i, j] >= magnitude_norm[i, j-1] and magnitude_norm[i, j] >= magnitude_norm[i, j+1]:
                    suppressed[i, j] = magnitude_norm[i, j]
            elif (angle > 22.5 and angle <= 67.5):  # 45 degrees
                if magnitude_norm[i, j] >= magnitude_norm[i-1, j+1] and magnitude_norm[i, j] >= magnitude_norm[i+1, j-1]:
                    suppressed[i, j] = magnitude_norm[i, j]
            elif (angle > 67.5 and angle <= 112.5):  # Vertical (90 degrees)
                if magnitude_norm[i, j] >= magnitude_norm[i-1, j] and magnitude_norm[i, j] >= magnitude_norm[i+1, j]:
                    suppressed[i, j] = magnitude_norm[i, j]
            else:  # 135 degrees
                if magnitude_norm[i, j] >= magnitude_norm[i-1, j-1] and magnitude_norm[i, j] >= magnitude_norm[i+1, j+1]:
                    suppressed[i, j] = magnitude_norm[i, j]
    
    # Hysteresis thresholding
    strong_edges = (suppressed > high_threshold).astype(np.uint8)
    weak_edges = ((suppressed > low_threshold) & (suppressed <= high_threshold)).astype(np.uint8)
    
    # Use connected components to link weak edges to strong edges
    labeled_strong, num_strong = ndimage.label(strong_edges)
    final_edges = np.copy(strong_edges)
    
    for i in range(1, num_strong+1):
        strong_region = (labeled_strong == i)
        dilated_region = ndimage.binary_dilation(strong_region, structure=np.ones((3, 3)))
        connected_weak = dilated_region & weak_edges
        final_edges = final_edges | connected_weak
    
    # Post-processing to remove isolated pixels and fill small gaps
    kernel = np.ones((3, 3), np.uint8)
    final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, kernel)
    
    return final_edges.astype(np.float32)

def adaptive_edge_detection(image, smoke_level):
    """Apply adaptive edge detection based on smoke level.
    
    Args:
        image: Input image
        smoke_level: Predicted smoke level (0-4)
    
    Returns:
        Binary edge map
    """
    # Convert smoke_level to string if it's a number
    if isinstance(smoke_level, (int, float)):
        smoke_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
        if 0 <= smoke_level < len(smoke_levels):
            smoke_level = smoke_levels[int(smoke_level)]
        else:
            smoke_level = 'medium'  # Default to medium if out of range
    
    # Check if deep learning model is available
    has_deep_model = os.path.exists(EDGE_MODEL_PATH)
    
    # For no smoke or light smoke, use hybrid method or traditional methods
    if smoke_level == 'none':
        if has_deep_model:
            return deep_edge_detection(image, threshold=0.4)
        else:
            return rethink_canny_edge_detection(image, adaptive_thresholds=True)
    
    elif smoke_level == 'light':
        # Light smoke - use deep learning with preprocessing
        if has_deep_model:
            # Enhance contrast before edge detection
            enhanced = enhance_smoky_image(image, 1)
            return deep_edge_detection(enhanced, threshold=0.35)
        else:
            # Use traditional methods with preprocessing
            enhanced = enhance_smoky_image(image, 1)
            return rethink_canny_edge_detection(enhanced, adaptive_thresholds=True)
    
    elif smoke_level == 'medium':
        # Medium smoke - heavy preprocessing with hybrid approach
        enhanced = enhance_smoky_image(image, 2)
        if has_deep_model:
            return hybrid_edge_detection(enhanced)
        else:
            return rethink_canny_edge_detection(enhanced, adaptive_thresholds=True)
    
    elif smoke_level == 'heavy':
        # Heavy smoke - use deep learning with stronger preprocessing
        enhanced = enhance_smoky_image(image, 3)
        if has_deep_model:
            # Lower threshold to improve recall
            return deep_edge_detection(enhanced, threshold=0.25)
        else:
            # Use combination of methods
            canny_edges = rethink_canny_edge_detection(enhanced, adaptive_thresholds=True)
            sobel_edges = sobel_edge(enhanced, threshold=0.1, ksize=5)
            return np.maximum(canny_edges, sobel_edges)
    
    elif smoke_level == 'extreme':
        # Extreme smoke - maximum enhancement and multi-scale approach
        enhanced = enhance_smoky_image(image, 4)
        
        if has_deep_model:
            # Use deep learning with lowest threshold
            return deep_edge_detection(enhanced, threshold=0.2)
        else:
            # Multi-scale approach
            canny_edges = rethink_canny_edge_detection(enhanced, adaptive_thresholds=True)
            sobel_edges1 = sobel_edge(enhanced, threshold=0.08, ksize=3)
            sobel_edges2 = sobel_edge(enhanced, threshold=0.12, ksize=5)
            sobel_edges3 = sobel_edge(enhanced, threshold=0.15, ksize=7)
            
            # Combine all methods (take maximum response)
            combined_edges = np.maximum(canny_edges, 
                                       np.maximum(sobel_edges1, 
                                                 np.maximum(sobel_edges2, sobel_edges3)))
            
            # Final refinement
            kernel = np.ones((3, 3), np.uint8)
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
            
            return combined_edges
    
    else:
        # Default to Sobel if smoke level is unknown
        return sobel_edge(image, threshold=0.1, ksize=3)

def thin_edges(edge_map, max_thickness=1):
    """Thin edges to improve precision.
    
    Args:
        edge_map: Binary edge map
        max_thickness: Maximum edge thickness
        
    Returns:
        Thinned edge map
    """
    # Ensure binary
    binary = (edge_map > 0.5).astype(np.uint8)
    
    # Apply skeletonization
    skeleton = ndimage.morphology.binary_thinning(binary)
    
    # If needed, dilate slightly to restore some thickness
    if max_thickness > 1:
        kernel = np.ones((max_thickness, max_thickness), np.uint8)
        skeleton = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
    
    return skeleton.astype(np.float32)

def visualize_edge_detection(image, edges, title="Edge Detection"):
    """Visualize edge detection results.
    
    Args:
        image: Input RGB image
        edges: Edge map
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def canny_edge_detection(image, threshold1=100, threshold2=200):
    """Apply Canny edge detection algorithm.
    
    Args:
        image: Input image (grayscale)
        threshold1: First threshold for the hysteresis procedure
        threshold2: Second threshold for the hysteresis procedure
    
    Returns:
        Binary edge map
    """
    # Convert to grayscale if the image is RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # Normalize to [0, 1] range
    edges = edges / 255.0
    
    return edges

def guided_filter_edge_detection(image, radius=8, eps=1e-6):
    """Apply guided filter for edge-preserving smoothing and detect edges.
    
    Args:
        image: Input image
        radius: Filter radius
        eps: Regularization parameter
    
    Returns:
        Binary edge map
    """
    # Convert to grayscale if the image is RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Convert to float32
    gray = gray.astype(np.float32) / 255.0
    
    # Apply guided filter
    guided = cv2.ximgproc.guidedFilter(gray, gray, radius, eps)
    
    # Calculate difference between original and filtered image to get edges
    edges = np.abs(gray - guided)
    
    # Normalize and threshold
    edges = edges / np.max(edges) if np.max(edges) > 0 else edges
    edges = (edges > 0.05).astype(np.float32)
    
    # Use morphological operations to clean up the edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges

def sobel_edge_detection(image, ksize=3, threshold=0.1):
    """Apply Sobel edge detection algorithm.
    
    Args:
        image: Input image
        ksize: Size of Sobel kernel
        threshold: Threshold for edge detection
    
    Returns:
        Binary edge map
    """
    return sobel_edge(image, threshold, ksize)

def create_smoke_aware_edge_detection_model(input_shape=(256, 256, 3)):
    """Create a model for edge detection that takes smoke level into account.
    
    This model has two inputs:
    1. The image
    2. The smoke level (one-hot encoded)
    
    Args:
        input_shape: Input image shape
        
    Returns:
        Keras model for edge detection
    """
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply

    # Image input
    img_input = Input(shape=input_shape, name='image_input')
    
    # Smoke level input (5 levels: none, light, medium, heavy, extreme)
    smoke_level_input = Input(shape=(5,), name='smoke_level_input')
    
    # Initial convolution
    x = Conv2D(64, 3, padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Encoder path with dense connections
    skip_connections = []
    
    # Level 1
    x1 = Conv2D(64, 3, padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    skip_connections.append(x1)
    x = MaxPooling2D(2)(x1)
    
    # Level 2
    x2 = Conv2D(128, 3, padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv2D(128, 3, padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    skip_connections.append(x2)
    x = MaxPooling2D(2)(x2)
    
    # Level 3
    x3 = Conv2D(256, 3, padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.3)(x3)
    x3 = Conv2D(256, 3, padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    skip_connections.append(x3)
    x = MaxPooling2D(2)(x3)
    
    # Bridge
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    # Incorporate smoke level information
    # Convert smoke level to feature modulation
    smoke_features = Dense(512, activation='relu')(smoke_level_input)
    smoke_features = Dense(512, activation='sigmoid')(smoke_features)
    smoke_features = Reshape((1, 1, 512))(smoke_features)
    
    # Apply feature modulation (adaptive scaling based on smoke level)
    x = Multiply()([x, smoke_features])
    
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Decoder path with dense connections
    skip_connections = skip_connections[::-1]  # Reverse for easy indexing
    
    # Level 3
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_connections[0]])
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Level 2
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_connections[1]])
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Level 1
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_connections[2]])
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    
    # Create model with two inputs
    model = Model(inputs=[img_input, smoke_level_input], outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_smoke_aware_edge_detection_model(train_images, train_gt, val_images=None, val_gt=None, 
                                         epochs=20, batch_size=8):
    """Train the smoke-aware edge detection model.
    
    Args:
        train_images: List of training image paths
        train_gt: List of ground truth edge map paths
        val_images: List of validation image paths (optional)
        val_gt: List of validation ground truth edge map paths (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from smoke_generation import apply_smoke_effect
    
    # Create model
    model = create_smoke_aware_edge_detection_model()
    
    # Load and preprocess data
    X_images = []
    X_smoke_levels = []
    y_gt = []
    
    print(f"Loading and processing {len(train_images)} training images with smoke augmentation...")
    
    for i, (img_path, gt_path) in enumerate(zip(train_images, train_gt)):
        if i % 20 == 0:
            print(f"Processing image {i}/{len(train_images)}")
            
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load and preprocess ground truth
        gt = load_ground_truth(gt_path)
        if gt is None:
            print(f"Failed to load ground truth: {gt_path}")
            continue
            
        # Resize images to model input size
        img_resized = cv2.resize(img, (256, 256))
        gt_resized = cv2.resize(gt, (256, 256))
        
        # Normalize to 0-1 range
        img_norm = img_resized.astype(np.float32) / 255.0
        gt_norm = gt_resized.astype(np.float32) / 255.0
        gt_norm = np.expand_dims(gt_norm, axis=-1)  # Add channel dimension
        
        # Add the original clean image (smoke level 0 - none)
        X_images.append(img_norm)
        X_smoke_levels.append([1, 0, 0, 0, 0])  # One-hot encoding for 'none'
        y_gt.append(gt_norm)
        
        # Generate different smoke levels
        smoke_levels = [1, 2, 3, 4]  # light, medium, heavy, extreme
        
        # Create one-hot encodings for each level
        smoke_level_encodings = {
            1: [0, 1, 0, 0, 0],  # light
            2: [0, 0, 1, 0, 0],  # medium
            3: [0, 0, 0, 1, 0],  # heavy
            4: [0, 0, 0, 0, 1]   # extreme
        }
        
        # Add smoky versions with the same ground truth
        for level in smoke_levels:
            # Apply smoke effect with perlin noise
            smoky_img = apply_smoke_effect(img, level, 'perlin')
            
            # Resize and normalize
            smoky_resized = cv2.resize(smoky_img, (256, 256))
            smoky_norm = smoky_resized.astype(np.float32) / 255.0
            
            # Add to training data with appropriate smoke level
            X_images.append(smoky_norm)
            X_smoke_levels.append(smoke_level_encodings[level])
            y_gt.append(gt_norm)  # Same ground truth
    
    X_images = np.array(X_images)
    X_smoke_levels = np.array(X_smoke_levels)
    y_gt = np.array(y_gt)
    
    print(f"Created training dataset with {len(X_images)} images")
    
    # Prepare validation data if provided
    validation_data = None
    if val_images and val_gt:
        val_X_images = []
        val_X_smoke_levels = []
        val_y_gt = []
        
        print(f"Loading and processing {len(val_images)} validation images...")
        
        for img_path, gt_path in zip(val_images, val_gt):
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load and preprocess ground truth
            gt = load_ground_truth(gt_path)
            if gt is None:
                continue
                
            # Resize and normalize
            img_resized = cv2.resize(img, (256, 256))
            gt_resized = cv2.resize(gt, (256, 256))
            
            img_norm = img_resized.astype(np.float32) / 255.0
            gt_norm = gt_resized.astype(np.float32) / 255.0
            gt_norm = np.expand_dims(gt_norm, axis=-1)
            
            # Add clean image
            val_X_images.append(img_norm)
            val_X_smoke_levels.append([1, 0, 0, 0, 0])  # none
            val_y_gt.append(gt_norm)
            
            # Add one version with medium smoke
            smoky_img = apply_smoke_effect(img, 2, 'perlin')
            smoky_resized = cv2.resize(smoky_img, (256, 256))
            smoky_norm = smoky_resized.astype(np.float32) / 255.0
            
            val_X_images.append(smoky_norm)
            val_X_smoke_levels.append([0, 0, 1, 0, 0])  # medium
            val_y_gt.append(gt_norm)
        
        if val_X_images:
            val_X_images = np.array(val_X_images)
            val_X_smoke_levels = np.array(val_X_smoke_levels)
            val_y_gt = np.array(val_y_gt)
            validation_data = ({'image_input': val_X_images, 'smoke_level_input': val_X_smoke_levels}, val_y_gt)
            print(f"Created validation dataset with {len(val_X_images)} images")
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(SMOKE_AWARE_MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train model
    print(f"Training smoke-aware edge detection model with {len(X_images)} images...")
    history = model.fit(
        {'image_input': X_images, 'smoke_level_input': X_smoke_levels},
        y_gt,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        batch_size=batch_size
    )
    
    # Load best model weights
    if os.path.exists(SMOKE_AWARE_MODEL_PATH):
        model = load_model(SMOKE_AWARE_MODEL_PATH)
    
    return model, history

def smoke_aware_edge_detection(image, smoke_level, model=None):
    """Apply smoke-aware edge detection to an image.
    
    Args:
        image: RGB or grayscale image
        smoke_level: Smoke level (0-4) or name ('none', 'light', etc.)
        model: Pre-trained edge detection model (will load from disk if None)
        
    Returns:
        Binary edge map
    """
    # Load model if not provided
    if model is None:
        if os.path.exists(SMOKE_AWARE_MODEL_PATH):
            model = load_model(SMOKE_AWARE_MODEL_PATH)
        else:
            print("Error: No smoke-aware edge detection model found. Please train the model first.")
            return None
    
    # Convert smoke level to string if it's a number
    if isinstance(smoke_level, (int, float)):
        smoke_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
        if 0 <= smoke_level < len(smoke_levels):
            smoke_level_name = smoke_levels[int(smoke_level)]
        else:
            smoke_level_name = 'medium'  # Default
    else:
        smoke_level_name = smoke_level
    
    # Convert smoke level name to one-hot encoding
    smoke_level_encoding = np.zeros(5)
    if smoke_level_name == 'none':
        smoke_level_encoding[0] = 1
    elif smoke_level_name == 'light':
        smoke_level_encoding[1] = 1
    elif smoke_level_name == 'medium':
        smoke_level_encoding[2] = 1
    elif smoke_level_name == 'heavy':
        smoke_level_encoding[3] = 1
    elif smoke_level_name == 'extreme':
        smoke_level_encoding[4] = 1
    
    # Preprocess input image
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] > 3:
        # Convert RGBA to RGB
        image = image[:, :, :3]
    
    # Resize to model input size
    original_size = image.shape[:2]
    input_image = cv2.resize(image, (256, 256))
    input_image = input_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    smoke_level_encoding = np.expand_dims(smoke_level_encoding, axis=0)
    
    # Predict edges
    edge_pred = model.predict({
        'image_input': input_image, 
        'smoke_level_input': smoke_level_encoding
    })[0, :, :, 0]
    
    # Resize back to original size
    edge_pred = cv2.resize(edge_pred, (original_size[1], original_size[0]))
    
    # Thresholding for binary edges
    edge_binary = (edge_pred > 0.5).astype(np.float32)
    
    return edge_binary 