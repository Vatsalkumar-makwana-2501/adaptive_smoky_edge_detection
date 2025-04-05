import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Path to save the smoke removal model
SMOKE_REMOVAL_MODEL_PATH = 'models/smoke_removal_model.h5'

def create_conv_block(x, filters, kernel_size=3, stride=1, padding='same', use_bn=True, activation='relu'):
    """Create a convolutional block with optional batch normalization and activation."""
    x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    
    if use_bn:
        x = BatchNormalization()(x)
        
    if activation:
        x = Activation(activation)(x)
        
    return x

def create_residual_block(x, filters):
    """Create a residual block with two convolutional layers."""
    shortcut = x
    
    x = create_conv_block(x, filters)
    x = create_conv_block(x, filters, activation=None)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def create_attention_block(x, filters):
    """Create a channel attention block for feature recalibration."""
    shortcut = x
    
    # Global average pooling
    pooled = GlobalAveragePooling2D()(x)
    
    # FC layers for channel attention
    pooled = Dense(filters // 4, activation='relu')(pooled)
    pooled = Dense(filters, activation='sigmoid')(pooled)
    
    # Reshape to match the channels
    pooled = Reshape((1, 1, filters))(pooled)
    
    # Apply attention
    x = Multiply()([x, pooled])
    
    # Residual connection
    x = Add()([x, shortcut])
    
    return x

def create_smoke_removal_model(input_shape=(256, 256, 3)):
    """Create a structure representation network for smoke removal.
    
    Inspired by the paper: "Structure Representation Network and Uncertainty 
    Feedback Learning for Dense Non-Uniform Fog Removal"
    """
    inputs = Input(shape=input_shape)
    
    # Initial feature extraction
    x = create_conv_block(inputs, 64)
    
    # First level features
    level1 = create_conv_block(x, 64)
    level1 = create_residual_block(level1, 64)
    level1 = create_attention_block(level1, 64)
    
    # Encoder: downsample
    down1 = create_conv_block(level1, 128, stride=2)
    level2 = create_residual_block(down1, 128)
    level2 = create_attention_block(level2, 128)
    
    # Deeper features
    down2 = create_conv_block(level2, 256, stride=2)
    level3 = create_residual_block(down2, 256)
    level3 = create_residual_block(level3, 256)
    level3 = create_attention_block(level3, 256)
    
    # Decoder: upsample with skip connections
    up1 = tf.keras.layers.UpSampling2D()(level3)
    up1 = create_conv_block(up1, 128)  # Adjust channel size to match level2
    up1 = Concatenate()([up1, level2])
    # After concatenation, we have 256 channels (128 from up1 + 128 from level2)
    up1 = create_conv_block(up1, 128)  # Reduce channels back to 128
    up1 = create_residual_block(up1, 128)
    
    up2 = tf.keras.layers.UpSampling2D()(up1)
    up2 = create_conv_block(up2, 64)  # Adjust channel size to match level1
    up2 = Concatenate()([up2, level1])
    # After concatenation, we have 128 channels (64 from up2 + 64 from level1)
    up2 = create_conv_block(up2, 64)  # Reduce channels back to 64
    up2 = create_residual_block(up2, 64)
    
    # Final reconstruction
    outputs = Conv2D(3, 3, padding='same', activation='sigmoid')(up2)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    # Use the actual MSE loss function instead of string 'mse'
    model.compile(optimizer=Adam(learning_rate=0.0002), loss=MeanSquaredError())
    
    return model

def train_smoke_removal_model(train_hazy_paths, train_clean_paths, 
                            val_hazy_paths=None, val_clean_paths=None,
                            epochs=50, batch_size=4):
    """Train the smoke removal model.
    
    Args:
        train_hazy_paths: List of paths to hazy/smoky training images
        train_clean_paths: List of paths to corresponding clean training images
        val_hazy_paths: List of paths to hazy/smoky validation images
        val_clean_paths: List of paths to corresponding clean validation images
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model and training history
    """
    # Create model directories if they don't exist
    os.makedirs(os.path.dirname(SMOKE_REMOVAL_MODEL_PATH), exist_ok=True)
    
    # Remove existing model file if it exists to avoid loading issues
    if os.path.exists(SMOKE_REMOVAL_MODEL_PATH):
        try:
            os.remove(SMOKE_REMOVAL_MODEL_PATH)
            print(f"Removed existing model at {SMOKE_REMOVAL_MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not remove existing model: {e}")
    
    # Create model
    input_shape = (256, 256, 3)  # Default input shape
    model = create_smoke_removal_model(input_shape)
    
    # Create dataset
    def load_and_preprocess_image(hazy_path, clean_path):
        # Load images
        hazy_img = cv2.imread(hazy_path)
        clean_img = cv2.imread(clean_path)
        
        if hazy_img is None or clean_img is None:
            print(f"Failed to load image: {hazy_path} or {clean_path}")
            return None, None
        
        # Convert BGR to RGB
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        
        # Resize
        hazy_img = cv2.resize(hazy_img, (input_shape[1], input_shape[0]))
        clean_img = cv2.resize(clean_img, (input_shape[1], input_shape[0]))
        
        # Normalize to [0, 1]
        hazy_img = hazy_img.astype(np.float32) / 255.0
        clean_img = clean_img.astype(np.float32) / 255.0
        
        return hazy_img, clean_img
    
    # Create training dataset
    print("Loading training data...")
    X_train = []
    y_train = []
    
    for hazy_path, clean_path in zip(train_hazy_paths, train_clean_paths):
        hazy_img, clean_img = load_and_preprocess_image(hazy_path, clean_path)
        if hazy_img is not None and clean_img is not None:
            X_train.append(hazy_img)
            y_train.append(clean_img)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Create validation dataset if provided
    validation_data = None
    if val_hazy_paths is not None and val_clean_paths is not None:
        print("Loading validation data...")
        X_val = []
        y_val = []
        
        for hazy_path, clean_path in zip(val_hazy_paths, val_clean_paths):
            hazy_img, clean_img = load_and_preprocess_image(hazy_path, clean_path)
            if hazy_img is not None and clean_img is not None:
                X_val.append(hazy_img)
                y_val.append(clean_img)
        
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        validation_data = (X_val, y_val)
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            SMOKE_REMOVAL_MODEL_PATH,
            save_best_only=True,
            monitor='val_loss' if validation_data else 'loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    # Train model
    print("Training smoke removal model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    
    # Create a new model instance for safety
    best_model = create_smoke_removal_model(input_shape)
    
    # Load the best weights from saved model if it exists
    if os.path.exists(SMOKE_REMOVAL_MODEL_PATH):
        best_model.load_weights(SMOKE_REMOVAL_MODEL_PATH)
        print(f"Loaded best weights from {SMOKE_REMOVAL_MODEL_PATH}")
    else:
        best_model = model
    
    return best_model, history

def remove_smoke(image, model=None):
    """Remove smoke/fog from an image.
    
    Args:
        image: Input smoky/hazy image (RGB or BGR)
        model: Pre-trained smoke removal model. If None, try to load from default path.
        
    Returns:
        Clean image with smoke removed
    """
    # Load model if not provided
    if model is None:
        if os.path.exists(SMOKE_REMOVAL_MODEL_PATH):
            try:
                # Create a fresh model and load weights directly
                input_shape = (256, 256, 3)
                model = create_smoke_removal_model(input_shape)
                model.load_weights(SMOKE_REMOVAL_MODEL_PATH)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using original image.")
                return image
        else:
            print("No smoke removal model found. Using original image.")
            return image
    
    # Handle string path input
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Store original image size
    original_size = image.shape[:2]
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        is_bgr = True
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        is_bgr = False
        image_rgb = image
    
    # Resize for model input
    image_resized = cv2.resize(image_rgb, (256, 256))
    
    # Normalize
    image_norm = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_norm, axis=0)
    
    # Predict
    clean_image = model.predict(image_batch)[0]
    
    # Rescale to 0-255
    clean_image = (clean_image * 255.0).astype(np.uint8)
    
    # Resize back to original size
    clean_image = cv2.resize(clean_image, (original_size[1], original_size[0]))
    
    # Convert back to BGR if the input was BGR
    if is_bgr:
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR)
    
    return clean_image

def load_smoke_dataset(dataset_dir):
    """Load smoke dataset containing hazy and clean image pairs.
    
    Args:
        dataset_dir: Directory containing 'train' and 'test' folders with 'hazy' and 'clean' subfolders
        
    Returns:
        Dictionary with train and test hazy/clean image paths
    """
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    train_hazy_dir = os.path.join(train_dir, 'hazy')
    train_clean_dir = os.path.join(train_dir, 'clean')
    test_hazy_dir = os.path.join(test_dir, 'hazy')
    test_clean_dir = os.path.join(test_dir, 'clean')
    
    # Get file lists
    train_hazy_files = sorted([os.path.join(train_hazy_dir, f) for f in os.listdir(train_hazy_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))])
    
    train_clean_files = sorted([os.path.join(train_clean_dir, f) for f in os.listdir(train_clean_dir) 
                               if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))])
    
    test_hazy_files = sorted([os.path.join(test_hazy_dir, f) for f in os.listdir(test_hazy_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))])
    
    test_clean_files = sorted([os.path.join(test_clean_dir, f) for f in os.listdir(test_clean_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))])
    
    # Verify matching pairs
    if len(train_hazy_files) != len(train_clean_files):
        print(f"Warning: Number of hazy ({len(train_hazy_files)}) and clean ({len(train_clean_files)}) training images doesn't match!")
    
    if len(test_hazy_files) != len(test_clean_files):
        print(f"Warning: Number of hazy ({len(test_hazy_files)}) and clean ({len(test_clean_files)}) test images doesn't match!")
    
    # Create dataset dictionary
    dataset = {
        'train': {
            'hazy': train_hazy_files,
            'clean': train_clean_files
        },
        'test': {
            'hazy': test_hazy_files,
            'clean': test_clean_files
        }
    }
    
    print(f"Found {len(train_hazy_files)} training pairs and {len(test_hazy_files)} test pairs")
    
    return dataset

def enhance_contrast(image):
    """Enhance contrast of the image using CLAHE."""
    # Convert to LAB color space
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
    
    return enhanced

def estimate_smoke_level(image):
    """Estimate smoke level in an image using statistical features.
    
    Args:
        image: Input image
        
    Returns:
        Smoke level (0-4) and name
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Extract features
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
    
    # Apply CLAHE to enhance edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Calculate edge density using Sobel
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(magnitude) / 255.0
    
    # Calculate sharpness using Laplacian
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    sharpness = np.var(laplacian) / 10000.0  # Normalize
    
    # Calculate visibility score
    visibility = (contrast * edge_density * sharpness) ** (1/3)
    
    # Map visibility score to smoke level
    if visibility > 0.15:
        level = 0  # none
    elif visibility > 0.10:
        level = 1  # light
    elif visibility > 0.05:
        level = 2  # medium
    elif visibility > 0.02:
        level = 3  # heavy
    else:
        level = 4  # extreme
    
    # Map level to name
    level_names = ['none', 'light', 'medium', 'heavy', 'extreme']
    level_name = level_names[level]
    
    return level, level_name 