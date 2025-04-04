import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

def prepare_smoke_level_dataset(dataset, target_size=(224, 224), augment=True):
    """Prepare dataset for smoke level estimation model with improved preprocessing."""
    X = []
    y = []
    
    for item in dataset:
        # Load image
        img = cv2.imread(item['smoky_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing for better feature extraction
        # Convert to LAB color space to better handle smoke characteristics
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel for contrast enhancement
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize for model
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        X.append(img)
        y.append(item['smoke_level'])  # 0=none, 1=light, 2=medium, 3=heavy, 4=extreme
    
    return np.array(X), np.array(y)

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=16):
    """Create data generators with augmentation for training."""
    # Calculate class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator, class_weights_dict

def create_smoke_level_model(input_shape=(224, 224, 3), num_classes=5):
    """Create an improved CNN model for smoke level classification."""
    model = Sequential([
        # First convolutional block
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=5, model_type='mobilenet'):
    """Create a more effective transfer learning model using MobileNetV2 or ResNet50V2."""
    if model_type == 'resnet':
        # Use ResNet50V2 for better feature extraction
        base_model = ResNet50V2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        # Use MobileNetV2 for efficiency
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualize_training_history(history):
    """Visualize the training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_smoke_level_model(dataset, model_save_path='models/smoke_level_model.h5', 
                           val_split=0.2, epochs=30, batch_size=16, use_transfer_learning=True):
    """Train an improved model to estimate smoke level in images."""
    # Prepare dataset
    X, y = prepare_smoke_level_dataset(dataset)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create data generators for augmentation
    train_generator, val_generator, class_weights = create_data_generators(
        X_train, y_train, X_val, y_val, batch_size=batch_size
    )
    
    # Create model
    if use_transfer_learning:
        model = create_transfer_learning_model(model_type='resnet')
        print("Using ResNet50V2 transfer learning model")
    else:
        model = create_smoke_level_model()
        print("Using custom CNN model")
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    steps_per_epoch = max(1, len(X_train) // batch_size)
    validation_steps = max(1, len(X_val) // batch_size)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )
    
    # Plot training history
    visualize_training_history(history)
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc*100:.2f}%")
    
    return model, history

def predict_smoke_level(model, image_path, target_size=(224, 224)):
    """Predict smoke level in an image with improved preprocessing."""
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply same preprocessing as training
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Resize and normalize
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict smoke level
    pred = model.predict(img)[0]
    level = np.argmax(pred)
    confidence = pred[level]
    
    # Map level to category name
    level_names = ['none', 'light', 'medium', 'heavy', 'extreme']
    level_name = level_names[level]
    
    return level, level_name, confidence, pred 