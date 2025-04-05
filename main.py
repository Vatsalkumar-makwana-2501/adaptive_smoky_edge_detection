import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
from tensorflow.keras.models import Model, load_model

# Import project modules
from utils import load_dataset, setup_directories, find_corresponding_gt, load_ground_truth
from smoke_removal import (
    load_smoke_dataset, train_smoke_removal_model, remove_smoke, 
    estimate_smoke_level, enhance_contrast, SMOKE_REMOVAL_MODEL_PATH,
    create_smoke_removal_model
)
from edge_detection import (
    canny_edge, sobel_edge, laplacian_edge, guided_filter_edge, 
    enhance_smoky_image, adaptive_edge_detection, deep_edge_detection,
    hybrid_edge_detection, rethink_canny_edge_detection, train_edge_detection_model,
    EDGE_MODEL_PATH, train_smoke_aware_edge_detection_model, smoke_aware_edge_detection,
    SMOKE_AWARE_MODEL_PATH
)
from smoke_level_model import (
    train_smoke_level_model, predict_smoke_level
)
from evaluation import (
    evaluate_edge_detection, compare_methods, plot_comparative_metrics
)

def main():
    # Configure paths
    base_dir = 'BSDS500'
    gt_dir = os.path.join(base_dir, 'ground_truth')
    smoke_dataset_dir = 'Smoke'
    
    # Create project directories
    setup_directories()
    
    # Load original dataset for edge detection training
    print("Loading BSDS500 dataset...")
    train_images, train_gt = load_dataset(base_dir, 'train')
    val_images, val_gt = load_dataset(base_dir, 'val')
    test_images, test_gt = load_dataset(base_dir, 'test')
    
    # Load smoke dataset
    print("Loading smoke dataset...")
    smoke_dataset = load_smoke_dataset(smoke_dataset_dir)
    
    # Train or load smoke removal model
    smoke_removal_model = None
    try:
        # Create model directories if they don't exist
        os.makedirs(os.path.dirname(SMOKE_REMOVAL_MODEL_PATH), exist_ok=True)
        
        if not os.path.exists(SMOKE_REMOVAL_MODEL_PATH):
            print("Training smoke removal model...")
            train_hazy = smoke_dataset['train']['hazy']
            train_clean = smoke_dataset['train']['clean']
            test_hazy = smoke_dataset['test']['hazy']
            test_clean = smoke_dataset['test']['clean']
            
            # Use subset of training data for validation
            val_idx = int(len(train_hazy) * 0.8)
            val_hazy = train_hazy[val_idx:]
            val_clean = train_clean[val_idx:]
            train_hazy = train_hazy[:val_idx]
            train_clean = train_clean[:val_idx]
            
            smoke_removal_model, _ = train_smoke_removal_model(
                train_hazy, train_clean,
                val_hazy, val_clean,
                epochs=10,  # Reduced epochs for quicker testing
                batch_size=4
            )
        else:
            print("Loading pre-trained smoke removal model...")
            # Create a fresh model instance and load weights
            input_shape = (256, 256, 3)
            smoke_removal_model = create_smoke_removal_model(input_shape)
            smoke_removal_model.load_weights(SMOKE_REMOVAL_MODEL_PATH)
            print("Smoke removal model loaded successfully")
    except Exception as e:
        print(f"Error with smoke removal model: {e}")
        print("Continuing without smoke removal model...")
    
    # First, train the smoke-aware edge detection model (this is our best approach)
    if not os.path.exists(SMOKE_AWARE_MODEL_PATH):
        print("\nTraining smoke-aware edge detection model...")
        try:
            smoke_aware_model, _ = train_smoke_aware_edge_detection_model(
                train_images, train_gt,
                val_images, val_gt,
                epochs=10,
                batch_size=8
            )
        except Exception as e:
            print(f"Error training smoke-aware model: {e}")
            print("Continuing without smoke-aware edge detection model...")
    
    # Test edge detection on smoky images and their clean versions
    print("\nTesting edge detection on original smoky images and after smoke removal...")
    test_edge_detection_on_smoke_dataset(smoke_dataset['test'], smoke_removal_model)
    
    # Train smoke level estimation model
    print("\nTraining smoke level estimation model...")
    smoke_model = None
    try:
        # Create dataset for smoke level estimation
        smoke_level_dataset = []
        for hazy_path in smoke_dataset['train']['hazy']:
            # Estimate smoke level using our heuristic method
            img = cv2.imread(hazy_path)
            if img is None:
                continue
            level, level_name = estimate_smoke_level(img)
            
            smoke_level_dataset.append({
                'image_path': hazy_path,
                'smoke_level': level,
                'smoke_level_name': level_name
            })
        
        # Use separate validation set
        val_dataset = []
        for hazy_path in smoke_dataset['test']['hazy']:
            img = cv2.imread(hazy_path)
            if img is None:
                continue
            level, level_name = estimate_smoke_level(img)
            
            val_dataset.append({
                'image_path': hazy_path,
                'smoke_level': level,
                'smoke_level_name': level_name
            })
        
        # Train model
        smoke_model, _ = train_smoke_level_model(
            smoke_level_dataset + val_dataset,  # Combine for better training
            epochs=10,
            batch_size=8,
            use_transfer_learning=True
        )
    except Exception as e:
        print(f"Error training smoke level model: {e}")
        print("Using heuristic smoke level estimation...")
    
    # Test adaptive edge detection
    print("\nTesting adaptive edge detection system...")
    try:
        test_adaptive_edge_detection_with_removal(
            smoke_dataset['test']['hazy'], 
            smoke_removal_model,
            smoke_model
        )
    except Exception as e:
        print(f"Error in adaptive edge detection: {e}")
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    try:
        compare_all_methods_with_removal(
            smoke_dataset['test']['hazy'],
            smoke_dataset['test']['clean'],
            smoke_removal_model,
            smoke_model,
            num_samples=3
        )
    except Exception as e:
        print(f"Error in comprehensive evaluation: {e}")
        
    print("Processing completed!")

def test_edge_detection_on_smoke_dataset(test_dataset, smoke_removal_model=None):
    """Compare edge detection on original smoky images and after smoke removal."""
    # Select a few test images to compare
    num_samples = min(5, len(test_dataset['hazy']))
    test_indices = random.sample(range(len(test_dataset['hazy'])), num_samples)
    
    for idx in test_indices:
        hazy_path = test_dataset['hazy'][idx]
        clean_path = test_dataset['clean'][idx]
        
        # Load images
        hazy_img = cv2.imread(hazy_path)
        clean_img = cv2.imread(clean_path)
        
        if hazy_img is None or clean_img is None:
            print(f"Failed to load images: {hazy_path} or {clean_path}")
            continue
        
        # Apply smoke removal if model is available
        if smoke_removal_model is not None:
            removed_img = remove_smoke(hazy_img, smoke_removal_model)
        else:
            # Fallback: use contrast enhancement if no model available
            removed_img = enhance_contrast(hazy_img)
        
        # Convert images to RGB for display
        hazy_rgb = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        removed_rgb = cv2.cvtColor(removed_img, cv2.COLOR_BGR2RGB)
        
        # Estimate smoke level in the hazy image
        level, level_name = estimate_smoke_level(hazy_img)
        print(f"\nImage: {os.path.basename(hazy_path)} - Estimated smoke level: {level_name}")
        
        # Apply various edge detection methods
        print("Applying edge detection methods to original smoky image...")
        smoky_edges = {
            'canny': canny_edge(hazy_rgb, sigma=1.5, low_threshold=40, high_threshold=120),
            'sobel': sobel_edge(hazy_rgb, threshold=0.15),
            'laplacian': laplacian_edge(hazy_rgb, threshold=0.15),
            'improved_canny': rethink_canny_edge_detection(hazy_rgb, adaptive_thresholds=True)
        }
        
        print("Applying edge detection methods to smoke-removed image...")
        removed_edges = {
            'canny': canny_edge(removed_rgb, sigma=1.5, low_threshold=40, high_threshold=120),
            'sobel': sobel_edge(removed_rgb, threshold=0.15),
            'laplacian': laplacian_edge(removed_rgb, threshold=0.15),
            'improved_canny': rethink_canny_edge_detection(removed_rgb, adaptive_thresholds=True)
        }
        
        print("Applying edge detection methods to clean image (ground truth)...")
        clean_edges = {
            'canny': canny_edge(clean_rgb, sigma=1.5, low_threshold=40, high_threshold=120),
            'sobel': sobel_edge(clean_rgb, threshold=0.15),
            'laplacian': laplacian_edge(clean_rgb, threshold=0.15),
            'improved_canny': rethink_canny_edge_detection(clean_rgb, adaptive_thresholds=True)
        }
        
        # Display comparison
        plt.figure(figsize=(15, 10))
        
        # Original, Removed, and Clean images
        plt.subplot(3, 5, 1)
        plt.imshow(hazy_rgb)
        plt.title("Smoky Image")
        plt.axis('off')
        
        plt.subplot(3, 5, 2)
        plt.imshow(removed_rgb)
        plt.title("Smoke Removed")
        plt.axis('off')
        
        plt.subplot(3, 5, 3)
        plt.imshow(clean_rgb)
        plt.title("Clean (Ground Truth)")
        plt.axis('off')
        
        # Display edge detection results
        methods = ['canny', 'sobel', 'laplacian', 'improved_canny']
        titles = ['Canny', 'Sobel', 'Laplacian', 'Improved Canny']
        
        for i, (method, title) in enumerate(zip(methods, titles)):
            # Smoky edges
            plt.subplot(3, 5, 6 + i)
            plt.imshow(smoky_edges[method], cmap='gray')
            plt.title(f"Smoky - {title}")
            plt.axis('off')
            
            # Removed smoke edges
            plt.subplot(3, 5, 11 + i)
            plt.imshow(removed_edges[method], cmap='gray')
            plt.title(f"Removed - {title}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Compare edge detection metrics
        print("\nEdge Detection Metrics - Original Smoky vs Smoke Removed:")
        for method, title in zip(methods, titles):
            smoky_metrics = evaluate_edge_detection(smoky_edges[method], clean_edges[method])
            removed_metrics = evaluate_edge_detection(removed_edges[method], clean_edges[method])
            
            print(f"\n{title} Method:")
            print(f"Smoky: F1={smoky_metrics['f1']:.4f}, IoU={smoky_metrics['iou']:.4f}, SSIM={smoky_metrics['ssim']:.4f}")
            print(f"Removed: F1={removed_metrics['f1']:.4f}, IoU={removed_metrics['iou']:.4f}, SSIM={removed_metrics['ssim']:.4f}")
            print(f"Improvement: F1={removed_metrics['f1']-smoky_metrics['f1']:.4f}, IoU={removed_metrics['iou']-smoky_metrics['iou']:.4f}, SSIM={removed_metrics['ssim']-smoky_metrics['ssim']:.4f}")

def test_adaptive_edge_detection_with_removal(hazy_images, smoke_removal_model, smoke_model=None):
    """Test adaptive edge detection with smoke removal."""
    # Select a few test images
    num_samples = min(5, len(hazy_images))
    test_indices = random.sample(range(len(hazy_images)), num_samples)
    
    # Check if deep learning models are available
    has_deep_model = os.path.exists(EDGE_MODEL_PATH)
    has_smoke_aware_model = os.path.exists(SMOKE_AWARE_MODEL_PATH)
    
    if has_deep_model:
        deep_model = tf.keras.models.load_model(EDGE_MODEL_PATH)
    
    if has_smoke_aware_model:
        smoke_aware_model = tf.keras.models.load_model(SMOKE_AWARE_MODEL_PATH)
    
    for idx in test_indices:
        hazy_path = hazy_images[idx]
        
        # Load image
        hazy_img = cv2.imread(hazy_path)
        if hazy_img is None:
            print(f"Failed to load image: {hazy_path}")
            continue
        
        # Predict smoke level 
        if smoke_model is not None:
            predicted_level_idx, confidence = predict_smoke_level(smoke_model, hazy_img)
            smoke_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
            predicted_level = smoke_levels[predicted_level_idx]
            print(f"\nPredicted smoke level for {os.path.basename(hazy_path)}: {predicted_level} (confidence: {confidence:.2f})")
        else:
            # Use our heuristic method if no model is available
            predicted_level_idx, predicted_level = estimate_smoke_level(hazy_img)
            print(f"\nEstimated smoke level for {os.path.basename(hazy_path)}: {predicted_level}")
        
        # Apply smoke removal
        print("Applying smoke removal...")
        removed_img = remove_smoke(hazy_img, smoke_removal_model)
        
        # Convert to RGB for processing
        hazy_rgb = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        removed_rgb = cv2.cvtColor(removed_img, cv2.COLOR_BGR2RGB)
        
        # Apply adaptive edge detection on both original and smoke-removed images
        methods = {}
        
        # Original smoky image
        if has_smoke_aware_model:
            methods['Original + Smoke-Aware'] = smoke_aware_edge_detection(
                hazy_rgb, predicted_level, model=smoke_aware_model
            )
        
        methods['Original + Adaptive'] = adaptive_edge_detection(hazy_rgb, predicted_level)
        
        if has_deep_model:
            methods['Original + Deep'] = deep_edge_detection(hazy_rgb, model=deep_model)
        
        # Smoke-removed image
        if has_smoke_aware_model:
            methods['Removed + Smoke-Aware'] = smoke_aware_edge_detection(
                removed_rgb, 'none', model=smoke_aware_model
            )
        
        methods['Removed + Adaptive'] = adaptive_edge_detection(removed_rgb, 'none')
        
        if has_deep_model:
            methods['Removed + Deep'] = deep_edge_detection(removed_rgb, model=deep_model)
        
        # Display results
        plt.figure(figsize=(15, 8))
        
        # Original and smoke-removed images
        plt.subplot(2, len(methods) + 1, 1)
        plt.imshow(hazy_rgb)
        plt.title(f"Original ({predicted_level} smoke)")
        plt.axis('off')
        
        plt.subplot(2, len(methods) + 1, len(methods) + 2)
        plt.imshow(removed_rgb)
        plt.title("Smoke Removed")
        plt.axis('off')
        
        # Edge detection results
        for i, (method_name, edge_map) in enumerate(methods.items()):
            plt.subplot(2, len(methods) + 1, i + 2)
            plt.imshow(edge_map, cmap='gray')
            plt.title(method_name)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def compare_all_methods_with_removal(hazy_images, clean_images, smoke_removal_model, smoke_model=None, num_samples=3):
    """Compare all edge detection methods on original and smoke-removed images."""
    # Get samples with ground truth
    results = []
    
    # Sample random images for testing
    test_indices = random.sample(range(len(hazy_images)), min(num_samples, len(hazy_images)))
    
    # Check if models are available
    has_deep_model = os.path.exists(EDGE_MODEL_PATH)
    has_smoke_aware_model = os.path.exists(SMOKE_AWARE_MODEL_PATH)
    
    if has_deep_model:
        deep_model = tf.keras.models.load_model(EDGE_MODEL_PATH)
    
    if has_smoke_aware_model:
        smoke_aware_model = tf.keras.models.load_model(SMOKE_AWARE_MODEL_PATH)
    
    for idx in test_indices:
        hazy_path = hazy_images[idx]
        clean_path = clean_images[idx]
        
        # Load images
        hazy_img = cv2.imread(hazy_path)
        clean_img = cv2.imread(clean_path)
        
        if hazy_img is None or clean_img is None:
            print(f"Failed to load images: {hazy_path} or {clean_path}")
            continue
        
        # Apply smoke removal
        removed_img = remove_smoke(hazy_img, smoke_removal_model)
        
        # Convert to RGB for processing
        hazy_rgb = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        removed_rgb = cv2.cvtColor(removed_img, cv2.COLOR_BGR2RGB)
        
        # Predict smoke level
        if smoke_model is not None:
            predicted_level_idx, confidence = predict_smoke_level(smoke_model, hazy_img)
            smoke_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
            predicted_level = smoke_levels[predicted_level_idx]
            print(f"\nPredicted smoke level for {os.path.basename(hazy_path)}: {predicted_level} (confidence: {confidence:.2f})")
        else:
            # Use heuristic method
            predicted_level_idx, predicted_level = estimate_smoke_level(hazy_img)
            print(f"\nEstimated smoke level for {os.path.basename(hazy_path)}: {predicted_level}")
        
        # Apply edge detection methods to original smoky image
        original_methods = {
            'original_canny': canny_edge(hazy_rgb, sigma=1.5, low_threshold=40, high_threshold=120),
            'original_sobel': sobel_edge(hazy_rgb, threshold=0.15),
            'original_improved_canny': rethink_canny_edge_detection(hazy_rgb, adaptive_thresholds=True),
            'original_adaptive': adaptive_edge_detection(hazy_rgb, predicted_level)
        }
        
        # Apply edge detection methods to smoke-removed image
        removed_methods = {
            'removed_canny': canny_edge(removed_rgb, sigma=1.5, low_threshold=40, high_threshold=120),
            'removed_sobel': sobel_edge(removed_rgb, threshold=0.15),
            'removed_improved_canny': rethink_canny_edge_detection(removed_rgb, adaptive_thresholds=True),
            'removed_adaptive': adaptive_edge_detection(removed_rgb, 'none')  # Use 'none' since smoke is removed
        }
        
        # Apply edge detection methods to clean (ground truth) image
        clean_methods = {
            'clean_canny': canny_edge(clean_rgb, sigma=1.5, low_threshold=40, high_threshold=120),
            'clean_sobel': sobel_edge(clean_rgb, threshold=0.15),
            'clean_improved_canny': rethink_canny_edge_detection(clean_rgb, adaptive_thresholds=True)
        }
        
        # Add deep learning methods if available
        if has_deep_model:
            original_methods['original_deep'] = deep_edge_detection(hazy_rgb, model=deep_model)
            removed_methods['removed_deep'] = deep_edge_detection(removed_rgb, model=deep_model)
        
        # Add smoke-aware edge detection if available
        if has_smoke_aware_model:
            original_methods['original_smoke_aware'] = smoke_aware_edge_detection(
                hazy_rgb, predicted_level, model=smoke_aware_model
            )
            removed_methods['removed_smoke_aware'] = smoke_aware_edge_detection(
                removed_rgb, 'none', model=smoke_aware_model
            )
        
        # Combine all methods for evaluation
        all_methods = {}
        all_methods.update(original_methods)
        all_methods.update(removed_methods)
        
        # Calculate and display metrics
        print("\n" + "="*70)
        print(f"{'Method':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'IoU':<10} {'SSIM':<10}")
        print("="*70)
        
        # Evaluate against clean image edges (ground truth)
        metrics_dict = {}
        for method_name, edge_map in all_methods.items():
            # Determine which clean edge map to use for comparison
            if 'canny' in method_name:
                gt_edge_map = clean_methods['clean_canny']
            elif 'sobel' in method_name:
                gt_edge_map = clean_methods['clean_sobel']
            else:
                gt_edge_map = clean_methods['clean_improved_canny']
            
            metrics = evaluate_edge_detection(edge_map, gt_edge_map)
            metrics_dict[method_name] = metrics
            
            print(f"{method_name:<25} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}      {metrics['iou']:.4f}     {metrics['ssim']:.4f}")
        
        print("="*70)
        
        # Print optimal F1 scores
        print("\nOptimal F1 Scores:")
        for method_name, metrics in metrics_dict.items():
            print(f"{method_name:<25}: F1={metrics['optimal_f1']:.4f} at threshold={metrics['optimal_threshold']:.2f}")
        
        # Store results
        results.append({
            'image': os.path.basename(hazy_path),
            'smoke_level': predicted_level,
            'metrics': metrics_dict
        })
    
    # Calculate average metrics across all test images
    if results:
        # Create method groups for better comparison
        method_groups = {
            'Original': [m for m in results[0]['metrics'].keys() if m.startswith('original_')],
            'Removed': [m for m in results[0]['metrics'].keys() if m.startswith('removed_')]
        }
        
        # Calculate average metrics for each method group
        group_metrics = {}
        for group_name, methods in method_groups.items():
            group_metrics[group_name] = {
                'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'ssim': 0, 
                'count': 0
            }
            
            for result in results:
                for method in methods:
                    if method in result['metrics']:
                        metrics = result['metrics'][method]
                        group_metrics[group_name]['precision'] += metrics['precision']
                        group_metrics[group_name]['recall'] += metrics['recall']
                        group_metrics[group_name]['f1'] += metrics['f1']
                        group_metrics[group_name]['iou'] += metrics['iou']
                        group_metrics[group_name]['ssim'] += metrics['ssim']
                        group_metrics[group_name]['count'] += 1
            
            # Calculate averages
            count = group_metrics[group_name]['count']
            if count > 0:
                for metric in ['precision', 'recall', 'f1', 'iou', 'ssim']:
                    group_metrics[group_name][metric] /= count
        
        # Display summary of results
        print("\n=== SUMMARY OF RESULTS ===")
        for group_name, metrics in group_metrics.items():
            print(f"\n{group_name} Methods (Average):")
            for metric_name, value in metrics.items():
                if metric_name != 'count':
                    print(f"{metric_name.capitalize()}: {value:.4f}")
        
        # Calculate improvement
        if 'Original' in group_metrics and 'Removed' in group_metrics:
            print("\n=== IMPROVEMENT WITH SMOKE REMOVAL ===")
            for metric in ['precision', 'recall', 'f1', 'iou', 'ssim']:
                improvement = group_metrics['Removed'][metric] - group_metrics['Original'][metric]
                percentage = (improvement / max(group_metrics['Original'][metric], 1e-5)) * 100
                print(f"{metric.capitalize()}: {improvement:.4f} ({percentage:+.2f}%)")
        
        # Plot comparative metrics
        plot_comparative_metrics(group_metrics)
        
        return results, group_metrics
    
    return results

def visualize_edge_detection(image, edges, gt_edges, title):
    """Visualize edge detection results with ground truth."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(gt_edges, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def predict_smoke_level(model, image):
    """Predict smoke level in an image.
    
    Args:
        model: Trained smoke level classification model
        image: Image to predict smoke level for
        
    Returns:
        Tuple of (predicted_level_idx, confidence)
    """
    # Preprocess the image
    if isinstance(image, str):
        # Load the image if path is provided
        image = cv2.imread(image)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to match model input size
    image = cv2.resize(image, (224, 224))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Predict
    prediction = model.predict(image)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    return predicted_class, float(confidence)

if __name__ == "__main__":
    main() 