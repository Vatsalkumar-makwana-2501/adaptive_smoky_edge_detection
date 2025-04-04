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

# Import project modules
from utils import load_dataset, setup_directories, find_corresponding_gt, load_ground_truth
from smoke_generation import create_smoky_dataset
from edge_detection import (
    canny_edge, sobel_edge, laplacian_edge, guided_filter_edge, 
    enhance_smoky_image, adaptive_edge_detection
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
    
    # Create project directories
    setup_directories()
    
    # Load dataset
    print("Loading dataset...")
    train_images, train_gt = load_dataset(base_dir, 'train')
    val_images, val_gt = load_dataset(base_dir, 'val')
    test_images, test_gt = load_dataset(base_dir, 'test')
    
    # Create a smoke-augmented dataset
    print("\nCreating smoke-augmented dataset...")
    # Use more images for better model training
    test_subset = train_images[:20]  # Use 20 images for demonstration
    smoky_dataset = create_smoky_dataset(test_subset, methods=['perlin', 'gaussian', 'texture'])
    
    # Display some smoke-augmented images
    show_smoke_augmented_samples(smoky_dataset, 2)
    
    # Test edge detection on smoky images
    print("\nTesting edge detection algorithms on smoky images...")
    test_edge_detection_on_smoke_levels(smoky_dataset, gt_dir)
    
    # Train smoke level estimation model with improved parameters
    print("\nTraining smoke level estimation model...")
    # Use transfer learning and more epochs for better accuracy
    smoke_model, _ = train_smoke_level_model(
        smoky_dataset, 
        epochs=20,  # Increased epochs
        batch_size=8,  # Smaller batch size for better generalization
        use_transfer_learning=True
    )
    
    # Test adaptive edge detection
    print("\nTesting adaptive edge detection system...")
    test_adaptive_edge_detection(smoky_dataset, smoke_model, gt_dir)
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    compare_all_methods(smoky_dataset, smoke_model, gt_dir, num_samples=5)  # More samples for better comparison

def show_smoke_augmented_samples(dataset, num_samples=2):
    """Display samples with different smoke levels."""
    # Get unique original images
    unique_images = list(set([item['original_path'] for item in dataset]))
    sample_images = random.sample(unique_images, min(num_samples, len(unique_images)))
    
    for original in sample_images:
        # Find all variations of this image
        variations = [item for item in dataset if item['original_path'] == original]
        variations.sort(key=lambda x: x['smoke_level'])  # Sort by smoke level
        
        plt.figure(figsize=(15, 3))
        for i, var in enumerate(variations):
            img = cv2.imread(var['smoky_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, len(variations), i+1)
            plt.imshow(img)
            plt.title(f"{var['smoke_level_name']}\n{var['smoke_method']}")
            plt.axis('off')
        plt.suptitle(f"Smoke Variations for {os.path.basename(original)}")
        plt.tight_layout()
        plt.show()

def test_edge_detection_on_smoke_levels(dataset, gt_dir):
    """Compare edge detection on different smoke levels."""
    # Get a sample image in all smoke levels
    unique_images = list(set([item['original_path'] for item in dataset]))
    sample_image_path = random.choice(unique_images)
    sample_image_name = os.path.basename(sample_image_path).split('.')[0]
    
    # Find corresponding ground truth
    gt_path = find_corresponding_gt(sample_image_path, gt_dir)
    
    if gt_path is None:
        print(f"Ground truth not found for {sample_image_name}, skipping.")
        return
    
    # Get all variations of this image
    variations = [item for item in dataset if item['original_path'] == sample_image_path]
    variations.sort(key=lambda x: x['smoke_level'])  # Sort by smoke level
    
    # Compare edge detection for each smoke level
    for variation in variations:
        print(f"\nTesting edge detection on {variation['smoke_level_name']} smoke ({variation['smoke_method']})")
        image = cv2.imread(variation['smoky_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply different edge detection methods with improved parameters
        edges = {
            'canny': canny_edge(image, sigma=1.5, low_threshold=40, high_threshold=120),
            'sobel': sobel_edge(image, threshold=0.15),
            'laplacian': laplacian_edge(image, threshold=0.15),
            'guided': guided_filter_edge(image, radius=10, eps=0.25**2, threshold=0.25)
        }
        
        # Compare methods
        compare_methods(variation['smoky_path'], gt_path, edges, 
                       ['Canny', 'Sobel', 'Laplacian', 'Guided Filter'])

def test_adaptive_edge_detection(smoky_dataset, smoke_model, gt_dir):
    """Test adaptive edge detection system with smoke level prediction.
    
    Args:
        smoky_dataset: List of smoky image dictionaries
        smoke_model: Trained smoke level classification model
        gt_dir: Directory with ground truth edge maps
    """
    smoke_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
    
    # Test one example image from each smoke level
    for level_idx, level_name in enumerate(smoke_levels):
        # Find images of this smoke level
        level_samples = [item for item in smoky_dataset if item['smoke_level'] == level_idx]
        if not level_samples:
            print(f"No samples found for {level_name} smoke level")
            continue
            
        # Choose a random sample
        sample = random.choice(level_samples)
        smoke_method = sample['smoke_method']
        
        print(f"\nTesting on {level_name} smoke image ({smoke_method})")
        
        # Load image
        image = cv2.imread(sample['smoky_path'])
        if image is None:
            print(f"Failed to load image: {sample['smoky_path']}")
            continue
        
        # Predict smoke level 
        predicted_level_idx, confidence = predict_smoke_level(smoke_model, image)
        predicted_level = smoke_levels[predicted_level_idx]
        print(f"Predicted smoke level: {predicted_level} (confidence: {confidence:.2f})")
        
        # Apply adaptive edge detection based on predicted level
        edges = adaptive_edge_detection(image, predicted_level)
        
        # Try to load ground truth if it exists
        gt_path = find_corresponding_gt(sample['original_path'], gt_dir)
        
        # Display without ground truth if it's not available or can't be loaded
        if gt_path is None:
            print(f"Ground truth not found for {os.path.basename(sample['smoky_path'])}")
            display_result_without_gt(image, edges, predicted_level)
            continue
        
        # Load ground truth from .mat file
        gt_edges = load_ground_truth(gt_path)
        if gt_edges is None:
            print(f"Failed to extract ground truth data from {gt_path}")
            display_result_without_gt(image, edges, predicted_level)
            continue
            
        # Ensure ground truth is properly normalized
        if gt_edges.max() > 1.0:
            gt_edges = gt_edges / 255.0
            
        # Evaluate against ground truth
        metrics = evaluate_edge_detection(edges, gt_edges)
        
        print("\nEvaluating adaptive edge detection:")
        print("Evaluation Metrics:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"Optimal F1 Score: {metrics['optimal_f1']:.4f} at threshold {metrics['optimal_threshold']:.2f}")
        
        # Visualize results
        visualize_edge_detection(image, edges, gt_edges, f"Adaptive Edge Detection ({predicted_level} smoke)")

def display_result_without_gt(image, edges, predicted_level):
    """Display edge detection results without ground truth.
    
    Args:
        image: Original image
        edges: Detected edges
        predicted_level: Predicted smoke level
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title(f"Adaptive Edge Detection ({predicted_level} smoke)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_all_methods(dataset, smoke_model, gt_dir, num_samples=3):
    """Compare all edge detection methods on sample images.
    
    Args:
        dataset: List of smoky image dictionaries 
        smoke_model: Trained smoke level classification model
        gt_dir: Directory with ground truth edge maps
        num_samples: Number of samples to evaluate
    
    Returns:
        List of result dictionaries with metrics for each method
    """
    # Get samples with ground truth
    results = []
    
    # Sample from all smoke levels to ensure we test on diverse conditions
    selected_samples = []
    for level in range(5):  # none, light, medium, heavy, extreme
        level_samples = [item for item in dataset if item['smoke_level'] == level]
        if level_samples:
            selected_samples.extend(random.sample(level_samples, min(1, len(level_samples))))
    
    # If we need more samples, add random ones
    if len(selected_samples) < num_samples:
        remaining = random.sample(
            [item for item in dataset if item not in selected_samples],
            min(num_samples - len(selected_samples), len(dataset) - len(selected_samples))
        )
        selected_samples.extend(remaining)
    
    for item in selected_samples:
        # Find corresponding ground truth
        gt_path = find_corresponding_gt(item['original_path'], gt_dir)
        
        if gt_path is None:
            print(f"Ground truth not found for {item['smoky_path']}, skipping...")
            continue
        
        # Load image
        image = cv2.imread(item['smoky_path'])
        if image is None:
            print(f"Failed to load image: {item['smoky_path']}")
            continue
        
        # Load ground truth
        gt_image = load_ground_truth(gt_path)
        if gt_image is None:
            print(f"Failed to load ground truth image: {gt_path}")
            continue
            
        # Normalize ground truth
        gt_edges = gt_image / 255.0
        
        print(f"\nEvaluating on {item['smoke_level_name']} smoke image {os.path.basename(item['original_path'])}")
        
        # Predict smoke level
        predicted_level_idx, confidence = predict_smoke_level(smoke_model, image)
        smoke_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
        predicted_level = smoke_levels[predicted_level_idx]
        print(f"Predicted level: {predicted_level} (confidence: {confidence:.2f}), Actual level: {item['smoke_level_name']}")
        
        # Apply different methods with improved parameters
        canny_edges = canny_edge(image, sigma=1.5, low_threshold=40, high_threshold=120)
        sobel_edges = sobel_edge(image, threshold=0.15)
        guided_edges = guided_filter_edge(image, radius=10, eps=0.25**2, threshold=0.25)
        adaptive_edges = adaptive_edge_detection(image, predicted_level)
        
        # Evaluate all methods
        methods = {
            'canny': canny_edges,
            'sobel': sobel_edges,
            'guided': guided_edges,
            'adaptive': adaptive_edges
        }
        
        # Calculate and display metrics
        print("\n" + "="*50)
        print(f"{'Method':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'IoU':<10} {'SSIM':<10}")
        print("="*50)
        
        metrics_dict = {}
        for method_name, edge_map in methods.items():
            metrics = evaluate_edge_detection(edge_map, gt_edges)
            metrics_dict[method_name] = metrics
            
            print(f"{method_name.capitalize():<15} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}      {metrics['iou']:.4f}     {metrics['ssim']:.4f}")
        
        print("="*50)
        
        # Print optimal F1 scores
        print("\nOptimal F1 Scores:")
        for method_name, metrics in metrics_dict.items():
            print(f"{method_name.capitalize()}: F1={metrics['optimal_f1']:.4f} at threshold={metrics['optimal_threshold']:.2f}")
        
        # Store results
        results.append({
            'image': os.path.basename(item['original_path']),
            'smoke_level': item['smoke_level_name'],
            'metrics': metrics_dict
        })
    
    # Calculate average metrics across all test images
    if results:
        avg_metrics = {method: {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'ssim': 0} 
                      for method in ['canny', 'sobel', 'guided', 'adaptive']}
        
        # Create counters for each method to handle cases where not all metrics are present
        method_counts = {method: 0 for method in avg_metrics.keys()}
        
        for result in results:
            for method, metrics in result['metrics'].items():
                method_counts[method] += 1
                for metric_name, value in metrics.items():
                    # Only include the standard metrics
                    if metric_name in avg_metrics[method]:
                        avg_metrics[method][metric_name] += value
        
        # Calculate averages
        for method in avg_metrics:
            if method_counts[method] > 0:
                for metric_name in avg_metrics[method]:
                    avg_metrics[method][metric_name] /= method_counts[method]
        
        # Display summary of results
        print("\n=== SUMMARY OF RESULTS ===")
        for method, metrics in avg_metrics.items():
            print(f"\n{method.capitalize()} Edge Detection:")
            for metric_name, value in metrics.items():
                print(f"{metric_name.capitalize()}: {value:.4f}")
        
        # Plot average metrics
        plot_comparative_metrics(avg_metrics)
        
        return results, avg_metrics
    
    return results

def comprehensive_evaluation(smoky_dataset, smoke_model, gt_dir):
    """Run comprehensive evaluation comparing all methods.
    
    Args:
        smoky_dataset: List of smoky image dictionaries
        smoke_model: Trained smoke level classification model
        gt_dir: Directory with ground truth edge maps
    """
    from utils import load_ground_truth  # Import the function
    
    smoke_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
    results = []
    
    # Choose one image from each smoke level
    for level_idx, level_name in enumerate(smoke_levels):
        # Find images of this smoke level
        level_samples = [item for item in smoky_dataset if item['smoke_level'] == level_idx]
        if not level_samples:
            print(f"No samples found for {level_name} smoke level")
            continue
            
        # Choose a random sample
        sample = random.choice(level_samples)
        
        # Load image
        image = cv2.imread(sample['smoky_path'])
        if image is None:
            print(f"Failed to load image: {sample['smoky_path']}")
            continue
            
        # Try to find corresponding ground truth
        gt_path = find_corresponding_gt(sample['original_path'], gt_dir)
        if gt_path is None:
            print(f"Ground truth not found for {sample['smoky_path']}, skipping...")
            continue
        
        # Load ground truth from .mat file
        gt_edges = load_ground_truth(gt_path)
        if gt_edges is None:
            print(f"Failed to extract ground truth data from {gt_path}")
            continue
            
        # Ensure ground truth is properly normalized
        if gt_edges.max() > 1.0:
            gt_edges = gt_edges / 255.0
        
        print(f"\nEvaluating on {level_name} smoke image {os.path.basename(sample['smoky_path'])}")
        
        # Predict smoke level
        predicted_level_idx, confidence = predict_smoke_level(smoke_model, image)
        predicted_level = smoke_levels[predicted_level_idx]
        print(f"Predicted level: {predicted_level} (confidence: {confidence:.2f}), Actual level: {level_name}")
        
        # Apply different edge detection methods
        canny_edges = canny_edge(image, sigma=1.5, low_threshold=40, high_threshold=120)
        sobel_edges = sobel_edge(image, threshold=0.15)
        guided_edges = guided_filter_edge(image, radius=10, eps=0.25**2, threshold=0.25)
        adaptive_edges = adaptive_edge_detection(image, predicted_level)
        
        # Evaluate all methods
        methods = {
            'canny': canny_edges,
            'sobel': sobel_edges,
            'guided': guided_edges,
            'adaptive': adaptive_edges
        }
        
        # Calculate metrics
        result = {'image': os.path.basename(sample['smoky_path']), 'actual_level': level_name, 
                 'predicted_level': predicted_level, 'confidence': confidence, 'metrics': {}}
        
        print("\n" + "="*50)
        print(f"{'Method':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'IoU':<10} {'SSIM':<10}")
        print("="*50)
        
        for method_name, edge_map in methods.items():
            metrics = evaluate_edge_detection(edge_map, gt_edges)
            result['metrics'][method_name] = metrics
            
            print(f"{method_name.capitalize():<15} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}      {metrics['iou']:.4f}     {metrics['ssim']:.4f}")
        
        print("="*50)
        
        # Print optimal F1 scores
        print("\nOptimal F1 Scores:")
        for method_name, metrics in result['metrics'].items():
            print(f"{method_name.capitalize()}: F1={metrics['optimal_f1']:.4f} at threshold={metrics['optimal_threshold']:.2f}")
        
        results.append(result)
        
    return results

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

def visualize_edge_detection(image, edges, gt_edges=None, title="Edge Detection Results"):
    """Visualize edge detection results.
    
    Args:
        image: Original image
        edges: Detected edges
        gt_edges: Ground truth edges (optional)
        title: Plot title
    """
    plt.figure(figsize=(15, 5))
    
    # Display original image
    plt.subplot(131)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Display detected edges
    plt.subplot(132)
    plt.imshow(edges, cmap='gray')
    plt.title("Detected Edges")
    plt.axis('off')
    
    # Display ground truth if provided
    if gt_edges is not None:
        plt.subplot(133)
        plt.imshow(gt_edges, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 