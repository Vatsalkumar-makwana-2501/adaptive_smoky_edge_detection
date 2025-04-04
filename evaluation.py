import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.morphology import dilation, disk
from utils import load_ground_truth

def calculate_f1_score(edge_map, gt_edges, tolerance=0.01):
    """Calculate F1 score for edge detection.
    
    Args:
        edge_map: Predicted edge map (0-1 range)
        gt_edges: Ground truth edge map (0-1 range)
        tolerance: Tolerance for edge matching
        
    Returns:
        Tuple of precision, recall, f1, and optimal threshold
    """
    # Ensure inputs are numpy arrays with proper dtype
    if isinstance(edge_map, np.ndarray) and edge_map.max() > 1.0:
        edge_map = edge_map / 255.0
    if isinstance(gt_edges, np.ndarray) and gt_edges.max() > 1.0:
        gt_edges = gt_edges / 255.0
    
    # Convert gt_edges to binary if needed
    if np.unique(gt_edges).size > 2:
        gt_binary = (gt_edges > 0.5).astype(np.uint8)
    else:
        gt_binary = gt_edges.astype(np.uint8)
    
    # Create kernel for dilation
    kernel = np.ones((3, 3), np.uint8)
    
    # Dilate ground truth to allow for some tolerance
    gt_dilated = cv2.dilate(gt_binary, kernel)
    
    # Try different thresholds to find optimal F1 score
    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0
    
    # Try a range of thresholds
    thresholds = np.linspace(0, 1, 20)
    for threshold in thresholds:
        # Binarize edge map using current threshold
        edge_binary = (edge_map > threshold).astype(np.uint8)
        
        # Calculate true positives, false positives, false negatives
        tp = np.sum(np.logical_and(edge_binary, gt_dilated))
        fp = np.sum(np.logical_and(edge_binary, np.logical_not(gt_dilated)))
        fn = np.sum(np.logical_and(np.logical_not(edge_binary), gt_binary))
        
        # Calculate precision and recall
        precision = tp / max(tp + fp, 1e-7)
        recall = tp / max(tp + fn, 1e-7)
        
        # Calculate F1 score
        f1 = 2 * precision * recall / max(precision + recall, 1e-7)
        
        # Update best F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    return best_precision, best_recall, best_f1, best_threshold

def calculate_optimal_f1(pred_prob, gt_edges, num_thresholds=100):
    """Calculate optimal F1 score by trying different thresholds."""
    thresholds = np.linspace(0, 1, num_thresholds)
    f1_scores = []
    precision_values = []
    recall_values = []
    
    for threshold in thresholds:
        pred_binary = (pred_prob > threshold).astype(np.uint8)
        metrics = calculate_f1_score(pred_binary, gt_edges)
        f1_scores.append(metrics[2])
        precision_values.append(metrics[0])
        recall_values.append(metrics[1])
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    
    return {
        'optimal_threshold': thresholds[optimal_idx],
        'optimal_f1': f1_scores[optimal_idx],
        'thresholds': thresholds,
        'f1_values': f1_scores,
        'precision_values': precision_values,
        'recall_values': recall_values
    }

def calculate_iou(pred_edges, gt_edges):
    """Calculate Intersection over Union for edge detection."""
    intersection = np.logical_and(pred_edges, gt_edges)
    union = np.logical_or(pred_edges, gt_edges)
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    return iou

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        SSIM value
    """
    # Ensure images are in the same format
    if isinstance(img1, np.ndarray) and img1.max() > 1.0:
        img1 = img1 / 255.0
    if isinstance(img2, np.ndarray) and img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Ensure images are the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Calculate SSIM
    return structural_similarity(img1, img2, data_range=1.0)

def evaluate_edge_detection(edge_map, gt_edges):
    """Evaluate edge detection against ground truth.
    
    Args:
        edge_map: Predicted edge map
        gt_edges: Ground truth edge map
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Ensure inputs are in the same range
    if isinstance(edge_map, np.ndarray) and edge_map.max() > 1.0:
        edge_map = edge_map / 255.0
    if isinstance(gt_edges, np.ndarray) and gt_edges.max() > 1.0:
        gt_edges = gt_edges / 255.0
    
    # Convert to gray if edge_map is RGB
    if len(edge_map.shape) == 3 and edge_map.shape[2] == 3:
        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
    if len(gt_edges.shape) == 3 and gt_edges.shape[2] == 3:
        gt_edges = cv2.cvtColor(gt_edges, cv2.COLOR_BGR2GRAY)
    
    # Calculate F1 score
    precision, recall, f1, optimal_threshold = calculate_f1_score(edge_map, gt_edges)
    
    # Calculate IoU (Intersection over Union)
    edge_binary = (edge_map > optimal_threshold).astype(np.uint8)
    gt_binary = (gt_edges > 0.5).astype(np.uint8)
    
    intersection = np.logical_and(edge_binary, gt_binary).sum()
    union = np.logical_or(edge_binary, gt_binary).sum()
    iou = intersection / max(union, 1e-7)
    
    # Calculate SSIM (Structural Similarity Index)
    ssim_value = calculate_ssim(edge_map, gt_edges)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'ssim': ssim_value,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': f1
    }

def compare_methods(image_path, gt_path, edges_dict, methods_names=None):
    """Compare multiple edge detection methods on the same image."""
    if methods_names is None:
        methods_names = list(edges_dict.keys())
    
    # Load original image
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Load ground truth
    gt_edges = load_ground_truth(gt_path)
    
    # Calculate metrics for each method
    metrics = {}
    for method, edge_map in edges_dict.items():
        # Resize if needed
        if edge_map.shape != gt_edges.shape:
            edge_map = cv2.resize(edge_map.astype(np.float32), (gt_edges.shape[1], gt_edges.shape[0]))
            # Re-binarize if needed
            if edge_map.max() <= 1:
                edge_map = (edge_map > 0.5).astype(np.uint8)
            else:
                edge_map = (edge_map > 128).astype(np.uint8)
        
        f1_metrics = calculate_f1_score(edge_map, gt_edges)
        metrics[method] = {
            'precision': f1_metrics[0],
            'recall': f1_metrics[1],
            'f1': f1_metrics[2],
            'iou': calculate_iou(edge_map, gt_edges),
            'ssim': calculate_ssim(edge_map, gt_edges)
        }
        
        # Calculate optimal F1 if edge map is in float format
        if np.issubdtype(edge_map.dtype, np.floating) or edge_map.max() <= 1:
            optimal_metrics = calculate_optimal_f1(edge_map, gt_edges)
            metrics[method]['optimal_f1'] = optimal_metrics['optimal_f1']
            metrics[method]['optimal_threshold'] = optimal_metrics['optimal_threshold']
    
    # Visualize results
    num_methods = len(edges_dict)
    plt.figure(figsize=(15, 4 * (2 + num_methods//3)))
    
    # Original image
    plt.subplot(3 + num_methods//3, 3, 1)
    plt.imshow(orig_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth
    plt.subplot(3 + num_methods//3, 3, 2)
    plt.imshow(gt_edges, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Each method
    for i, (method, edge_map) in enumerate(edges_dict.items()):
        method_idx = list(edges_dict.keys()).index(method)
        method_name = methods_names[method_idx] if methods_names else method
        
        # Resize if needed
        if edge_map.shape != gt_edges.shape:
            edge_map = cv2.resize(edge_map.astype(np.float32), (gt_edges.shape[1], gt_edges.shape[0]))
            
        plt.subplot(3 + num_methods//3, 3, i+3)
        plt.imshow(edge_map, cmap='gray')
        
        # Show F1 score in title
        f1 = metrics[method]['f1']
        precision = metrics[method]['precision']
        recall = metrics[method]['recall']
        
        plt.title(f'{method_name}\nF1: {f1:.4f}\nP: {precision:.2f}, R: {recall:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics table
    print("\n" + "="*50)
    print(f"{'Method':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'IoU':<10} {'SSIM':<10}")
    print("="*50)
    
    for method in edges_dict.keys():
        method_idx = list(edges_dict.keys()).index(method)
        method_name = methods_names[method_idx] if methods_names else method
        
        print(f"{method_name:<15} "
              f"{metrics[method]['precision']:.4f}     "
              f"{metrics[method]['recall']:.4f}     "
              f"{metrics[method]['f1']:.4f}      "
              f"{metrics[method]['iou']:.4f}     "
              f"{metrics[method]['ssim']:.4f}")
    
    print("="*50)
    
    # If we have optimal F1 scores, show them
    optimal_methods = [m for m in metrics if 'optimal_f1' in metrics[m]]
    if optimal_methods:
        print("\nOptimal F1 Scores:")
        for method in optimal_methods:
            method_idx = list(edges_dict.keys()).index(method)
            method_name = methods_names[method_idx] if methods_names else method
            
            print(f"{method_name}: "
                  f"F1={metrics[method]['optimal_f1']:.4f} at threshold={metrics[method]['optimal_threshold']:.2f}")
    
    return metrics

def plot_comparative_metrics(metrics_dict, title="Edge Detection Performance Comparison"):
    """Plot comparative metrics for different edge detection methods."""
    methods = list(metrics_dict.keys())
    metric_names = ['precision', 'recall', 'f1', 'iou', 'ssim']
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Bar chart
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    index = np.arange(len(methods))
    
    for i, metric in enumerate(metric_names):
        plt.bar(
            index + i * bar_width, 
            [metrics_dict[method][metric] for method in methods],
            bar_width,
            label=metric.capitalize(),
            color=plt.cm.viridis(i/len(metric_names))
        )
    
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(index + bar_width * 2, [m.capitalize() for m in methods])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.show()
    
    # Radar chart (alternative visualization)
    plt.figure(figsize=(10, 8))
    
    # Create angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    ax = plt.subplot(111, polar=True)
    
    for i, method in enumerate(methods):
        values = [metrics_dict[method][metric] for metric in metric_names]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method.capitalize(), color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    
    plt.title('Performance Radar Chart')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.show()

def analyze_smoke_impact(method, dataset, gt_dir):
    """Analyze the impact of smoke levels on a given edge detection method."""
    # Group data by smoke level
    smoke_levels = {}
    for item in dataset:
        level = item['smoke_level']
        if level not in smoke_levels:
            smoke_levels[level] = []
        smoke_levels[level].append(item)
    
    # Calculate metrics for each smoke level
    results = {}
    for level, images in smoke_levels.items():
        level_results = []
        
        for img_data in images[:5]:  # Limit to 5 images per level for speed
            # Find ground truth
            img_name = os.path.basename(img_data['original_path']).split('.')[0]
            gt_path = os.path.join(gt_dir, f"{img_name}.png")
            if not os.path.exists(gt_path):
                continue
            
            # Apply edge detection
            img = cv2.imread(img_data['smoky_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            edges = method(img)
            
            # Evaluate
            metrics = evaluate_edge_detection(edges, gt_path)
            level_results.append(metrics)
        
        # Calculate average metrics for this level
        if level_results:
            results[level] = {
                'avg_precision': np.mean([r['precision'] for r in level_results]),
                'avg_recall': np.mean([r['recall'] for r in level_results]),
                'avg_f1': np.mean([r['f1'] for r in level_results]),
                'count': len(level_results)
            }
    
    # Plot results
    levels = sorted(results.keys())
    level_names = ['None', 'Light', 'Medium', 'Heavy', 'Extreme']
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(levels, [results[l]['avg_precision'] for l in levels], 'ro-', label='Precision')
    plt.plot(levels, [results[l]['avg_recall'] for l in levels], 'go-', label='Recall')
    plt.plot(levels, [results[l]['avg_f1'] for l in levels], 'bo-', label='F1 Score')
    
    plt.xlabel('Smoke Level')
    plt.ylabel('Score')
    plt.title('Edge Detection Performance vs. Smoke Level')
    plt.xticks(levels, [level_names[l] for l in levels])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results 