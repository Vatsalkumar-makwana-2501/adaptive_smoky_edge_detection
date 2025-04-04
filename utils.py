import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat

def load_dataset(base_dir, mode='train'):
    """Load images and ground truth from BSDS500 dataset."""
    image_dir = os.path.join(base_dir, 'images')
    gt_dir = os.path.join(base_dir, 'ground_truth')
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, mode, '*.jpg')))
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, mode, '*.mat')))
    
    print(f"Found {len(image_paths)} images and {len(gt_paths)} ground truth files in {mode} set.")
    return image_paths, gt_paths

def display_sample(image_path, gt_path):
    """Display a sample image and its ground truth edges."""
    # Load and display image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load ground truth
    gt_data = loadmat(gt_path)
    # BSDS500 ground truth contains multiple human annotations
    # We'll use the first one for display
    boundaries = gt_data['groundTruth'][0,0]['Boundaries'][0,0]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(boundaries, cmap='gray')
    plt.title('Ground Truth Edges')
    plt.axis('off')
    plt.show()
    
    return img, boundaries

def load_ground_truth(gt_path):
    """Load ground truth edge map from MATLAB .mat file.
    
    Args:
        gt_path: Path to the ground truth .mat file
        
    Returns:
        Ground truth edge map as a numpy array, or None if it can't be loaded
    """
    try:
        from scipy import io
        import numpy as np
        
        # Load the .mat file
        mat_data = io.loadmat(gt_path)
        
        # BSDS500 specific format
        if 'groundTruth' in mat_data:
            gt_data = mat_data['groundTruth']
            
            # Create an average boundary from all human annotations
            h, w = 0, 0
            
            # First determine image dimensions from the first annotation
            if gt_data.shape[1] > 0:
                first_gt = gt_data[0, 0]
                if 'Boundaries' in first_gt.dtype.names:
                    boundary_data = first_gt['Boundaries']
                    if isinstance(boundary_data, np.ndarray):
                        if boundary_data.size > 0:
                            b = boundary_data[0, 0]
                            h, w = b.shape
                
            # If we found dimensions, create an average boundary
            if h > 0 and w > 0:
                # Create empty image to accumulate boundaries
                avg_boundary = np.zeros((h, w), dtype=np.float32)
                count = 0
                
                # Accumulate all boundaries
                for i in range(gt_data.shape[1]):
                    gt = gt_data[0, i]
                    if 'Boundaries' in gt.dtype.names:
                        boundary_data = gt['Boundaries']
                        if isinstance(boundary_data, np.ndarray) and boundary_data.size > 0:
                            b = boundary_data[0, 0]
                            if b.shape == (h, w):
                                avg_boundary += b
                                count += 1
                
                # Normalize
                if count > 0:
                    avg_boundary /= count
                    # Convert to 0-255 range for easier processing
                    return (avg_boundary * 255).astype(np.uint8)
        
        # General case - try to find the first suitable 2D array
        for key in mat_data:
            if isinstance(mat_data[key], np.ndarray):
                # Check if it's a 2D array
                if mat_data[key].ndim == 2:
                    # Convert to 0-255 range if needed
                    arr = mat_data[key]
                    if arr.dtype == bool:
                        arr = arr.astype(np.uint8) * 255
                    elif arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    return arr
        
        print(f"Could not extract edge data from {gt_path}")
        return None
        
    except Exception as e:
        print(f"Error loading ground truth from {gt_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_corresponding_gt(image_path, gt_dir):
    """Find the corresponding ground truth file for an image.
    
    Args:
        image_path: Path to the original image
        gt_dir: Directory with ground truth files
        
    Returns:
        Path to the corresponding ground truth file, or None if not found
    """
    # Extract the base filename without extension
    basename = os.path.basename(image_path)
    image_id = os.path.splitext(basename)[0]
    
    # Search for .mat files
    for dirpath, _, filenames in os.walk(gt_dir):
        for filename in filenames:
            if filename.startswith(image_id) and filename.endswith('.mat'):
                return os.path.join(dirpath, filename)
    
    return None

def setup_directories():
    """Create necessary directories for the project."""
    dirs = ['data/smoky_images', 'data/edge_maps', 'models', 'results']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Project directories created.") 