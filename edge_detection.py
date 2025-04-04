import numpy as np
import cv2
from skimage import feature, filters, color, transform
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.metrics import structural_similarity
from scipy import ndimage

# Check if OpenCV contrib modules are available
has_ximgproc = True
try:
    _ = cv2.ximgproc
except AttributeError:
    has_ximgproc = False
    print("Warning: cv2.ximgproc not available. Using fallback implementation for guided filter.")

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
    
    # Select the best edge detection method and parameters based on smoke level
    if smoke_level == 'none':
        # No smoke - use Sobel with standard parameters
        return sobel_edge(image, threshold=0.1, ksize=3)
    
    elif smoke_level == 'light':
        # Light smoke - use Sobel with enhanced contrast
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply contrast enhancement
        enhanced = cv2.equalizeHist(gray)
        return sobel_edge(enhanced, threshold=0.1, ksize=3)
    
    elif smoke_level == 'medium':
        # Medium smoke - use pre-processing with guided filter and then Sobel
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply guided filter for smoothing while preserving edges
        gray_float = gray.astype(np.float32) / 255.0
        filtered = cv2.ximgproc.guidedFilter(gray_float, gray_float, 8, 0.02)
        filtered = (filtered * 255).astype(np.uint8)
        # Apply contrast enhancement
        enhanced = cv2.equalizeHist(filtered)
        return sobel_edge(enhanced, threshold=0.1, ksize=5)
    
    elif smoke_level == 'heavy':
        # Heavy smoke - use denoising, enhancement and then Sobel with larger kernel
        enhanced = enhance_smoky_image(image, 0.75)
        if len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        # Apply contrast enhancement
        enhanced = cv2.equalizeHist(enhanced)
        return sobel_edge(enhanced, threshold=0.1, ksize=5)
    
    elif smoke_level == 'extreme':
        # Extreme smoke - heavy denoising, enhancement, and multi-scale approach
        enhanced = enhance_smoky_image(image, 1.0)
        if len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        # Apply contrast enhancement
        enhanced = cv2.equalizeHist(enhanced)
        
        # Multi-scale edge detection (combine edges at different scales)
        edges1 = sobel_edge(enhanced, threshold=0.1, ksize=3)
        edges2 = sobel_edge(enhanced, threshold=0.1, ksize=5)
        edges3 = sobel_edge(enhanced, threshold=0.1, ksize=7)
        
        # Combine the edges (take maximum response)
        combined_edges = np.maximum(edges1, np.maximum(edges2, edges3))
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