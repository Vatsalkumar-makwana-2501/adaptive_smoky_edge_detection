import os
import random
import numpy as np
import cv2
from skimage import exposure, io
from noise import pnoise2, snoise2
from PIL import Image, ImageFilter

def generate_perlin_noise(shape, scale=10.0):
    """Generate Perlin noise of the specified shape."""
    # Create a smaller noise field and resize
    small_shape = (int(shape[0] // scale), int(shape[1] // scale))
    noise = np.random.randn(*small_shape).astype(np.float32)
    # Resize to desired shape
    noise = cv2.resize(noise, (shape[1], shape[0]))
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def add_synthetic_smoke(image, smoke_intensity=0.5):
    """Add synthetic smoke using Perlin noise."""
    # Generate smoke-like noise
    smoke = generate_perlin_noise(image.shape[:2], scale=8.0)
    # Apply Gaussian blur to make it more smoke-like
    smoke = cv2.GaussianBlur(smoke, (15, 15), 0)
    
    # Convert to 3-channel if needed
    if len(smoke.shape) < 3:
        smoke = np.stack([smoke] * 3, axis=-1)
    
    # Create white smoke effect (higher values = whiter)
    white_smoke = np.ones_like(image) * 255
    
    # Blend original image with white smoke based on the smoke mask
    smoke_mask = smoke * smoke_intensity
    smoky_image = image * (1 - smoke_mask) + white_smoke * smoke_mask
    
    return np.clip(smoky_image, 0, 255).astype(np.uint8)

def add_gaussian_haze(image, intensity=0.3):
    """Add Gaussian haze/fog to the image."""
    fog = np.ones_like(image) * 255  # White fog
    return cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)

def create_smoke_textures():
    """Create synthetic smoke textures for blending."""
    textures = []
    for i in range(5):  # Generate 5 different smoke textures
        # Create noise of different scales
        texture = generate_perlin_noise((512, 512), scale=random.uniform(5, 15))
        # Apply different Gaussian blurs
        blur_size = random.choice([7, 11, 15, 21])
        texture = cv2.GaussianBlur(texture, (blur_size, blur_size), 0)
        textures.append(texture)
    return textures

def add_smoke_texture(image, smoke_textures=None, intensity=0.4):
    """Add smoke texture via alpha blending."""
    if smoke_textures is None:
        smoke_textures = create_smoke_textures()
    
    # Select a random smoke texture
    smoke = random.choice(smoke_textures)
    # Resize to match the image
    smoke = cv2.resize(smoke, (image.shape[1], image.shape[0]))
    
    # Convert to 3-channel
    if len(smoke.shape) < 3:
        smoke = np.stack([smoke] * 3, axis=-1)
    
    # Create white smoke
    white_smoke = np.ones_like(image) * 255
    
    # Blend with different intensity
    smoke_mask = smoke * intensity
    smoky_image = image * (1 - smoke_mask) + white_smoke * smoke_mask
    
    return np.clip(smoky_image, 0, 255).astype(np.uint8)

def apply_smoke_effect(image, smoke_level, method='perlin'):
    """Apply smoke effect to a single image.
    
    Args:
        image: Input image (BGR or RGB)
        smoke_level: Smoke intensity level (0-4)
                    0: none, 1: light, 2: medium, 3: heavy, 4: extreme
        method: Smoke generation method ('perlin', 'gaussian', 'texture')
        
    Returns:
        Smoke-augmented image
    """
    # Convert smoke_level to float intensity (0.0 - 1.0)
    if smoke_level == 0:  # none
        return image.copy()  # No smoke applied
        
    if isinstance(smoke_level, str):
        # Convert string level names to numeric indices
        smoke_levels = {
            'none': 0,
            'light': 1, 
            'medium': 2, 
            'heavy': 3, 
            'extreme': 4
        }
        if smoke_level in smoke_levels:
            smoke_level = smoke_levels[smoke_level]
        else:
            smoke_level = 2  # Default to medium
    
    # Normalize smoke level to 0.0-1.0 scale
    intensity = min(1.0, max(0.0, smoke_level / 4.0))
    
    # Apply the selected smoke generation method
    if method == 'perlin':
        return apply_perlin_smoke(image, intensity)
    elif method == 'gaussian':
        return apply_gaussian_smoke(image, intensity)
    elif method == 'texture':
        return apply_texture_smoke(image, intensity)
    else:
        # Default to perlin
        return apply_perlin_smoke(image, intensity)

def create_smoky_dataset(images, output_dir=None, methods=None):
    """Create a dataset of images with varying levels of smoke.
    
    Args:
        images: List of image file paths
        output_dir: Directory to save smoky images (optional)
        methods: List of smoke generation methods to use (default: all)
        
    Returns:
        List of dictionaries with original and smoky image paths
    """
    # Set default methods if none provided
    if methods is None:
        methods = ['perlin', 'gaussian', 'texture']
    
    # Set default output directory
    if output_dir is None:
        output_dir = 'smoky_dataset'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define smoke levels
    smoke_levels = [
        {'level': 0, 'name': 'none'},
        {'level': 1, 'name': 'light'},
        {'level': 2, 'name': 'medium'},
        {'level': 3, 'name': 'heavy'},
        {'level': 4, 'name': 'extreme'}
    ]
    
    # List to store all smoky image variations
    smoky_dataset = []
    
    # Process each image
    for i, image_path in enumerate(images):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(images)}")
            
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Get file name without extension
        base_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(base_name)
        
        # For each smoke level
        for level in smoke_levels:
            # Skip some methods for efficiency when no smoke is applied
            if level['level'] == 0:
                # For 'none' level, we only need one copy of the original image
                method = 'none'
                
                # Create output file path
                output_name = f"{file_name}_{level['name']}_{method}{file_ext}"
                output_path = os.path.join(output_dir, output_name)
                
                # Save original image
                cv2.imwrite(output_path, image)
                
                # Add to dataset
                smoky_dataset.append({
                    'original_path': image_path,
                    'smoky_path': output_path,
                    'smoke_level': level['level'],
                    'smoke_level_name': level['name'],
                    'smoke_method': method
                })
            else:
                # For each smoke generation method
                for method in methods:
                    # Apply smoke effect
                    smoky_image = apply_smoke_effect(image, level['level'], method)
                    
                    # Create output file path
                    output_name = f"{file_name}_{level['name']}_{method}{file_ext}"
                    output_path = os.path.join(output_dir, output_name)
                    
                    # Save smoky image
                    cv2.imwrite(output_path, smoky_image)
                    
                    # Add to dataset
                    smoky_dataset.append({
                        'original_path': image_path,
                        'smoky_path': output_path,
                        'smoke_level': level['level'],
                        'smoke_level_name': level['name'],
                        'smoke_method': method
                    })
    
    return smoky_dataset

def apply_perlin_smoke(image, intensity):
    """Apply Perlin noise-based smoke to an image.
    
    Args:
        image: Input image (BGR or RGB)
        intensity: Smoke intensity (0.0-1.0)
        
    Returns:
        Smoky image
    """
    # Copy the input image
    smoky_image = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create Perlin noise mask
    scale = 0.005 + 0.015 * intensity  # Adjust scale based on intensity
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    
    # Create noise array
    noise = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            noise[y, x] = pnoise2(x * scale, 
                                 y * scale, 
                                 octaves=octaves, 
                                 persistence=persistence, 
                                 lacunarity=lacunarity)
    
    # Normalize to 0-1
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    # Adjust contrast and brightness based on intensity
    noise = np.power(noise, 0.7 + 0.6 * intensity)  # Increase contrast
    noise_strength = 0.3 + 0.7 * intensity  # Adjust how strong the effect is
    
    # Apply smoke effect to each channel
    for c in range(3):
        channel = smoky_image[:, :, c].astype(np.float32)
        
        # Add smoke (weighted blend)
        # Higher intensity makes image brighter and lower contrast
        smoky_channel = (1.0 - noise_strength) * channel + noise_strength * (255.0 * noise)
        
        # Adjust contrast based on intensity
        smoky_channel = exposure.adjust_gamma(smoky_channel / 255.0, 1.0 / (1.0 + 0.5 * intensity)) * 255.0
        
        smoky_image[:, :, c] = np.clip(smoky_channel, 0, 255).astype(np.uint8)
    
    # Add blur for smoky effect
    blur_radius = int(3 + 7 * intensity)
    if blur_radius % 2 == 0:
        blur_radius += 1  # Ensure odd kernel size
    smoky_image = cv2.GaussianBlur(smoky_image, (blur_radius, blur_radius), 0)
    
    return smoky_image

def apply_gaussian_smoke(image, intensity):
    """Apply Gaussian noise-based smoke to an image.
    
    Args:
        image: Input image (BGR or RGB)
        intensity: Smoke intensity (0.0-1.0)
        
    Returns:
        Smoky image
    """
    # Copy the input image
    smoky_image = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Generate smoke mask using Gaussian blobs
    smoke_mask = np.zeros((height, width), dtype=np.float32)
    
    # Number of smoke particles increases with intensity
    num_particles = int(10 + 40 * intensity)
    
    # Generate random smoke particles
    for _ in range(num_particles):
        # Random position
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        
        # Random size (larger with higher intensity)
        size = random.randint(20, int(50 + 150 * intensity))
        
        # Random intensity
        strength = random.uniform(0.5, 1.0)
        
        # Create Gaussian blob
        for i in range(max(0, y - size), min(height, y + size)):
            for j in range(max(0, x - size), min(width, x + size)):
                # Distance from center
                dist = np.sqrt((i - y) ** 2 + (j - x) ** 2)
                
                # Gaussian falloff
                if dist < size:
                    falloff = np.exp(-0.5 * (dist / (size / 3)) ** 2)
                    smoke_mask[i, j] = max(smoke_mask[i, j], strength * falloff)
    
    # Normalize mask
    if smoke_mask.max() > 0:
        smoke_mask = smoke_mask / smoke_mask.max()
    
    # Adjust overall smoke strength
    smoke_strength = 0.3 + 0.7 * intensity
    smoke_mask = smoke_mask * smoke_strength
    
    # Apply smoke to each channel
    for c in range(3):
        channel = smoky_image[:, :, c].astype(np.float32)
        smoky_image[:, :, c] = np.clip((1.0 - smoke_mask) * channel + smoke_mask * 255.0, 0, 255).astype(np.uint8)
    
    # Add blur for smoky effect
    blur_radius = int(3 + 7 * intensity)
    if blur_radius % 2 == 0:
        blur_radius += 1  # Ensure odd kernel size
    smoky_image = cv2.GaussianBlur(smoky_image, (blur_radius, blur_radius), 0)
    
    return smoky_image

def apply_texture_smoke(image, intensity):
    """Apply texture-based smoke to an image.
    
    Args:
        image: Input image (BGR or RGB)
        intensity: Smoke intensity (0.0-1.0)
        
    Returns:
        Smoky image
    """
    # Copy the input image
    smoky_image = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create noise texture
    texture = np.zeros((height, width), dtype=np.float32)
    
    # Create fractal-like noise texture
    scale = 0.01
    for octave in range(6):
        octave_scale = scale * (2 ** octave)
        octave_weight = 1.0 / (2 ** octave)
        
        for y in range(height):
            for x in range(width):
                texture[y, x] += snoise2(x * octave_scale, y * octave_scale) * octave_weight
    
    # Normalize texture
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    
    # Apply threshold to create more defined smoke structures
    texture = np.power(texture, 2.0 - intensity)  # Higher intensity = more smoke
    
    # Convert to PIL for filter operations
    pil_image = Image.fromarray(cv2.cvtColor(smoky_image, cv2.COLOR_BGR2RGB))
    
    # Apply blur for smoky effect
    blur_size = int(3 + 7 * intensity)
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_size))
    
    # Convert back to numpy
    smoky_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Apply smoke texture
    smoke_strength = 0.3 + 0.7 * intensity
    for c in range(3):
        channel = smoky_image[:, :, c].astype(np.float32)
        smoky_image[:, :, c] = np.clip((1.0 - smoke_strength * texture) * channel + smoke_strength * texture * 255.0, 0, 255).astype(np.uint8)
    
    return smoky_image 