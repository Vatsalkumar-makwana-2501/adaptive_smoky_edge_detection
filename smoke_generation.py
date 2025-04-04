import os
import random
import numpy as np
import cv2

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

def create_smoky_dataset(images, output_dir='data/smoky_images', methods=['perlin', 'gaussian', 'texture']):
    """Create a dataset with varying smoke intensities using different methods."""
    # Create smoke textures once
    smoke_textures = create_smoke_textures()
    
    # Define smoke levels
    smoke_levels = [0.0, 0.2, 0.4, 0.6, 0.8]  # No smoke to heavy smoke
    level_names = ['none', 'light', 'medium', 'heavy', 'extreme']
    
    # Process each image
    results = []
    for idx, img_path in enumerate(images):
        if idx % 10 == 0:
            print(f"Processing image {idx+1}/{len(images)}")
            
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = os.path.basename(img_path).split('.')[0]
        
        # For each smoke level
        for level_idx, (level, level_name) in enumerate(zip(smoke_levels, level_names)):
            # Skip original image (just copy it)
            if level == 0.0:
                smoky_img = image.copy()
                method_used = 'none'
            else:
                # Choose a random method or follow specified methods
                if methods == 'random':
                    method = random.choice(['perlin', 'gaussian', 'texture'])
                else:
                    method = methods[idx % len(methods)]
                
                # Apply selected smoke method
                if method == 'perlin':
                    smoky_img = add_synthetic_smoke(image, level)
                    method_used = 'perlin'
                elif method == 'gaussian':
                    smoky_img = add_gaussian_haze(image, level)
                    method_used = 'gaussian'
                else:  # texture
                    smoky_img = add_smoke_texture(image, smoke_textures, level)
                    method_used = 'texture'
            
            # Save image
            output_path = os.path.join(output_dir, f"{img_name}_{level_name}_{method_used}.jpg")
            smoky_img_rgb = cv2.cvtColor(smoky_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, smoky_img_rgb)
            
            # Store metadata
            results.append({
                'original_path': img_path,
                'smoky_path': output_path,
                'smoke_level': level_idx,  # 0=none, 1=light, 2=medium, 3=heavy, 4=extreme
                'smoke_level_name': level_name,
                'smoke_method': method_used
            })
    
    return results 