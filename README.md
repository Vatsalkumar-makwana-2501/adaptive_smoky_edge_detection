# Adaptive Edge Detection System for Smoky Images

This project implements an adaptive edge detection system that works effectively on images affected by smoke at varying levels. The approach includes:

1. Creating a smoke-augmented dataset
2. Testing multiple edge detection algorithms
3. Training a smoke level estimation model
4. Implementing an adaptive edge detection system
5. Evaluating performance against ground truth

## Project Structure

- `main.py`: Main script that orchestrates the entire pipeline
- `utils.py`: Utilities for data loading and visualization
- `smoke_generation.py`: Functions to generate smoke-augmented images
- `edge_detection.py`: Implementation of various edge detection algorithms
- `smoke_level_model.py`: CNN model for smoke level estimation
- `evaluation.py`: Functions for evaluation and metrics calculation
- `requirements.txt`: Required dependencies

## Setup

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure the BSDS500 dataset is available in the project directory with the structure:
   ```
   BSDS500/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── ground_truth/
       ├── train/
       ├── val/
       └── test/
   ```

## Usage

Run the main script to execute the complete pipeline:

```
python main.py
```

This will:

1. Load the BSDS500 dataset
2. Create a smoke-augmented dataset
3. Test edge detection algorithms on smoky images
4. Train a smoke level estimation model
5. Test the adaptive edge detection system
6. Run a comprehensive evaluation

## Smoke Augmentation Methods

- **Perlin Noise**: Generates synthetic smoke patterns
- **Gaussian Haze**: Simulates uniform fog/haze
- **Texture Blending**: Overlays smoke textures with alpha blending

## Edge Detection Algorithms

- **Traditional Methods**: Canny, Sobel, Laplacian
- **Advanced Methods**: Guided Filter
- **Adaptive Approach**: Selects the best algorithm based on smoke level

## Smoke Level Estimation

The system includes a CNN model that classifies images into five smoke levels:

- None
- Light
- Medium
- Heavy
- Extreme

## Evaluation Metrics

- F1 Score (with tolerance)
- Intersection over Union (IoU)
- Structural Similarity Index (SSIM)

## Customization

You can modify the parameters in each module to experiment with:

- Different smoke generation techniques
- Edge detection parameters
- Model architectures
- Evaluation settings
