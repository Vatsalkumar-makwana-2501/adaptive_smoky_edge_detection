
# Adaptive Edge Detection in Smoke-Augmented BSDS500 Images

## Project Overview
This project implements an adaptive edge detection system using the **BSDS500 dataset**. The system performs the following tasks:

- **Smoke Augmentation:** Adds synthetic smoke with varying densities to simulate real-world obstructions.
- **Edge Detection Algorithms:** Evaluates multiple edge detection algorithms on the smoke-augmented images.
- **Smoke Level Estimation Model:** Trains a CNN to classify the smoke level as Low, Medium, or High.
- **Adaptive Edge Detection:** Dynamically selects the most appropriate edge detection method based on predicted smoke levels.
- **Evaluation:** Compares predicted edges with ground truth using F1-score, IoU, and SSIM.

---

## Dataset Information
The BSDS500 dataset is assumed to have the following folder structure:
```
BSDS500/
├── images/
│   └── test/
│       ├── 123.jpg
│       └── ...
└── ground_truth/
    └── test/
        ├── 123.mat
        └── ...
```

**Download BSDS500 Dataset:**  
The dataset can be obtained from [Kaggle](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500).

---

## Installation Instructions
Follow these steps to set up the project:

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. **Create and Activate Virtual Environment**
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate

# Activate the virtual environment (Windows)
venv\Scripts\activate
```

### 3. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. **Run the Jupyter Notebook**
```bash
jupyter notebook
```
Open the notebook `Adaptive_Edge_Detection_Complete.ipynb` and run the cells to execute all stages of the pipeline.

### 2. **Run Python Script (Optional)**
To run the code as a standalone Python script:
```bash
python main.py
```

---

## Evaluation Metrics
The model evaluates edge detection performance using the following metrics:
- **F1-score:** Measures the balance between precision and recall.
- **IoU (Intersection over Union):** Evaluates the overlap between predicted and ground truth edges.
- **SSIM (Structural Similarity Index):** Measures the structural similarity between predicted and true edges.

---

## Key Features
- Dynamic edge detection model selection based on predicted smoke levels.
- Multiple edge detectors, including:
    - Canny
    - Sobel
    - Laplacian
    - HED (simulated using histogram equalization)
- CNN-based smoke level classification.
- Evaluation against ground truth data for accuracy assessment.

---

## Project Structure
```
├── BSDS500/
│   ├── images/
│   └── ground_truth/
├── Adaptive_Edge_Detection_Complete.ipynb
├── requirements.txt
└── README.md
```

---

## Contributing
We welcome contributions! Follow these steps to contribute:
1. Fork the project.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to your branch: `git push origin feature-name`.
5. Open a Pull Request.

---

## License
This project is licensed under the [MIT License](LICENSE).
