# **Object Detection and Training System**

This project implements an object detection model using PyTorch, including peak extraction from heatmaps, model architecture design, and training logic. Below is an overview of the files in this project and their purposes.

---

## **Project Overview**

The system is designed to detect objects in images by:
1. Extracting local maxima (peaks) from heatmaps generated during model training or inference.
2. Building a convolutional neural network with both down-sampling and up-sampling layers.
3. Training the model using a dataset with annotated object locations and evaluating its performance.

---

## **File Descriptions**

### **1. `models.py`**
- **Purpose:** Defines the core detection model (`Detector`) and its architecture.
- **Key Components:**
  - `Block`: A convolutional block with skip connections for down-sampling.
  - `UpBlock`: A transposed convolutional block with optional skip connections for up-sampling.
  - `Detector`: Combines multiple `Block` and `UpBlock` layers, with a classifier for predicting object locations and sizes.

### **2. `heatmap_utils.py`**
- **Purpose:** Implements functions for processing heatmaps and extracting object peaks.
- **Key Functions:**
  - `extract_peak`: Identifies local maxima in a 2D heatmap using max pooling and score thresholds.
  - `detections_to_heatmap`: Converts object detections into heatmaps for training purposes.

### **3. `train.py`**
- **Purpose:** Handles the training process for the detection model.
- **Key Features:**
  - Loads training and validation datasets using custom data loaders.
  - Uses `BCEWithLogitsLoss` for object presence detection and `MSELoss` for size prediction.
  - Supports logging metrics and visualizations to TensorBoard.
  - Saves the trained model for later use.

### **4. `utils.py`**
- **Purpose:** Provides utility functions for data preprocessing, evaluation, and visualization.
- **Key Functions:**
  - `load_detection_data`: Loads and transforms the training and validation datasets.
  - `accuracy`: Computes the accuracy of predictions against ground truth.

### **5. `transforms.py`**
- **Purpose:** Defines data augmentation and transformation pipelines for image preprocessing.
- **Key Components:**
  - `RandomHorizontalFlip`: Randomly flips images and their corresponding bounding boxes.
  - `ColorJitter`: Applies random changes to image brightness, contrast, and saturation.
  - `ToTensor` and `Normalize`: Converts images to PyTorch tensors and normalizes them.
  - `ToHeatmap`: Converts object annotations into heatmaps for training.

### **6. `main.py`**
- **Purpose:** Visualizes the model's detection performance on sample images.
- **Key Features:**
  - Loads a pre-trained model and runs inference on validation images.
  - Displays detections and ground truth annotations using bounding boxes and heatmaps.

### **7. `README.md`**
- **Purpose:** Provides an overview of the project and its components (this file).

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/object-detection-system.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Training**
To train the model, run:
```bash
python train.py --path_train <path_to_training_data> --path_valid <path_to_validation_data> --num_epoch 50 --log_dir logs/
```

### **Inference**
To visualize detections, run:
```bash
python main.py
```

---

## **Key Features**

- **Model Design:** Modular network with skip connections for improved feature learning.
- **Peak Extraction:** Efficient heatmap processing to extract object positions.
- **Data Augmentation:** Random transformations to improve model robustness.
- **Logging:** TensorBoard integration for monitoring training and evaluation metrics.

