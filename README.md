# PyTorch Vision Models and Tools

This repository contains implementations for various computer vision tasks, including image classification, semantic segmentation, and object detection. The repository is organized into five key modules:

---

## 1. Image Classification and Segmentation

### Overview
This module includes implementations for image classification and semantic segmentation using convolutional neural networks (CNNs) and fully convolutional networks (FCNs).

### Features
- **Custom Data Transforms**:
  - Includes resizing, cropping, padding, and augmentation techniques like random horizontal flips and color jitter.
  - Input normalization with ImageNet statistics.
- **CNN Classifier**:
  - Residual CNN architecture for class probability prediction.
- **FCN for Segmentation**:
  - Skip connections and transposed convolutions for dense predictions.
  - IoU metric evaluation for segmentation.

### Training and Visualization
- Training utilities with optimizers (SGD, Adam) and metrics for accuracy and IoU.
- Visualization of predictions for classification and segmentation.

---

## 2. Image Classification with CNN

### Overview
This module implements a CNN-based classification system for multi-class image classification tasks.

### Features
- **Custom Models**:
  - Residual CNN and lightweight CNN architectures.
- **Training Framework**:
  - Efficient training logic with cross-entropy loss, batch size, and learning rate customization.
  - Model checkpointing and validation.

### Visualization
- Displays random image samples with predicted labels and probabilities.

---

## 3. Multi-Class Image Classification

### Overview
Focuses on multi-class classification using linear classifiers, MLPs, and custom loss functions.

### Features
- **Classifier Options**:
  - Linear and MLP classifiers for different levels of complexity.
- **Training Support**:
  - Custom loss functions, accuracy metrics, and flexible training parameters.
- **Visualization Tools**:
  - Dataset visualization with image grids and predicted labels.

### Future Enhancements
- Adding support for more advanced architectures and automated hyperparameter tuning.

---

## 4. Object Detection

### Overview
Implements an object detection pipeline using heatmaps for object localization and size estimation.

### Features
- **Heatmap Processing**:
  - Local maxima extraction and bounding box prediction from heatmaps.
- **Detection Model**:
  - Fully convolutional architecture with skip connections and transposed convolutions.
- **Data Augmentation**:
  - Random horizontal flips, color jitter, and heatmap generation.

### Training and Evaluation
- Uses BCEWithLogitsLoss for object presence detection and MSELoss for size estimation.
- TensorBoard integration for monitoring metrics and predictions.

---

## 5. Transformations

### Overview
This module provides utilities for data preprocessing and augmentation.

### Key Transformations
- **RandomHorizontalFlip**:
  - Flips images and corresponding labels with a probability.
- **ColorJitter**:
  - Adjusts brightness, contrast, and saturation.
- **ToTensor**:
  - Converts images into PyTorch tensors.
- **Compose**:
  - Chains multiple transformations for sequential application.
