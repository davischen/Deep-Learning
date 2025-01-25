# Image Classification with CNN in PyTorch

## Overview
This project implements a convolutional neural network (CNN) for multi-class image classification using PyTorch. It includes functionality for training, validating, and saving models, as well as visualizing predictions on test data.

---

## Features

### Classification Loss
- Implements cross-entropy loss for multi-class classification.
- Calculates the loss between predicted class probabilities and true labels.

### CNN Classifier
- A simple CNN architecture:
  - Convolutional layer with ReLU activation and max pooling.
  - Fully connected layer for classification tasks.
- Processes input images of size `(3, 64, 64)`.

### Training Pipeline
- Trains the CNN using labeled datasets.
- Includes validation steps to monitor model performance.
- Uses SGD (Stochastic Gradient Descent) with momentum for optimization.

### Model Persistence
- Supports saving the model state to disk for reproducibility.
- Allows loading saved models for further training or evaluation.

### Prediction and Visualization
- Predicts class probabilities for input images.
- Displays input images alongside bar charts of predicted probabilities.

---

## Setup Instructions

### Dependencies
1. Install Python 3.8 or later.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Directory Structure
```
.
├── models/               # Model definitions and utilities
├── utils/                # Helper functions for data loading and evaluation
├── data/                 # Training and validation datasets
├── train.py              # Training script
├── predict.py            # Prediction and visualization script
├── requirements.txt      # Python dependencies
```

---

## How to Run

### Training the Model
Run the following command to train the CNN model:
```bash
python train.py --path_train <path_to_training_data> --path_valid <path_to_validation_data> --batch_size 128 --num_epoch 50 --learning_rate 1e-3
```

### Visualizing Predictions
To visualize predictions on a dataset, use:
```bash
python predict.py <dataset_path> -n 6
```

---

## Example Usage

### Training
Train the model with the provided dataset:
```bash
python train.py --path_train data/train --path_valid data/valid --batch_size 128 --num_epoch 50 --learning_rate 0.001
```

### Prediction Visualization
Visualize predictions for 6 random samples:
```bash
python predict.py data/val_data -n 6
```

---

## Future Enhancements
- Add support for data augmentation during training.
- Implement more advanced CNN architectures for improved performance.
- Include functionality for hyperparameter tuning and optimization.

