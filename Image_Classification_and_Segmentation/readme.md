# PyTorch Image Classification and Segmentation

## Overview
This repository includes implementations of image classification and semantic segmentation models using PyTorch. The project features custom data preprocessing, convolutional architectures, training utilities, and visualization tools.

---

## Features

### 1. **Custom Data Transforms**
- **Preprocessing Steps**:
  - Resizing, cropping, and padding for consistent input dimensions.
  - Augmentation techniques like random horizontal flips and color jitter.
  - Input normalization using ImageNet statistics.
- **Label Transformations**:
  - Converts segmentation labels to PyTorch tensors.
  - Generates visualization-friendly PIL images from tensors.

### 2. **CNN Classifier**
- **Architecture**:
  - Residual CNN with convolutional layers, batch normalization, and ReLU activation.
  - Input normalization for compatibility with pre-trained models.
- **Features**:
  - Processes input images of size `(3, 64, 64)`.
  - Outputs class probabilities for classification tasks.

### 3. **Fully Convolutional Network (FCN)**
- **Architecture**:
  - Downsampling using convolutional layers with residual connections.
  - Upsampling via transposed convolutions with skip connections for precise localization.
- **Features**:
  - Suitable for dense prediction tasks like semantic segmentation.
  - Evaluates predictions using metrics such as IoU (Intersection over Union).

### 4. **Training and Evaluation**
- **Optimization**:
  - Utilizes SGD (Stochastic Gradient Descent) or Adam optimizers.
  - Learning rate scheduling based on validation performance.
- **Metrics**:
  - Classification: Global accuracy and average accuracy.
  - Segmentation: IoU for evaluation.
- **Logging**:
  - TensorBoard integration for visualizing metrics and predictions.

### 5. **Model Saving and Loading**
- Supports saving trained models for reproducibility.
- Enables loading pre-trained models for evaluation or continued training.

### 6. **Visualization**
- Display input images alongside:
  - Predicted class probabilities for classification tasks.
  - Ground truth vs. predictions for segmentation tasks.

---

## Setup Instructions

### **Dependencies**
1. Install Python 3.8 or later.
2. Install required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

### **Directory Structure**
```
.
├── cnn.th                 # Trained CNN model weights
├── dense_transforms.py    # Data augmentation and preprocessing utilities
├── fcn.th                 # Trained FCN model weights
├── models.py              # Model definitions (CNN, FCN, etc.)
├── train_cnn.py           # Training script for CNN
├── train_fcn.py           # Training script for FCN
├── utils.py               # Helper functions (data loading, metrics, etc.)
```

---

## How to Run

### Training the Model
Run the following command to train the CNN or FCN model:
```bash
python train_cnn.py --path_train <path_to_training_data> --path_valid <path_to_validation_data> \
                    --batch_size 128 --num_epoch 50 --learning_rate 1e-3

python train_fcn.py --path_train <path_to_training_data> --path_valid <path_to_validation_data> \
                    --batch_size 32 --num_epoch 20 --learning_rate 1e-3
```

### Visualizing Predictions
To visualize predictions on a dataset, implement a `predict.py` script and use it with the trained model:
```bash
python predict.py <dataset_path> -n 6
```

---

## Example Usage

### Training
Train the model with the provided dataset:
```bash
python train_cnn.py --path_train data/train --path_valid data/valid --batch_size 128 --num_epoch 50 --learning_rate 0.001
python train_fcn.py --path_train data/train --path_valid data/valid --batch_size 32 --num_epoch 20 --learning_rate 0.001
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

