# **Convolutional Neural Network (CNN) Classifier System**

This project implements a CNN-based classification system using PyTorch. It includes a custom CNN model, training logic, dataset loading, and evaluation utilities.

---

## **File Descriptions**

### **1. `models.py`**
- **Purpose:** Defines the CNN model and the custom classification loss function.
- **Key Components:**
  - `ClassificationLoss`: A custom loss function using PyTorch's cross-entropy loss for classification tasks.
  - `CNNClassifier`: A simple CNN model with a single convolutional layer followed by a fully connected layer for classification.
  - `save_model`: Saves the trained model to a file.
  - `load_model`: Loads a saved model from a file.

### **2. `train.py`**
- **Purpose:** Handles the training process for the CNN model.
- **Key Features:**
  - Trains the CNN model on labeled image data.
  - Computes loss using `ClassificationLoss`.
  - Evaluates the model on a validation dataset and prints training/validation metrics.
  - Supports customization of training parameters such as batch size, learning rate, and number of epochs.
  - Saves the trained model after training.

### **3. `utils.py`**
- **Purpose:** Provides utility functions for data loading and evaluation.
- **Key Functions:**
  - `SuperTuxDataset`: A custom dataset class for loading image data and their labels from CSV files.
  - `load_data`: Loads and preprocesses the dataset into batches for training and validation.
  - `accuracy`: Computes the accuracy of model predictions.
  - `LABEL_NAMES`: A list of label names used for classification.

### **4. `visualize_data.py`**
- **Purpose:** Visualizes the dataset and model predictions.
- **Key Features:**
  - Displays a random selection of images from the dataset with their true labels.
  - Shows prediction probabilities as horizontal bar charts for each image.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/cnn-classifier.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Training**
To train the CNN model, run:
```bash
python train.py --path_train <path_to_training_data> --path_valid <path_to_validation_data> --num_epoch 50 --batch_size 128 --learning_rate 0.001
```
Example:
```bash
python train.py --path_train data/train --path_valid data/valid --num_epoch 50
```

### **Visualization**
To visualize the dataset and predictions, run:
```bash
python visualize_data.py <path_to_dataset> -n 6
```
Example:
```bash
python visualize_data.py val_data -n 6
```

---

## **Key Features**

- **Custom CNN Model:** A lightweight CNN architecture for image classification.
- **Custom Loss Function:** Implements `ClassificationLoss` for efficient cross-entropy computation.
- **Dataset Management:** Supports easy loading and preprocessing of labeled image datasets.
- **Visualization Tools:** Provides utilities for visualizing dataset samples and model predictions.

---

## **Future Work**

- Expand the CNN architecture with additional layers and features.
- Add support for data augmentation to improve model generalization.
- Include automated hyperparameter tuning.
- Optimize the training process for larger datasets and distributed training environments.
