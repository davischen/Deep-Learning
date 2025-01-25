# Multi-Class Image Classification with PyTorch

## **Overview**
This project implements a multi-class image classification system using PyTorch, with support for custom loss functions, linear classifiers, and multi-layer perceptrons. It also includes utilities for training, saving, and loading models.

---

## **Main Components**

### **1. Classification Loss**
A custom loss function designed for multi-class classification tasks. This module calculates cross-entropy loss, which measures the performance of the classification model by comparing predicted probabilities with actual labels.

- **Purpose**: 
  To evaluate the model's performance during training and guide the optimization process.

---

### **2. Linear Classifiers**
Baseline models for image classification tasks. These classifiers use fully connected layers to map input features directly to output classes.

- **LinearClassifier**: 
  A two-layer model with an intermediate hidden layer to process input features and produce predictions.

- **LinearClassifier2**: 
  A simpler implementation with a single fully connected layer, used as a baseline comparison.

---

### **3. MLP Classifier**
A Multi-Layer Perceptron (MLP) that incorporates multiple hidden layers and non-linear activation functions (e.g., ReLU). This model is designed to capture more complex patterns in the data compared to linear classifiers.

- **Purpose**:
  To improve the classification accuracy on non-linear datasets by leveraging deeper architectures.

---

### **4. Model Utilities**
Utility functions for managing models:
- **Saving Models**: 
  Saves the trained model's state to a file for later use.
- **Loading Models**: 
  Loads pre-trained models for inference or continued training.

---

### **5. Training Script**
A comprehensive training script is provided to train and evaluate the models. Key features include:
- Support for both linear and MLP classifiers.
- Adjustable hyperparameters such as learning rate, batch size, and number of epochs.
- Integration with GPU for faster training (if available).
- Tracks performance metrics such as training loss, accuracy, and validation accuracy.

---

## **Setup Instructions**

### **Dependencies**
Ensure you have Python 3.8 or later installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
