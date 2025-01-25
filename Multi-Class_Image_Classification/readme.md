# **Classification Model and Training System**

This project implements a classification system using PyTorch, including different classifier models (Linear and MLP), a custom loss function, and training logic. Below is an overview of the components and their functionalities.

---

## **File Descriptions**

### **1. `models.py`**
- **Purpose:** Defines the classifiers and custom loss function used in the project.
- **Key Components:**
  - `ClassificationLoss`: Custom loss function using `torch.nn.CrossEntropyLoss` for classification tasks.
  - `LinearClassifier`: A linear classifier with two fully connected layers.
  - `LinearClassifier2`: A simplified linear classifier with a single fully connected layer.
  - `MLPClassifier`: A multi-layer perceptron (MLP) classifier with multiple hidden layers and ReLU activations.
  - `model_factory`: A dictionary mapping model names to their corresponding classes for easy instantiation.

### **2. `train.py`**
- **Purpose:** Handles the training process for the classification models.
- **Key Features:**
  - Supports both `LinearClassifier` and `MLPClassifier` models.
  - Uses `ClassificationLoss` for calculating the training loss.
  - Includes accuracy evaluation for both training and validation datasets.
  - Implements checkpointing to save trained model weights.
  - Provides customizable training arguments such as batch size, learning rate, and number of epochs.

### **3. `visualize_data.py`**
- **Purpose:** Visualizes the dataset by displaying sample images for each label.
- **Key Features:**
  - Uses `SuperTuxDataset` for loading and iterating through the dataset.
  - Displays images with their corresponding labels in a grid format.
  - Allows customization of the number of samples per label using command-line arguments.

### **4. `utils.py`**
- **Purpose:** Provides utility functions for data loading and evaluation.
- **Key Functions:**
  - `load_data`: Loads and preprocesses the dataset for training and validation.
  - `accuracy`: Computes the accuracy of model predictions.
  - `LABEL_NAMES`: A list of label names for visualization purposes.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/classification-system.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Training**
To train a model, run:
```bash
python train.py --model <model_name> --path_train <path_to_training_data> --path_valid <path_to_validation_data> --epochs 50 --batch_size 128 --learning_rate 0.001
```
Example for MLPClassifier:
```bash
python train.py --model mlp --path_train data/train --path_valid data/valid --epochs 50
```

### **Visualization**
To visualize the dataset, run:
```bash
python visualize_data.py <path_to_dataset> -n 3
```

---

## **Key Features**

- **Model Choices:** Support for both linear and multi-layer perceptron classifiers.
- **Custom Loss Function:** `ClassificationLoss` for efficient computation of cross-entropy loss.
- **Visualization:** Easy-to-use dataset visualization tool.
- **Flexible Training:** Customizable training process with options for learning rate, batch size, and epochs.

---

## **Future Work**

- Add support for additional model architectures.
- Include automated hyperparameter tuning.
- Expand the dataset for more comprehensive training.
- Optimize training for distributed environments.
