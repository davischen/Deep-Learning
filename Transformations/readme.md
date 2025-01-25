# Code Overview: High-Level Planner and Supporting Components for PyTux

This codebase implements a high-level planner for autonomous driving in the SuperTuxKart environment, along with supporting components for data collection, preprocessing, and model training. Below is a structured breakdown of the files and their functionality:

## Transformations for Data Augmentation

#### `RandomHorizontalFlip`
- Applies a horizontal flip to an image with a given probability (`flip_prob`).
- Modifies the corresponding label points to match the flipped image.

#### `Compose`
- Chains multiple transformations and applies them sequentially to the input image and labels.

#### `ColorJitter`
- Adjusts the brightness, contrast, saturation, and hue of an image.

#### `ToTensor`
- Converts an image to a PyTorch tensor and leaves labels unchanged.

---

## Soft-Argmax Implementation

### `spatial_argmax_b`
- Computes the soft-argmax of a 2D heatmap to return coordinates in normalized (-1 to 1) space.
- Uses weighted sums of coordinate grids with softmax probabilities.

### `spatial_argmax`
- Simplified version of `spatial_argmax_b`.
- Computes the same soft-argmax operation with concise code.

---

## Planner Models

### `Planner_b`
- Implements a high-resolution FCN (Fully Convolutional Network) with skip and residual connections.
- Uses down-sampling (via convolutions) and up-sampling (via transposed convolutions).
- Outputs normalized image coordinates for the aim point.

#### Subcomponents:
1. `Block`:
   - Residual block with convolutions, batch normalization, and ReLU activation.
   - Includes down-sampling if input and output dimensions differ.
2. `UpBlock`:
   - Transposed convolution for up-sampling.
   - Optionally uses skip connections to merge high-resolution features.

### `Planner`
- A simplified planner with sequential convolutional layers.
- Outputs normalized aim point coordinates based on the input image.

---

## Model Persistence

#### `save_model`
- Saves the planner model's state dictionary to a file (`planner.th`).

#### `load_model`
- Loads a saved planner model from the file (`planner.th`).

---

## Training Framework

### `train`
- Trains the `Planner` model using the Adam optimizer and L1 loss.
- Logs metrics such as training loss to TensorBoard.
- Supports continued training from a checkpoint.

#### Key Arguments:
- `--path_train`: Path to training dataset.
- `--log_dir`: Directory for TensorBoard logs.
- `--num_epoch`: Number of training epochs.
- `--learning_rate`: Learning rate for the optimizer.

#### Logging Functionality:
- Visualizes predictions versus ground truth using matplotlib.

---

## Data Handling

### `SuperTuxDataset`
- Loads image and label data from a directory (`drive_data`).
- Supports transformations such as converting images to tensors.

### `load_data`
- Wraps `SuperTuxDataset` in a PyTorch `DataLoader` for batch processing.

---

## PyTux Environment Wrapper

### `PyTux`
- Manages interaction with the SuperTuxKart simulator via `pystk`.
- Provides methods for:
  - Rolling out a track with a planner and controller.
  - Collecting image and label data.

#### Key Methods:
- `_point_on_track`: Calculates a 3D point at a specific distance along the track.
- `_to_image`: Projects a 3D point onto the 2D image plane.
- `rollout`: Executes a simulation on a track and optionally collects data.
- `close`: Cleans up resources and shuts down the simulator.

---

## Data Collection Script

### `main`
- Collects a dataset by running the PyTux environment with noisy controls.
- Saves images and corresponding aim points as `.png` and `.csv` files.

#### Key Arguments:
- `--track`: List of tracks for data collection.
- `--output`: Directory to save the dataset.
- `--n_images`: Total number of images to collect.
- `--aim_noise`: Standard deviation of noise added to aim points.
- `--vel_noise`: Standard deviation of noise added to velocity.

---

## Usage Instructions

### Training the Planner
Run the following command to train the planner:
```bash
python train.py --path_train drive_data --log_dir logs --num_epoch 60 --learning_rate 1e-3
```

### Testing the Planner
Run the planner on a specific track:
```bash
python test_planner.py track_name -v
```

### Collecting Data
Generate a dataset for training:
```bash
python collect_data.py track_name -o drive_data -n 10000
```

