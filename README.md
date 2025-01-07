# ECG GAN Project

This repository contains an implementation of a Generative Adversarial Network (GAN) for ECG signal analysis using the MIT-BIH dataset. The project is structured to facilitate training and evaluation, with configurations managed via YAML files.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

Electrocardiogram (ECG) signal analysis is crucial for diagnosing various heart conditions. This project leverages a GAN-based architecture to model ECG signals, enabling tasks such as anomaly detection. The GAN comprises an encoder, two generators, and a discriminator, trained to distinguish between normal and abnormal ECG patterns.

## Features

- **Modular Codebase:** Organized into separate modules for data loading, model definitions, training, and evaluation.
- **Configurable Parameters:** All hyperparameters and settings are managed via a YAML configuration file.
- **Reproducibility:** Fixed random seeds ensure consistent results across runs.
- **Performance Metrics:** Comprehensive evaluation with ROC curves, AUC, F1 scores, and accuracy metrics.
- **Visualization:** Generates plots to visualize training progress and evaluation results.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/ecg-gan-project.git
    cd ecg-gan-project
    ```

2. **Create a Virtual Environment (Optional but Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the MIT-BIH Dataset:**

    - Download the MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).
    - Extract the dataset and place it in the `data/mitdb/` directory.

## Configuration

All configuration parameters are managed via the `config/config.yaml` file. You can adjust parameters such as training settings, model architecture, and data preprocessing options here.

### Example `config/config.yaml`

```yaml
# Configuration file for ECG GAN Project

# General Settings
random_seed: 999
device: auto  # Options: 'cpu', 'cuda', 'auto'

# Data Settings
data:
  path: "data/mitdb/"
  resample_length: 128
  test_size: 0.2
  normal_classes: ['N']
  abnormal_classes: ['V', 'F', 'S', 'A', 'E', 'R', 'J']
  lowcut: 1      # Lower frequency bound (Hz)
  highcut: 50.0  # Upper frequency bound (Hz)
  fs: 360        # Sampling frequency (Hz)

# Training Settings
training:
  batch_size: 128
  num_epochs: 200
  learning_rate: 1e-5
  beta1: 0.5
  ngpu: 1
  nc: 2

# Model Settings
model:
  nz: 50
  nx: 50
  ngf: 64
  ndf: 64

# Optimization Weights
weights:
  diversity: 0.01
  netE: 1e-5
  confidence: 1

# Evaluation Settings
evaluation:
  batch_size: 50
  z_epochs: 100
  z_lr: 0.1
```

## Usage

### Setting Up the Configuration

Before running any scripts, ensure that the configuration file `config/config.yaml` is set up according to your needs.

**Key sections to customize:**

- **Device Selection:** Set `device` to `cpu`, `cuda`, or `auto` (automatic detection based on availability).
- **Data Path:** Ensure `data.path` points to the directory containing the MIT-BIH dataset.
- **Model Parameters:** Adjust settings like `nz`, `nx`, `ngf`, and `ndf` for experimenting with model architectures.
- **Training Parameters:** Modify `num_epochs`, `batch_size`, and `learning_rate` for different training setups.
- **Evaluation Parameters:** Set `z_epochs` and `z_lr` for optimizing latent variables during evaluation.

---

### Training

To train the model, follow these steps:

1. Ensure the dataset is correctly placed in the directory specified by `data.path` in `config/config.yaml`.

2. Run the training script:

    ```bash
    python src/train.py
    ```

**What happens during training:**

- **Data Processing:** ECG signals are loaded, filtered, normalized, and split into training and test sets.
- **Model Initialization:** The encoder, generators, and discriminator are initialized with random weights.
- **Training Loop:** Models are trained iteratively, with generator, discriminator, diversity, and confidence losses computed and optimized.
- **Logging and Visualization:** Training progress is logged, and sample plots (e.g., real vs. generated ECG signals) are displayed periodically.
- **Checkpoint Saving:** Model checkpoints are saved in the `outputs/models/` directory.

**Expected Outputs:**

- **Model Checkpoints:** Saved in `outputs/models/`.
- **Training Logs:** Stored in `outputs/logs/`.
- **Visualizations:** Generated plots are saved in `outputs/figures/`.

---

### Evaluation

To evaluate the trained model, follow these steps:

1. Ensure the trained model checkpoints are available in the `outputs/models/` directory.

2. Run the evaluation script:

    ```bash
    python src/evaluate.py
    ```

**What happens during evaluation:**

- **Model Loading:** Trained models are loaded from the `outputs/models/` directory.
- **Latent Variable Optimization:** Latent variables (`z`) are optimized for each test sample to minimize reconstruction error.
- **Error Computation:** Various error metrics are calculated to differentiate between normal and abnormal samples.
- **Performance Metrics:** Metrics such as AUC, F1 score, accuracy, precision, and recall are computed.
- **Visualization:** ROC curves and other evaluation plots are generated.

**Expected Outputs:**

- **ROC Curves:** Saved in `outputs/figures/` as `ROC.png`.
- **Metrics:** Printed in the terminal, including AUC, F1 score, recall, precision, and accuracy.
- **Error Logs:** Logged for further analysis.

---

### Visualizing Results

The evaluation process generates several visualizations that can help analyze model performance:

- **ROC Curve (`ROC.png`):** Shows the trade-off between true positive rate (TPR) and false positive rate (FPR).
- **Training Loss Plot:** Visualizes the evolution of generator and discriminator losses over training epochs.
- **Signal Comparison Plots:** Displays real vs. generated ECG signals and confidence maps.

These visualizations are saved in the `outputs/figures/` directory.

---

### Running Experiments

You can customize and run different experiments by modifying parameters in the `config/config.yaml` file. Examples include:

- **Changing Latent Dimensions:** Increase `nz` or `nx` to use higher-dimensional latent spaces.
- **Adjusting Diversity Weight:** Modify the `diversity` weight to control the variety of generated signals.
- **Optimizing Training Hyperparameters:** Experiment with different `learning_rate`, `beta1`, or batch sizes.

---

### Example Workflow

1. **Prepare the Dataset:**

    - Download the MIT-BIH dataset from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).
    - Extract the dataset and place it in the `data/mitdb/` directory.

2. **Train the Model:**

    Run the following command to start training:

    ```bash
    python src/train.py
    ```

3. **Evaluate the Model:**

    After training, run the evaluation script:

    ```bash
    python src/evaluate.py
    ```

4. **Analyze Results:**

    - Review metrics printed in the terminal.
    - Inspect ROC curves and training loss plots saved in the `outputs/figures/` directory.


### Descriptions:

1. **`models/`:** Stores the trained model checkpoints (`.pth` files) at different epochs and the final state after training.
2. **`logs/`:** Contains training logs for monitoring progress and debugging.
3. **`figures/`:** Includes visualizations such as ROC curves, training loss plots, and signal comparison plots.

---

## Dependencies

The project requires the following Python libraries. Install them using the provided `requirements.txt` file:

```plaintext
wfdb==3.3.2
numpy==1.23.5
torch==2.0.1
scikit-learn==1.2.2
scipy==1.10.1
pandas==1.5.3
matplotlib==3.7.1
pyyaml==6.0
```

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this code with proper attribution.

## Acknowledgements：

1. IT-BIH Dataset:  The ECG signals used in this project come from the MIT-BIH Arrhythmia Database.
2. PyTorch:  The GAN implementation is built using the PyTorch library.
3. PhysioNet: Thanks to PhysioNet for providing access to the dataset and resources for ECG signal analysis.
