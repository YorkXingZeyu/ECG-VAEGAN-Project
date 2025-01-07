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
