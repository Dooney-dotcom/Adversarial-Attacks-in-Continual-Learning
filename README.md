# Adversarially Attacks in Continual Learning

> **An experimental study on the intersection of Continual Learning (CL) and Adversarial Robustness.**

This repository contains the implementation and experimental results for investigating how standard Continual Learning methods, specifically **Elastic Weight Consolidation (EWC)** and **Replay-Based Learning**, behave under adversarial attacks. We propose and evaluate two defense mechanisms: **Task-Aware Boundary Augmentation (TABA)** and **Adversary Aware Continual Learning (AACL)**.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
    - [Experiment 1: EWC & TABA](#experiment-1-ewc--taba)
    - [Experiment 2: Replay & AACL](#experiment-2-replay--aacl)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [License](#license)

---

## Overview

Continual Learning (CL) aims to enable models to learn sequentially from a stream of data without forgetting previously acquired knowledge. However, the robustness of these models against adversarial attacks during the continual learning process is often overlooked. 

This project explores:
1.  **Robustness Forgetting**
2.  **Adversarial Vulnerability**
3.  **Defenses**

All experiments are conducted on the **MNIST** dataset in a **Task-Incremental** setting (Task 1: Classes 0-4, Task 2: Classes 5-9).

---

## Key Features

- **Reproducible Pipelines**: Complete training and evaluation loops for EWC and Replay-based methods.
- **Adversarial Attacks**:
    - **Clean Label Attacks**: Evaluating robustness against gradient-based perturbations (PGD).
    - **Backdoor Attacks**: Injecting triggers to compromise model integrity.
- **Novel Defenses**:
    - **TABA**: Augments decision boundaries to be robust against perturbations while mitigating forgetting.
    - **AACL**: A distillation-based approach to suppress backdoor triggers in replay buffers.
- **Comprehensive Metrics**: evaluating Standard Accuracy (SA), Robust Accuracy (RA), and Attack Success Rate (ASR).

---

## Methodology

### Experiment 1: EWC & TABA
**Regularization-Based Continual Learning**

- **CL Method**: **Elastic Weight Consolidation (EWC)**. Preserves important weights for previous tasks.
- **Threat Model**: **Clean Label Poisoning**. Access to the training data of the current task to degrade test-time robustness.
- **Defense**: **Task-Aware Boundary Augmentation (TABA)**.
    - TABA actively augments the training data with adversarial examples during the CL process.
    - It ensures that the decision boundary remains robust for both the current and previous tasks, effectively reducing "robustness forgetting."

### Experiment 2: Replay & AACL
**Replay-Based Continual Learning**

- **CL Method**: **Experience Replay**. Stores a small buffer of data from previous tasks to rehearse during new task training.
- **Threat Model**: **Backdoor Attack**. The adversary injects a fixed trigger (pattern) into a subset of training data, causing the model to misclassify triggered images into a target class.
- **Defense**: **Adversary Aware Continual Learning (AACL)**.
    - AACL utilizes knowledge distillation to align the feature representations of clean and triggered samples.
    - It effectively "unlearns" the backdoor behavior while maintaining high accuracy on legitimate data.

---

## Project Structure

The repository is organized into distinct modules for each experimental pipeline.

```plaintext
.
├── init.sh                 # Environment setup script
├── main.py                 # Primary entry point for experiments
├── taba/                   # EXPERIMENT 1: EWC + TABA
│   ├── attacks.py          # PGD and Clean-Label attack logic
│   ├── ewc.py              # Elastic Weight Consolidation implementation
│   ├── trainer.py          # Training loops for Baseline and TABA
│   ├── model.py            # MLP Architecture
│   ├── config.py           # Hyperparameters
│   └── ...
└── aacl/                   # EXPERIMENT 2: Replay + AACL
    ├── losses.py           # Custom loss functions for distillation
    ├── trainer.py          # Unified learner for standard/vulnerable/defense modes
    ├── model.py            # CNN Architecture
    ├── config.py           # Hyperparameters
    └── ...
```

---

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

We provide an initialization script to set up the virtual environment and install all dependencies automatically.

```bash
# Make the script executable
chmod u+x init.sh

# Run the setup script
./init.sh
```

This will create a `.venv` directory and install the required packages (PyTorch, Pandas, Numpy, etc.).

---

## Usage

The `main.py` script serves as the unified interface for running the experiments. You can select the experiment mode using command-line arguments.

### Running Experiment 1 (TABA)
To run the EWC baseline, apply the clean-label attack, and verify the TABA defense:

```bash
python3 main.py taba
```

### Running Experiment 2 (AACL)
To run the Replay baseline, inject the backdoor, and verify the AACL defense:

```bash
python3 main.py aacl
```

---

## Experimental Results

### TABA Results (Regularization)

Comparison of Standard Accuracy (SA) and Robust Accuracy (RA) on Task A and B.

| Task | Metric | EWC Baseline | EWC Attacked | TABA Defense |
| :--- | :--- | :--- | :--- | :--- |
| **A** | **Standard Accuracy** | 94.08% | 51.14% | 86.79% |
| **A** | **Robust Accuracy** | 88.27% | 25.63% | **87.45%** |
| **B** | **Standard Accuracy** | 78.36% | 92.16% | 87.00% |
| **B** | **Robust Accuracy** | 60.75% | — | **79.90%** |

> **Insight**: TABA successfully restores adversarial robustness on the initial task (Task A) that was otherwise lost due to the attack, while also improving robustness generalization to the new task (Task B).

### AACL Results (Replay)

Comparison of accuracy and Backdoor Attack Success Rate (ASR).

| Metric | Baseline (Task 1) | Vulnerable | AACL Defense |
| :--- | :--- | :--- | :--- |
| **Task A Acc (0-4)** | 99.36% | 94.78% | 88.13% |
| **Task B Acc (5-9)** | N/A | 96.98% | 97.47% |
| **Backdoor ASR** | 0.00% | 99.56% | **0.04%** |

> **Insight**: The vulnerable model falls prey to the backdoor with nearly 100% success rate. AACL effectively neutralizes the backdoor (0.04% ASR) with a manageable trade-off in clean accuracy.

---

## License

This project is intended for educational and research purposes.
