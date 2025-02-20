# Zipperhead

## Overview
**Zipperhead** is a project designed to explore the differences between **Supervised Fine-Tuning (SFT)** and **Guided Reward Policy Optimization (GRPO)** for base models. The goal is to learn how reinforcement learning techniques can be applied to fine-tune a simple model trained on generated mathematical problems.

## Experiment Setup

### Objective
The primary objective is to assess the impact of **GRPO** on a distilled model trained on a biased dataset. We will use a custom **model** trained on a small, synthetic dataset of mathematical problems to iterate efficiently and compare the performance of **SFT** and **GRPO**.

### Methodology

#### Model Structure
We use a simple **autoencoder-style model** for experimentation purposes, then introduce a transformer model for comparison.

#### Data Generation
We generate two datasets to test different training scenarios:

1. **Biased Dataset:** Contains only **odd** `y` values to simulate a limited training scenario.
2. **Full Dataset:** Includes both **odd** and **even** `y` values for comprehensive training.

#### Training Phases
- **Supervised Fine-Tuning (SFT):** The model is first trained on the biased dataset and later fine-tuned on the full dataset.
- **GRPO Fine-Tuning:** Reinforcement learning techniques are applied to optimize performance using a reward function.

## Reward Function Design

### Objective
The reward function is designed to encourage the model to generate outputs that are **closer to target values** while also incentivizing exploration of even `y` values.

### Implementation
- The reward function assigns **higher rewards** for outputs with lower errors relative to true values.
- A bonus reward is given for generating even `y` values to promote exploration of unseen outputs.

## Exploration Strategies
A key challenge in GRPO is ensuring the model explores a **diverse range of outputs** to improve learning. Two strategies are implemented:

- **Exploration Noise:** Adding randomness to the model’s outputs to prevent overfitting to biased data.
- **Adaptive Reward Function:** Modifying the reward structure to reinforce the generation of even `y` values.

## Hyperparameter Tuning

We experiment with several key hyperparameters to optimize the model’s performance:

| Hyperparameter       | Description                                              |
|----------------------|----------------------------------------------------------|
| `even_data_ratio`   | Fraction of even `y` values in the initial training set  |
| `exploration_noise` | Random noise added to outputs for better exploration     |
| `even_bonus`        | Additional reward for generating even `y` values         |
| `grpo_steps`        | Number of GRPO iterations                               |
| `learning_rate`     | Learning rate for GRPO updates                          |

## Implementation

### Code Structure
The implementation includes the following components:

1. **Model Definition:** A simple autoencoder / decoder
2. **Data Generation:** Functions to create biased and full datasets.
3. **Training Functions:**
   - **SFT Training:** Uses supervised loss to train on biased and full datasets.
   - **GRPO Training:** Implements reinforcement learning with a reward function.
4. **Hyperparameter Tuning:** Adjustable parameters to optimize training.

### Supervised Fine-Tuning (SFT)
SFT is implemented using **MSE loss** and an optimizer to minimize prediction errors on the biased dataset before fine-tuning on the full dataset.

### GRPO Fine-Tuning
GRPO is implemented with:
- **A reward function** that assigns values based on error and even-number bonuses.
- **Exploration noise** to encourage variability in outputs.
- **Gradient updates** using reinforcement learning techniques.

## Conclusion
The **Zipperhead** project provides a hands-on approach to understanding the differences between **SFT** and **GRPO** in fine-tuning base models. By experimenting with biased datasets, reward functions, and exploration strategies, we aim to gain insights into how reinforcement learning can improve model generalization beyond traditional supervised training.

This project serves as a sandbox for further experimentation and improvements in model fine-tuning strategies.

## Repository Structure
```
zipperhead/
├── venv/               # Python virtual environment
├── README.md           # Project documentation
├── .gitignore          # Ignored files
├── zipperhead.ipynb    # Jupyter Notebook for experiments
```

## Getting Started
To activate the virtual environment:
```bash
source venv/bin/activate
```

Run the notebook using:
```bash
jupyter notebook zipperhead.ipynb
```

---

This README provides a structured summary of our **Zipperhead** experiment. Feel free to modify or expand it as the project evolves!


