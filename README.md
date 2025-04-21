# Deep Learning - AG News Classification with LoRA
## 🚀 Project Overview

This project is a competition from Kaggle, we used the RoBERTa-base model with Low-Rank Adaptation (LoRA) on the AG News dataset. \
The goal is to achieve a classification accuracy of over 80% on the test set while keeping trainable parameters under 1 million.\
The model is trained using PyTorch and the HuggingFace Trainer API, incorporating LoRA adapters, AdamW optimizer, and optional learning rate scheduling for efficient fine-tuning.

---

## 📊 Dataset

We used the **AG News** dataset via HuggingFace Datasets. It contains 120,000 training samples and 7,600 test samples, evenly distributed across 4 classes:
- World
- Sports
- Business
- Sci/Tech

The dataset includes 120,000 training and 7,600 test samples.

---

## 🚀 Method

We iteratively optimized a RoBERTa-base model with Low-Rank Adaptation (LoRA), aiming to maintain under 1 million trainable parameters while maximizing classification accuracy on the AG News dataset.

Key strategies:

⚡️ LoRA Injection: Applied "query" and "value" projections across all transformer layers.

⚡️ Manual Training Loop: Replaced Trainer.train() with a custom training loop to record per-step train and evaluation loss/accuracy.

⚡️ Cosine Learning Rate Scheduler: Used in some configurations with 500 warmup steps to stabilize early training.

⚡️ Evaluation Every 500 Steps: Validation accuracy and loss were monitored regularly to support model tuning.

⚡️ Parameter-Constrained Setup: Final version disabled label smoothing, early stopping, and mixed-precision training to stay within parameter limits.

---

## 🔧 LoRA Configuration Experiments
We performed multiple rounds of LoRA configuration search and training experiments, summarized below:

| Round | LoRA Config                                | Bias    | Scheduler | Params    | Accuracy |
|-------|---------------------------------------------|---------|-----------|-----------|----------|
| R1    | `r=8/16`, α=`16/32`, `value/query+value`    | none    | None      | 962,308   | 89.65%   |
| R2    | `r=16`, α=32, `value` only                  | none    | cosine    | 888,580   | 94.49%   |
| R3    | `r=8`, α=16, `query+value`                  | none    | None      | 888,580   | 94.42%   |
| ✅ R4 | `r=8`, α=16, `query+value`                  | all     | None      | 992,268   | **94.6%** |

---

## 📈 Training Analysis
Plotted training and validation loss every 500 steps

Logged validation accuracy curve

Visualized confusion matrix and per-class performance

All experiments were conducted on Google Colab Pro with Tesla T4 GPU (16GB RAM).

---

## 📃 Files
README: Summarizes the approach, experiments, and results for this project.\
main.ipynb: Final version

---

## ✨Contributors
Jiale Cai jc12423@nyu.edu
Chen Yang cy2683@nyu.edu
Yinuo Wang yw8041@nyu.edu


