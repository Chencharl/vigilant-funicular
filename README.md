# Deep Learning - AG News Classification with LoRA
## 🚀 Project Overview

This project is a competition from Kaggle, we used the RoBERTa-base model with Low-Rank Adaptation (LoRA) on the AG News dataset. \
The goal is to achieve a classification accuracy of over 80% on the test set while keeping trainable parameters under 1 million.\
The model is trained using PyTorch and the HuggingFace Trainer API, incorporating LoRA adapters, AdamW optimizer, and optional learning rate scheduling for efficient fine-tuning.

---

## 📊 Dataset

We used the AG News dataset via HuggingFace Datasets. It contains 120,000 training samples and 7,600 test samples, evenly distributed across 4 classes:
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
| R1    | `r=8/16`, α=`16/32`, `value/query+value`    | none    | None      | 962,308   | 89.7%   |
| R2    | `r=16`, α=32, `value` only                  | none    | Cosine + Warmup    | 888,580   | 94.5%   |
| R3    | `r=8`, α=16, `query+value`                  | none    | Cosine + Warmup     | 888,580   | 94.4%   |
| ✅ R4 | `r=8`, α=16, `query+value`                  | all     | Cosine + Warmup      | 992,268   | **94.6%** |

---

## 📈 Training Analysis
Plotted training and validation loss every 500 steps

Logged validation accuracy curve

Visualized confusion matrix and per-class performance

All experiments were conducted on Google Colab Pro with Tesla T4 GPU (16GB RAM).

---

## 📃 Reference 
Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang,S.; and Chen, W. 2021. Lora: Low-rank adaptation of large
language models. arXiv preprint arXiv:2106.09685.

Loshchilov, I., and Hutter, F. 2016. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint
arXiv:1608.03983.

---

## 📃 Files
README: Summarizes the approach, experiments, and results for this project.\
code_file.py: Full script of our final model, including dataset loading, LoRA model setup, training via HuggingFace Trainer, evaluation, and test prediction. Outputs include final accuracy and a CSV for submission.\
Project2.ipynb: Contains the final version of the model with the best performance. Run this file to reproduce the results.

To run: `python code_file.py` in Colab or a local Python environment with required packages.
---

## ✨Contributors
Jiale Cai jc12423@nyu.edu
Chen Yang cy2683@nyu.edu
Yinuo Wang yw8041@nyu.edu


