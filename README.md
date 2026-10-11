# Terrain Classification for Small-Legged Robots Using Deep Learning on Tactile Data

**Team:** F2  
**Team size:** 2

## 1. Project Overview

This project investigates terrain classification using tactile sensor data and machine learning. The goal is to classify the terrain encountered by a small-legged robot into one of eight categories using tactile readings and engineered features.

## 2. Dataset

- Total samples: 5,990
- Input features: 447
- Tactile readings: 414 values per sample, arranged as 6 channels × 69 readings
- Engineered features: 33
- Terrain classes: asphalt, cork, grass, gravel, lab, laminate wood, pebble and sand

The dataset was divided into training (70%), validation (15%) and test (15%) subsets using stratified sampling.

## 3. Methodology

We evaluated four classification approaches:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest
- Convolutional Neural Network (CNN)

The CNN processes the tactile readings using convolutional layers and combines the learned representation with all 33 engineered features. Input normalization parameters were estimated from the training data only.

Models were compared using accuracy and macro F1-score. The validation set was used for model selection, and the held-out test set was reserved for final evaluation.

## 4. Final Test Results

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 77.09% | 0.7698 |
| SVM (RBF) | 70.41% | 0.7093 |
| CNN | 61.96% | 0.5900 |
| Logistic Regression | 56.06% | 0.5593 |

Random Forest achieved the strongest performance among the evaluated models on this test split.

## 5. Findings and Limitations

The CNN performed well on grass and sand but struggled to identify cork. Random Forest achieved higher overall accuracy and macro F1-score than the CNN in this experiment.

These results indicate that the choice of model affects terrain classification performance. Further work could investigate feature selection, alternative neural network architectures and improved recognition of difficult terrain classes.

## 6. Repository Contents

- `notebooks/` — data preparation, baseline experiments and CNN implementation
- `results/` — model comparison tables and evaluation figures
- `docs/` — project report and presentation
- `data/` — dataset access instructions

## 7. Reproducibility

The experiments were conducted in Python using scikit-learn, PyTorch, pandas and NumPy. Refer to the notebooks for the complete preprocessing, training and evaluation workflow.
