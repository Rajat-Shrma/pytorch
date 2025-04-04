# PyTorch Neural Network Experiments

A collection of Jupyter notebooks demonstrating the evolution of neural network models for classifying the Fashion MNIST dataset using PyTorch, with a focus on improving accuracy and reducing overfitting.

## Table of Contents
- [About](#about)
- [Repository Contents](#repository-contents)
- [Journey to Improve Accuracy](#journey-to-improve-accuracy)


## About

This repository contains a series of Jupyter notebooks that showcase the development and optimization of neural networks using PyTorch for the Fashion MNIST classification task. Starting with a simple neural network, the project progresses through fully connected ANNs, CNNs, hyperparameter tuning with Optuna, and transfer learning, achieving a final accuracy above 95%. Each notebook represents a step in the journey to improve model performance, addressing challenges like overfitting and low accuracy.

## Repository Contents

The repository includes the following files:

1. **`pytorch_training_pipeline (1).ipynb`**
   - Implements a basic training pipeline with a simple neural network (`MySimpleNN`) for binary classification.
   - Features: Custom forward pass, loss function (binary cross-entropy), and manual gradient updates.
   - Accuracy: Not specified (binary task, threshold-based evaluation).

2. **`Building_ANN_using_pytorch_CPU.ipynb`**
   - Builds a fully connected ANN for Fashion MNIST (small dataset: 6,000 images).
   - Model: 3-layer MLP (784 → 128 → 64 → 10).
   - Accuracy: ~84%.
   - Issues: Overfitting and relatively low accuracy.

3. **`Building_ANN_using_pytorch_GPU.ipynb`**
   - Extends the ANN to the full Fashion MNIST dataset (60,000 images) using GPU.
   - Model: Same 3-layer MLP architecture.
   - Accuracy: ~88%.
   - Improvements: Reduced overfitting slightly due to larger dataset; still low accuracy.

4. **`Copy_of_Optimizing_Building_ANN_using_pytorch.ipynb`**
   - Optimizes the ANN using Optuna for hyperparameter tuning.
   - Model: Dynamic MLP with variable hidden layers, neurons, dropout, etc.
   - Best Accuracy: 89.21% (Trial 2 parameters).
   - Improvements: Better hyperparameter selection; overfitting persists.

5. **`Building_CNN_using_pytorch_GPU.ipynb`**
   - Transitions to a CNN architecture for improved feature extraction.
   - Model: 2 convolutional layers with batch norm, max pooling, followed by a classifier.
   - Accuracy: 92%.
   - Improvements: Significant accuracy boost and reduced overfitting.

6. **`Building_transfer_learning_CNN_using_pytorch_GPU.ipynb`**
   - Implements a CNN with insights from transfer learning (custom architecture inspired by VGG16).
   - Model: 2 convolutional layers with batch norm, followed by a tuned classifier.
   - Accuracy: >95%.
   - Improvements: Best performance with minimal overfitting.

## Journey to Improve Accuracy

This section details the step-by-step process of working with the Fashion MNIST dataset and models to increase accuracy and address overfitting:

### Step 1: Basic Training Pipeline
- **File**: `pytorch_training_pipeline (1).ipynb`
- **Approach**: Built a simple neural network (`MySimpleNN`) with a single layer for binary classification to understand PyTorch’s training pipeline.
- **Dataset**: Not Fashion MNIST-specific; likely a synthetic or simpler dataset.
- **Challenges**: Limited to binary tasks; no complex feature extraction.
- **Outcome**: Established a foundation for PyTorch implementation (manual gradients, loss calculation).

### Step 2: Initial ANN on Small Dataset
- **File**: `Building_ANN_using_pytorch_CPU.ipynb`
- **Approach**: Created a 3-layer MLP (784 → 128 → 64 → 10) on a subset of Fashion MNIST (6,000 images).
- **Dataset**: Fashion MNIST small (6,000 training images).
- **Accuracy**: ~84%.
- **Challenges**: Overfitting due to small dataset size; low accuracy due to simplistic architecture.
- **Lesson**: Needed a larger dataset and more robust model.

### Step 3: Scaling to Full Dataset with GPU
- **File**: `Building_ANN_using_pytorch_GPU.ipynb`
- **Approach**: Used the same MLP architecture on the full Fashion MNIST dataset (60,000 images) with GPU acceleration.
- **Dataset**: Fashion MNIST full (60,000 training, 10,000 test images).
- **Accuracy**: ~88%.
- **Improvements**: Larger dataset reduced overfitting slightly; GPU sped up training.
- **Challenges**: Accuracy still low; overfitting persisted due to lack of regularization and feature extraction.

### Step 4: Hyperparameter Tuning with Optuna
- **File**: `Copy_of_Optimizing_Building_ANN_using_pytorch.ipynb`
- **Approach**: Introduced Optuna to tune hyperparameters (layers, neurons, learning rate, dropout, etc.) for the ANN.
- **Dataset**: Full Fashion MNIST.
- **Best Parameters**: 2 hidden layers, 128 neurons, 100 epochs, Adam optimizer, etc.
- **Accuracy**: 89.21%.
- **Improvements**: Better hyperparameter selection improved performance.
- **Challenges**: Overfitting remained; MLP struggled with image data.

### Step 5: Transition to CNN
- **File**: `Building_CNN_using_pytorch_GPU.ipynb`
- **Approach**: Switched to a CNN with 2 convolutional layers (32 and 64 filters), batch normalization, and max pooling, followed by a classifier.
- **Dataset**: Full Fashion MNIST.
- **Accuracy**: 92%.
- **Improvements**: CNN’s spatial feature extraction significantly boosted accuracy; batch norm and dropout reduced overfitting.
- **Lesson**: CNNs are better suited for image data than MLPs.

### Step 6: Enhanced CNN with Transfer Learning Insights
- **File**: `Building_transfer_learning_CNN_using_pytorch_GPU.ipynb`
- **Approach**: Designed a custom CNN inspired by VGG16, with 2 convolutional layers, batch norm, and a refined classifier with dropout.
- **Dataset**: Full Fashion MNIST.
- **Accuracy**: >95%.
- **Improvements**: Optimal architecture, regularization (dropout), and larger dataset minimized overfitting and maximized accuracy.
- **Outcome**: Achieved the best performance, balancing accuracy and generalization.

