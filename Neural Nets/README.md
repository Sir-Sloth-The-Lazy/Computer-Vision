# Neural Network Evolution for Image Classification

This project documents the incremental improvements made to a neural network model designed to classify images of flowers (daisy, dandelion, roses, sunflowers, tulips).

## Project Overview

The goal was to build an image classifier and improve its accuracy through iterative enhancements, starting from a simple shallow network and evolving into a tuned deep neural network with regularization.

## Evolution Steps

### 1. Shallow Neural Network (Baseline)
**File:** `shallow_neural_network_for_imageclassification.py` / `linear_nn_for_images.py`

*   **Architecture:** We started with a simple linear model (or shallow network) consisting of a Flatten layer followed by a Dense output layer with Softmax activation.
*   **Performance:** The baseline accuracy was approximately **45%**.
*   **Limitation:** The model was too simple to capture complex patterns in the image data (high bias).

### 2. Hyperparameter Tuning with Weights & Biases
**File:** `weights&biases.py`

*   **Goal:** To systematic explore the hyperparameter space and identify the best configuration.
*   **Method:** We used [Weights & Biases (W&B)](https://wandb.ai/) to perform a hyperparameter sweep.
*   **Sweep Configuration:**
    *   **Method:** Grid Search
    *   **Metric:** Validation Accuracy (maximize)
    *   **Parameters Swept:**
        *   `batch_size`: [8, 16]
        *   `learning_rate`: [0.001, 0.0001]
        *   `hidden_nodes`: [64, 128]
        *   `img_size`: [16, 224]
        *   `epochs`: [5, 10]
*   **Outcome:** The sweep helped identify that a hidden layer size of **128 nodes** yielded better results.

### 3. Deep Neural Network with Regularization
**File:** `deep_neural_nets_imageclassification.py`

*   **Architecture:** Based on the tuning results, we implemented a deep neural network with:
    *   **Input Layer:** Flattened image vectors.
    *   **Hidden Layer:** One hidden layer with **128 nodes** and ReLU activation.
    *   **Output Layer:** Softmax for multi-class classification.
*   **Overfitting Prevention:** To improve generalization and prevent the model from memorizing the training data, we introduced several techniques:
    *   **Early Stopping:** Stops training when validation performance improves.
    *   **Regularization:** Penalizes complex models to reduce overfitting.
    *   **Batch Normalization:** Normalizes inputs to each layer for faster and more stable training.
    *   **Dropout:** Randomly ignores neurons during training to prevent co-adaptation.
*   **Result:** These incremental changes improved the model accuracy from **45% to 65%**.
