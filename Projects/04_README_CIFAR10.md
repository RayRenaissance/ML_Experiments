# 04_Image_Classification_on_CIFAR10

This folder contains a complete workflow for training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using TensorFlow/Keras.

## 📌 Overview

The notebook demonstrates how to:

* Load and preprocess the CIFAR-10 dataset
* Build a CNN architecture
* Compile the model with an appropriate optimizer and loss function
* Train with different batch sizes and validation splits
* Evaluate performance using accuracy/loss curves
* Visualize training and validation metrics

## 📂 Files

* **07_04_Image_Classification_on_CIFAR10.ipynb** — Main notebook containing the full code, training process, and graphs.

## 🧠 Key Concepts Used

* Convolutional Neural Networks (Conv2D, MaxPooling)
* Activation functions (ReLU, Softmax)
* Optimizers (Adam)
* Data normalization
* Train/validation split
* Batch size experimentation

## 📊 Results & Observations

* Baseline CNN model shows smooth training curves.
* Accuracy improves rapidly in initial epochs.
* Validation accuracy follows training accuracy without heavy overfitting.
* Good stability due to appropriate batch size and architecture.

## ▶️ How to Run

1. Install dependencies:

   ```bash
   pip install tensorflow numpy matplotlib
   ```
2. Open the notebook in Jupyter or VS Code.
3. Run all cells in order.

## ✔️ Recommended Usage

* Use this as your reference template for basic image classification tasks.
* Modify the CNN architecture to experiment with depth and regularization.
* Use data augmentation for better generalization.

## 📝 Notes

* Training time varies depending on GPU/CPU.
* Try experimenting with batch sizes: **32, 64, 96**.
* Add dropout layers if model starts overfitting.

---

**Author:** Rehan Shaikh
Happy Experimenting 🚀
