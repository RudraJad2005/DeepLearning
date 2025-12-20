# Deep Learning with PyTorch - Study Notes

A comprehensive collection of Jupyter notebooks documenting my journey learning Deep Learning with PyTorch. These notes cover fundamental concepts, neural network architectures, and training techniques.

## 📚 Notebooks Overview

### 1. Introduction to PyTorch (`Intro_to_Pytorch.ipynb`)
Fundamentals: tensors, tensor operations, `nn.Linear`, `nn.Sequential`, model parameters.

### 2. Neural Network Architecture & Hyperparameters (`NN_Architecture_and_Hyperparameters.ipynb`)
Activation functions (Sigmoid, Softmax, ReLU), architecture design patterns, regression vs classification, gradient descent.

### 3. Training a Neural Network (`Training a Neural Network.ipynb`)
Datasets & DataLoaders, loss functions (MSE), training loop workflow, optimizers, backpropagation.

### 4. Evaluating and Improving Models (`Evaluating_and_Improving_Models.ipynb`)
Layer freezing, transfer learning, weight initialization methods (Xavier, He/Kaiming), optimization strategies.

### 5. Training Robust Neural Networks (`Training Robust Neural Networks.ipynb`)
Custom `Dataset` class, binary classification with BCE loss, CSV data pipeline, `torchmetrics.Accuracy`, proper evaluation with `net.eval()` and `torch.no_grad()`.

## 🎯 Learning Objectives

Tensors & operations • Neural network architectures • Activation functions • Training loops • Loss functions & optimizers • DataLoaders • Transfer learning • Weight initialization • Custom datasets • Model evaluation

## 🛠️ Technologies Used

- **Pandas** - Data manipulation and CSV loading
- **Python 3.13.9**
- **PyTorch** - Deep learning framework
- **torchmetrics** - Model evaluation metrics
- **NumPy** - Numerical computing
- **Jupyter Notebooks** - Interactive development environment

## 📖 How to Use

Start with Introduction notebooks (1-4), then move to Intermediate (5). Run cells sequentially and experiment with the code.

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.13+
pip (Python package installer)
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/DeepLearning.git
cd DeepLearning
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

4. Install required packages:
```bash
pip install torch numpy pandas torchmetrics ipykernel jupyter
```

5. Launch Jupyter:
```bash
jupyter notebook
```



## 📊 Project Structure

```
DeepLearning/
│
├── Introduction/
│   ├── Intro_to_Pytorch.ipynb                      # Chapter 1: Basics
│   ├── NN_Architecture_and_Hyperparameters.ipynb   # Chapter 2: Architecture
│   ├── Training a Neural Network.ipynb             # Chapter 3: Training
│   └── Evaluating_and_Improving_Models.ipynb       # Chapter 4: Optimization
│
├── Intermediate/
│   ├── Training Robust Neural Networks.ipynb       # Chapter 5: Binary Classification
│   └── water_potability/
│       ├── water_train.csv                         # Training dataset
│       └── water_test.csv                          # Test dataset
│
├── .venv/                                          # Virtual environment
└── README.md                                       # This file
```



## 🤝 Contributing

These are personal study notes, but suggestions and improvements are welcome! Feel free to:
- Report errors or typos
- Suggest additional examples
- Share alternative explanations

## 📧 Contact

Created by Rudra Jadhav
---

⭐ **Star this repository if you find these notes helpful!**

*Last Updated: December 20, 2025*
