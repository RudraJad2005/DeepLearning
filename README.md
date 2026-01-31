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

### 6. Images & Convolutional Neural Networks (`Images & Convolutional Neural Networks.ipynb`)
Image classification using CNNs, `ImageFolder` datasets, data augmentation (random flips, rotations), convolutional layers with max-pooling, multi-class metrics (Precision, Recall), per-class performance evaluation. Includes sections on:
- **Importing Libraries and Data Visualization**: Setting up PyTorch vision tools and visualizing cloud images
- **Defining CNNs**: Building feature extractors with Conv2d, ELU activations, and max-pooling
- **Data Augmentation**: Using transforms to improve model generalization
- **Training Loop**: End-to-end training with CrossEntropyLoss and Adam optimizer
- **Model Evaluation**: Computing macro-averaged precision and recall metrics
- **Per-Class Precision**: Analyzing performance across 7 cloud types

## 🎯 Learning Objectives

Tensors & operations • Neural network architectures • Activation functions • Training loops • Loss functions & optimizers • DataLoaders • Transfer learning • Weight initialization • Custom datasets • Model evaluation • Convolutional Neural Networks • Image classification • Data augmentation • Multi-class metrics

## 🛠️ Technologies Used

- **Pandas** - Data manipulation and CSV loading
- **Python 3.13.9**
- **PyTorch** - Deep learning framework
- **torchvision** - Image processing and computer vision utilities
- **torchmetrics** - Model evaluation metrics
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Jupyter Notebooks** - Interactive development environment

## 📖 How to Use

Start with Introduction notebooks (1-4), then move to Intermediate (5-6). Run cells sequentially and experiment with the code. Each notebook now includes detailed markdown notes explaining the purpose and concepts behind each section.

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
pip install torch torchvision numpy pandas torchmetrics matplotlib ipykernel jupyter
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
│   ├── Images & Convolutional Neural Networks.ipynb # Chapter 6: CNNs & Image Classification
│   ├── water_potability/
│   │   ├── water_train.csv                         # Training dataset
│   │   └── water_test.csv                          # Test dataset
│   └── clouds/
│       ├── clouds_train/                           # Training images (7 cloud types)
│       └── clouds_test/                            # Test images (7 cloud types)
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

*Last Updated: December 24, 2025*
