# Deep Learning with PyTorch - Study Notes

A comprehensive collection of Jupyter notebooks documenting my journey learning Deep Learning with PyTorch. These notes cover fundamental concepts, neural network architectures, and training techniques.

## 📚 Notebooks Overview

### 1. Introduction to PyTorch (`Intro_to_Pytorch.ipynb`)
**Chapter 1: Deep Learning Fundamentals**

Topics covered:
- **Tensors**: Creating and manipulating tensors, the building blocks of PyTorch
- **Tensor Operations**: Element-wise operations (addition, subtraction, multiplication)
- **Linear Layers**: Understanding `nn.Linear` and linear transformations
- **Sequential Models**: Building multi-layer neural networks
- **Model Parameters**: Counting and understanding learnable weights
- **Key PyTorch Methods**: Essential functions and their usage

Key concepts:
- What tensors are and how to create them
- How to perform operations on tensors
- Building neural networks with `nn.Sequential`
- Calculating the number of parameters in a model

### 2. Neural Network Architecture & Hyperparameters (`NN_Architecture_and_Hyperparameters.ipynb`)
**Chapter 2: Architecture Design & Activation Functions**

Topics covered:
- **Activation Functions**: Sigmoid and Softmax explained with mathematical formulas
- **Network Architecture Design**: Patterns for different task types
- **Regression vs Classification**: How to structure networks for each task
- **Hyperparameters**: Understanding depth, width, and activation choices
- **One-Hot Encoding**: Label encoding techniques
- **Accessing Model Parameters**: Working with weights and biases
- **Gradient Descent**: Manual weight updates using gradients.

Architecture patterns:
- **Binary Classification**: Sigmoid activation with 1 output
- **Multi-Class Classification**: Softmax activation with N outputs
- **Regression**: No activation, continuous output values

Key concepts:
- When to use Sigmoid vs Softmax
- Designing network architectures for specific tasks
- Understanding hyperparameters
- How gradient descent updates work

### 3. Training a Neural Network (`Training a Neural Network.ipynb`)
**Chapter 3: Training & Optimization**

Topics covered:
- **Datasets & DataLoaders**: Creating and loading training data
- **Loss Functions**: Mean Squared Error (MSE) and other loss metrics
- **Training Loop**: Complete training workflow
- **Optimizers**: Gradient descent and parameter updates
- **Backpropagation**: Computing gradients automatically

Training workflow:
1. Prepare data with `TensorDataset` and `DataLoader`
2. Define loss function (criterion)
3. Set up optimizer
4. Training loop: forward pass → compute loss → backward pass → update weights

## 🎯 Learning Objectives

After studying these notebooks, you will understand:

- ✅ How to create and manipulate tensors in PyTorch
- ✅ How to build neural networks using `nn.Sequential` and `nn.Linear`
- ✅ The difference between Sigmoid and Softmax activation functions
- ✅ How to design networks for regression vs classification tasks
- ✅ What hyperparameters are and how they affect model architecture
- ✅ How to prepare datasets and create data loaders
- ✅ The complete training loop workflow
- ✅ How backpropagation and gradient descent work

## 🛠️ Technologies Used

- **Python 3.13.9**
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Jupyter Notebooks** - Interactive development environment

## 📖 How to Use These Notes

1. **Sequential Learning**: Start with `Intro_to_Pytorch.ipynb`, then move to architecture and training notebooks
2. **Hands-On Practice**: Run each code cell to see outputs and experiment
3. **Comprehensive Explanations**: Each code example has detailed markdown explanations
4. **Mathematical Context**: Key formulas are included with LaTeX formatting

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
pip install torch numpy ipykernel jupyter
```

5. Launch Jupyter:
```bash
jupyter notebook
```

## 📝 Key Formulas & Concepts

### Sigmoid Function
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- Output range: [0, 1]
- Use: Binary classification

### Softmax Function
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
- Output range: [0, 1] per class, sum = 1
- Use: Multi-class classification

### Linear Layer Parameters
For `nn.Linear(in_features=m, out_features=n)`:
- Weights: m × n parameters
- Bias: n parameters
- Total: (m × n) + n = n(m + 1) parameters

## 📊 Project Structure

```
DeepLearning/
│
├── Intro_to_Pytorch.ipynb               # Chapter 1: Basics
├── NN_Architecture_and_Hyperparameters.ipynb  # Chapter 2: Architecture
├── Training a Neural Network.ipynb      # Chapter 3: Training
├── .venv/                               # Virtual environment
└── README.md                            # This file
```

## 🎓 Topics by Difficulty

### Beginner
- Creating tensors
- Basic tensor operations
- Building simple models with `nn.Sequential`

### Intermediate
- Activation functions (Sigmoid, Softmax)
- Network architecture design
- Accessing and understanding parameters

### Advanced
- Manual gradient descent implementation
- Training loop construction
- Optimization techniques

## 💡 Best Practices Covered

- ✅ Matching output layer size to task requirements
- ✅ Choosing appropriate activation functions
- ✅ Using DataLoader for efficient batch processing
- ✅ Proper training loop structure (zero_grad → forward → loss → backward → step)
- ✅ Understanding gradient flow and backpropagation

## 🤝 Contributing

These are personal study notes, but suggestions and improvements are welcome! Feel free to:
- Report errors or typos
- Suggest additional examples
- Share alternative explanations

## 📧 Contact

Created by Rudra Jadhav

---

⭐ **Star this repository if you find these notes helpful!**

*Last Updated: December 12, 2025*
