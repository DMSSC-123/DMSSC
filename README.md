# Dual Self-Expression Subspace Clustering with Multi-Scale Features and Structural Consistency Alignment

This repository provides the implementation of a deep clustering framework based on multi-scale visual features, graph-based structural learning, and dual self-expression with consistency alignment.

---

## 📌 Features

- Multi-scale feature extraction using EfficientNet
- Structural representation learning using Chebyshev Graph Convolution
- Dual self-expression (content + structure)
- Consistency alignment for robust affinity learning
- Spectral clustering for final label assignment

---

## 📁 Usage

### **1. Install dependencies**

```bash
git clone https://github.com/DMSSC-123/DeepSubspaceClustering.git
cd DMSSC
pip install -r requirements.txt

```
### **2. Feature extraction**
```bash
python Feature_extract.py --dataset=cifar100
```
### **3. Train the networks**
```bash
python main.py --dataset=cifar100


