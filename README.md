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
git clone git@github.com:cs-whh/DSASC.git
cd DSASC 
pip install -r requirements.txt


### **2. Feature extraction**


python Feature_extract.py --dataset fashion_mnist

### **3. Train the networks**
python main.py --dataset fashion_mnist


