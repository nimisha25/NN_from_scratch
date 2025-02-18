# Neural Network From Scratch (NumPy vs. PyTorch)

## 📌 Overview
This project implements a **Neural Network from scratch** using only **NumPy** to classify handwritten digits. The model is compared against a **PyTorch-based implementation** to understand the trade-offs between custom and optimized deep learning frameworks.

---

## 📊 Final Results & Model Comparison

| **Model**        | **Hidden Layers** | **Validation Loss** | **Validation Error** | **Validation Accuracy** |
|-----------------|------------------|--------------------|---------------------|----------------------|
| **NumPy Model (Before)** | **1 Layer (784 → 10)** | **0.335** | **9.62%** | **90.38%** |
| **NumPy Model (Final)** | **2 Layers (784 → 128 → 10)** | **0.113** ✅ | **3.42% ✅** | **96.58% ✅** |
| **PyTorch Model** | **2 Layers (Optimized)** | **0.0779** | **2.41%** | **97.59%** |

**Adding an extra hidden layer (128 neurons) improved validation accuracy from ~90% to ~96.6%!**  

---

## 📊 Training & Performance Visualization

### **🔹 1-Layer Model (784 → 10)**
- The validation curve stabilizes early but at a higher loss.

<div style="text-align: center;">
    <img src="Figure_1.png" alt="1-Layer Model Training" width="400">
</div>

---

### **🔹 2-Layer Model (784 → 128 → 10)**
- The extra hidden layer helps the network learn better feature representations.
- **Early stopping was triggered at iteration 17,500**, preventing overfitting.
<div style="text-align: center;">
    <img src="Figure_2.png" alt="1-Layer Model Training" width="400">
</div>

