# 🧠 ResNet50 from Scratch — Deep Learning Study Project

---

## 📌 Project Overview

This project implements **ResNet50 from scratch using TensorFlow/Keras**, with a strong focus on:

- Deep theoretical understanding
- Manual implementation of residual blocks
- Proper Batch Normalization behavior
- Training on a custom 6-class image dataset (64×64 resolution)
- Transfer learning using a pretrained ResNet model

This repository is designed as:

> 📚 A study-focused, in-depth reference notebook  
> 🧪 A reproducible deep learning experiment  
> 🏗 A structural understanding of modern CNN architectures

---

## 🎯 Objectives

- Understand the **degradation problem** in deep networks
- Implement **Residual Learning**
- Build:
  - Identity Block
  - Convolutional Block
  - Full ResNet50 architecture
- Handle **BatchNorm training vs inference correctly**
- Train and evaluate the model
- Compare with a pretrained version

---

# 🏗 Architecture Overview

## Residual Learning Principle

Instead of learning:

$$
H(x)
$$

ResNet learns:

$$
F(x) = H(x) - x
$$

So:

$$
H(x) = F(x) + x
$$

This enables:

- Stable gradient flow
- Easier optimization
- Training of very deep networks

---

## 🔁 Identity Block

Used when input and output dimensions are the same.

Structure:

```text
input
│
Conv → BN → ReLU
│
Conv → BN → ReLU
│
Conv → BN
│
Add Shortcut
│
ReLU
```

Key characteristics:

- No dimensional change
- Pure identity shortcut
- Enables deep stacking

---

## 🔄 Convolutional Block

Used when dimensions change (downsampling).

Differences:

- Stride `s > 1`
- Shortcut includes Conv layer

$$
Y = \text{ReLU}(F(X) + W_s X)
$$

Used at the beginning of each new stage.

---

## 🏢 ResNet50 Structure (Simplified 64×64)

| Stage   | Output Channels |
| ------- | --------------- |
| Conv1   | 64              |
| Stage 2 | 256             |
| Stage 3 | 512             |
| Stage 4 | 1024            |
| Stage 5 | 2048            |

Followed by:

- Average Pooling
- Flatten
- Dense Softmax (6 classes)

Total ≈ 50 layers.

---

# 📂 Dataset

- Training examples: **1080**
- Test examples: **120**
- Image shape: **64 × 64 × 3**
- Number of classes: **6**

Preprocessing:

- Pixel normalization: `X / 255`
- One-hot encoding of labels

---

# 🧪 Training Configuration

| Parameter  | Value                    |
| ---------- | ------------------------ |
| Optimizer  | Adam                     |
| Loss       | Categorical Crossentropy |
| Batch Size | 32                       |
| Epochs     | 10                       |

Loss function:

$$
\mathcal{L} = -\sum\_{i=1}^{C} y_i \log(\hat{y}\_i)
$$

---

# 🔬 Batch Normalization Handling

Important distinction:

### Training Mode

- Uses batch statistics
- Updates moving averages

### Inference Mode

- Uses stored moving averages
- No updates

In custom blocks:

```python
BatchNormalization()(X, training=training)
```

This ensures correct behavior during:

- Training
- Evaluation
- Manual testing

---

# 🚀 Transfer Learning

We also experiment with:

- Loading pretrained resnet50.h5
- Fine-tuning on our dataset

Benefits:

- Faster convergence
- Better accuracy with small datasets
- Improved generalization

---

# 📊 Expected Learning Outcomes

By completing this project, you will deeply understand:

- Why ResNet works
- How skip connections stabilize training
- How BatchNorm behaves internally
- How to construct complex architectures manually
- How pretrained networks accelerate performance

---

# 📦 Repository Structure

```text
├── Residual_Network_ResNets.ipynb
├── resnet50.h5
├── resnets_utils.py
├── images/
├── datasets/
├── test_utils.py
├── public_tests.py
└── README.md
```

---

# 🧠 Key Takeaways

- Deep networks fail without residual connections
- Identity shortcuts enable gradient preservation
- Projection shortcuts allow dimensional changes
- BatchNorm must be handled carefully
- Transfer learning is powerful for small datasets
- Implementing architectures manually builds real intuition

---

# 📖 Future Improvements

- Add learning rate scheduling
- Add data augmentation
- Replace Flatten with GlobalAveragePooling
- Implement ResNet50V2 variant
- Add Grad-CAM visualization
- Add confusion matrix evaluation

---

# 👨‍💻 Author Notes

This project was built as:

- A deep learning study reference
- A conceptual clarity exercise
- A practical implementation guide
