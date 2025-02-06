# UNET Oxford Pets - Semantic Segmentation

This repository implements an **enhanced UNET model** for **semantic segmentation** on the **Oxford-IIIT Pet Dataset**. The model builds on a baseline UNET by adding **additional encoder-decoder blocks** and **increasing base channels**, leading to improved feature extraction and a **per-pixel accuracy of at least 88%**.

## 📌 Model Enhancements
- Increased the number of **encoder and decoder blocks** to capture deeper spatial features.
- Expanded the **base number of channels** to improve representation learning.
- Applied **optimized training settings** with tuned hyperparameters.

## 🗂 Dataset
The Oxford-IIIT Pet Dataset consists of **37 categories** of pet images, with segmentation masks having three classes:
- **Pet body (foreground)**
- **Pet outline (boundary)**
- **Background**

### 📌 Data Preprocessing
Images are resized and converted to tensors using:
```python
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

🚀 Training Setup

Hyperparameters:
Optimizer: Adam
Loss Function: CrossEntropyLoss
Batch Size: 16
Learning Rate: 1e-3
Epochs: Optimized for 88% accuracy

🎯 Predictions

200 random samples are selected for validation:
