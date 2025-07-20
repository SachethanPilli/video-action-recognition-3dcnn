# Human Action Recognition Using Deep Learning (I3D & 3D-CNN + LSTM)

This project explores deep learning architectures for **video-based human action recognition** using the UCF-11 dataset. Two approaches were implemented and compared:  
- **I3D (Inflated 3D ConvNet)** with LSTM  
- **3D CNN + LSTM** (custom architecture)  

---

## 📁 Dataset
- **UCF11 (Human Action Dataset)**
- 11 action categories including basketball shooting, biking, diving, etc.
- Each video is preprocessed to extract 16 frames resized to 224×224 resolution.

---

## Architectures

### 1. I3D (Inflated 3D ConvNet) + LSTM
- Based on pretrained **I3D ResNet-50** from `pytorchvideo`
- Inflates 2D kernels into 3D for spatiotemporal feature extraction
- Appended with:
  - LSTM for sequence modeling
  - Dense layers for classification
- Fine-tuned on UCF-11 dataset  
-  Pretrained on **Kinetics-400**

### 2. Custom 3D CNN + LSTM
- Three 3D convolutional layers with batch normalization & pooling
- TimeDistributed Flatten layer
- LSTM layer to capture motion patterns across time
- Dense + Dropout + Softmax for final classification

---

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Training/Validation Loss & Accuracy plots

---

## Libraries Used
- Python, PyTorch, TensorFlow
- OpenCV, NumPy, Matplotlib
- Scikit-learn
- PyTorchVideo

---

##  Results
- Both models achieved high performance on the UCF-11 dataset.
- **I3D + LSTM** showed stronger generalization due to pretrained weights.
- **3D CNN + LSTM** performed competitively with fewer parameters and more flexibility.
<img width="685" height="580" alt="image" src="https://github.com/user-attachments/assets/83b0b3e1-c1e6-47ab-a2a1-f7aa18c455c1" />
<img width="800" height="723" alt="image" src="https://github.com/user-attachments/assets/46266830-cf38-4906-88f6-dd4a0bba3e4d" />

I3D model Confusion Matrix

<img width="685" height="540" alt="image" src="https://github.com/user-attachments/assets/3473a146-4914-4e82-ad03-66511ec838b0" />
<img width="800" height="723" alt="image" src="https://github.com/user-attachments/assets/d415fb6f-b73f-42a1-a467-ff54f1bfaeb5" />

Custom 3D CNN model Confusion Matrix 
 

---

##  Highlights
- Action recognition across video sequences
- Comparison between pretrained vs custom feature extraction
- Temporal modeling using LSTM in both pipelines
- Fine-tuning and evaluation on real-world dataset

---

## 📂 Files in This Repository
- `3DCNN + LSTM.ipynb` – Custom-built architecture
- `I3D .ipynb` – Pretrained I3D + LSTM model pipeline
- `README.md` – This documentation

---

- Expand dataset and generalize to real-world footage
- Deploy as a real-time action recognition app

---
## Acknowledgments
[PyTorchVideo](https://pytorchvideo.org) for providing the I3D model.
Dataset credits: [UCF11 – Action Recognition Data Set](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php) 
