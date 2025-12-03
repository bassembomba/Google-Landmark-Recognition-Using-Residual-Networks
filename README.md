# 🏛️ Google Landmark Recognition Using Residual Networks (ResNet)

<p align="center">
  <img src="https://img.shields.io/badge/Thesis-Bachelor-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-Computer%20Vision-red?style=for-the-badge"/>
</p>

---

## 🚀 Tech Stack & Frameworks

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

---

## 🎓 Project Overview

This repository contains the full implementation of my **Bachelor Thesis Project**, which focuses on:

> ✅ **Recognizing world landmarks from images using a deep Residual Neural Network (ResNet).**

The model is trained on the **Google Landmarks Recognition V2 Dataset** and performs **multi-class image classification** across **300 different landmarks** using **transfer learning with ResNet50**.

The project includes:

- Full data preprocessing pipeline  
- Class filtering and balancing  
- Data augmentation  
- Model training & validation  
- Performance visualization  
- Confusion matrices & classification report  
- Model prediction visualization  

---

## 📂 Dataset

- **Dataset Used:** Google Landmarks Recognition 2021 (V2)
- **Source:** Kaggle
- **Total Classes:** Thousands
- **Used in This Project:**  
  - ✅ Top **300 landmark classes**
  - ✅ Only classes with **200–400 images each** for balance

### 🧹 Data Processing Steps:
- CSV parsing & image path reconstruction
- Class distribution analysis
- Histogram visualization
- Random sample visualization
<img width="400" alt="histogram" src="https://github.com/user-attachments/assets/202ee05c-4756-479a-95d9-e488ac818d2c" />
<img width="423"  alt="matrix4" src="https://github.com/user-attachments/assets/1d676cb9-8c89-4ca5-9f41-13cffac9157c" />


---

## 🧠 Model Architecture

This project uses **Transfer Learning with ResNet50**:

- ✅ Pretrained on **ImageNet**
- ✅ Feature extraction using:
  - `include_top = False`
  - `pooling = "avg"`

### Custom Classification Head:
- Flatten Layer  
- Dense (4096 units, ReLU)  
- Dropout (0.2)  
- Dense (4096 units, ReLU)  
- Dropout (0.2)  
- Final Dense Layer (Softmax, 300 classes)
<img width="562" alt="model architecture" src="https://github.com/user-attachments/assets/563ed41e-5544-47e7-9091-24c910512acb" />

---

## 🧪 Training Setup

| Parameter        | Value |
|------------------|--------|
| Image Size       | 128 × 128 |
| Batch Size       | 32 |
| Epochs           | 30 |
| Classes          | 300 |
| Train Split      | 70% |
| Validation Split| 20% |
| Optimizer        | Adagrad |
| Loss Function   | Categorical Crossentropy |

---

## 🔀 Data Augmentation

To improve generalization, advanced data augmentation techniques were applied:

- Horizontal Flip  
- Vertical Flip  
- Rotation  
- Zoom  
- Width & Height Shift  
- Shear Transformation  
- Random Cropping  

This significantly improves model robustness against lighting and viewpoint changes.

---

## 📊 Model Evaluation

The project includes:

✅ Accuracy & Loss Curves  
✅ Confusion Matrices (Split into batches of 50 classes)  
✅ Per-class Precision, Recall & F1-Score  
✅ Full Classification Report Heatmap  
✅ Visual inspection of:
- Perfect predictions
- Low-confidence predictions
<img width="400" alt="finalexperiment acc" src="https://github.com/user-attachments/assets/e09b747e-fbf5-4071-991c-ac0a86612e7b" />
<img width="400" alt="finalexperiment loss" src="https://github.com/user-attachments/assets/beb83df3-3297-48ac-9ab8-4f14d3f7e07b" />

---

## 🖼️ Prediction Visualization

The model visualizes:

- ✅ Correct predictions with confidence score
- ✅ Poor predictions (confidence < 10%)
- ✅ Side-by-side comparison of similar landmark images

<img width="902" alt="correct prediction" src="https://github.com/user-attachments/assets/ccfc87c9-3d0e-48e1-9d44-c68caa24242d" />

This allows deep inspection of model behavior.

---
 
## 🏆 Key Achievements
✅ Successfully trained a 300-class landmark classifier

✅ Built a full computer vision pipeline from scratch

✅ Applied transfer learning, augmentation, and deep CNN tuning

✅ Complete evaluation framework with visualization

✅ Real-world scale dataset handling

---

## 🔮 Future Improvements
Use ResNet101 / EfficientNet

Add Learning Rate Scheduling

Implement Mixed Precision Training

Deploy as a Web API / Mobile App

Integrate Grad-CAM Visual Explanations
