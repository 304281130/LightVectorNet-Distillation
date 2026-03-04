<h1 align="center"><font color="red">LightVectorNet: Knowledge Distillation for Vehicle Trajectory Prediction</font></h1>

This repository contains the official implementation for the project: **"Design and Mechanism Analysis of Lightweight Vehicle Trajectory Prediction Model Based on Knowledge Distillation"**.

## 📖 Introduction (简介)
Trajectory prediction is crucial for autonomous driving. However, state-of-the-art Graph Neural Networks (GNNs) are often too computationally expensive for real-time edge deployment (e.g., Jetson Nano). 

Based on the baseline implementation [yet-another-vectornet](https://github.com/xk-huang/yet-another-vectornet), we propose **LightVectorNet**, a lightweight prediction model trained via **Knowledge Distillation**. By using a complex HGNN as the Teacher and a lightweight MLP as the Student, and applying a hybrid loss function ($\alpha=0.5$), we achieved higher accuracy with significantly fewer parameters.

## 📊 Performance (性能对比)
Evaluated on the Argoverse Validation Set (100% data):

| Model (模型) | minADE (m) | minFDE (m) |
| :--- | :--- | :--- |
| Teacher (HGNN) | 16.79 | 32.04 |
| **Student (LightVectorNet)** | **15.60** | **29.99** |

## 🛠️ Data & Pre-trained Models Preparation (数据与权重准备)
To fully reproduce this project, you need both the original dataset and the pre-trained weights. We highly appreciate the original authors for providing these fundamental resources.

### 1. Dataset (数据集)
Please download the pre-processed Argoverse dataset provided by the original author and extract it into the `./interm_data/` directory.
* **Dataset Download Link**: [Google Drive](https://drive.google.com/drive/folders/1w7P9dK0lUoK7B0x7-jXo6gS5qgI2H_hH?usp=sharing)

*(Note: If you encounter a PyG `RuntimeError` regarding version mismatch during evaluation, simply delete the `processed` folder inside the test/val/train directory to force a PyG cache rebuild.)*

### 2. Teacher Model Weights (名师模型权重 - HGNN)
To run the distillation process or visualize the Teacher's predictions, download the pre-trained Teacher model provided by the original author and place it in the `./pretrained_teacher/` directory.
* **Teacher Weights Download**: [Google Drive](https://drive.google.com/drive/folders/1zH1tZJ-D3-q-cM0fJ4I_G91E_l0Eey1A?usp=sharing)

### 3. Student Model Weights (高徒模型权重 - LightVectorNet)
The lightweight student model trained via our hybrid knowledge distillation strategy achieved a minADE of 15.60m. 
* **Student Weights Download**: Available in the **Releases** section of this GitHub repository. Please download `student_light_vectornet.pth` and place it in the root directory.

## 🚀 How to Run (如何运行)

### 1. Training the Student Model (训练学生模型)
```bash
python train_student.py

### 2. Evaluation (定量评估)
Evaluate the models on the Validation/Test Set to get ADE and FDE metrics:
```bash
python evaluate.py
```

### 3. Visualization (定性可视化)
Visualize and compare the trajectories of the Teacher, Student, and Ground Truth:
```bash
python visualize.py
```

## 🙏 Acknowledgements (致谢)
The baseline backbone, data processing pipeline, and original pre-trained model of this project are built upon the excellent work of [xkhuang/yet-another-vectornet](https://github.com/xk-huang/yet-another-vectornet). We extend their architecture to a lightweight distillation framework for efficient edge deployment.
