# Pneumonia Detection from Chest X-Rays using a Custom CNN

This repository contains the code and resources for a deep learning project focused on detecting pneumonia from pediatric chest X-ray images. The project utilizes a custom-built Convolutional Neural Network (CNN) implemented in TensorFlow and Keras.

![Sample X-Rays](https://github.com/Harish-lvrk/Pneumonia-Detection/blob/main/A_Custom_CNN_Architecture/Difference-in-Chest-X-Ray-Images-in-Normal-and-Pneumonia.png)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Methodology](#-methodology)
    - [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
    - [Model Architecture](#model-architecture)
4. [Results](#-results)
5. [How to Use](#-how-to-use)
6. [Acknowledgments](#-acknowledgments)

---

## 📝 Project Overview

This project implements a binary image classifier to distinguish between 'Normal' and 'Pneumonia' chest X-rays. The core of the project is a custom 5-layer CNN built from scratch. The model addresses common challenges in medical imaging, such as class imbalance, through data augmentation and employs advanced training techniques like an adaptive learning rate to achieve high performance.

The final model achieves **90.4% accuracy** on the unseen test set, demonstrating strong potential as a diagnostic aid.

---

## 🗂️ Dataset

The model was trained on the **Chest X-Ray Images (Pneumonia)** dataset, which contains 5,863 pediatric chest X-ray images. The data is organized into `train`, `test`, and `val` sets, with each set containing subfolders for 'PNEUMONIA' and 'NORMAL' classes.

- **Source:** [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Class Imbalance:** The training set has a significantly higher number of 'Pneumonia' images than 'Normal' images, a key challenge addressed in this project.

---

## 🛠️ Methodology

### Data Preprocessing & Augmentation

To prepare the data for the CNN, the following steps were taken:
1.  **Image Resizing:** All images were resized to a uniform `150x150` pixels.
2.  **Grayscale Conversion:** Images were converted to a single channel (grayscale).
3.  **Normalization:** Pixel values were scaled from `[0, 255]` to `[0, 1]`.
4.  **Data Augmentation:** To combat class imbalance and prevent overfitting, Keras's `ImageDataGenerator` was used to apply random transformations to the training images, including:
    -   Rotation (`30` degrees)
    -   Zoom (`0.2` range)
    -   Width and Height Shift (`0.1` range)
    -   Horizontal Flip

### Model Architecture

A custom CNN was designed with a focus on progressive feature extraction and regularization.

**Key Layers:**
-   **Five Convolutional Blocks:** Each block contains `Conv2D`, `BatchNormalization`, and `MaxPooling2D` layers. The number of filters increases with depth (32 → 64 → 128 → 256).
-   **Dropout Layers:** Applied strategically after convolutional blocks to reduce overfitting.
-   **Classifier Head:** A `Flatten` layer followed by `Dense` layers, with a final `sigmoid` activation function for binary classification.

The model was trained for **12 epochs** using the **RMSprop optimizer** and a `ReduceLROnPlateau` callback to adapt the learning rate.

---

## 📊 Results

The model's performance was evaluated on the unseen test set, achieving an overall accuracy of **90.4%**.

| Class     | Precision | Recall | F1-Score |
| :-------- | :-------: | :----: | :------: |
| Pneumonia |   0.93    |  0.92  |   0.92   |
| Normal    |   0.87    |  0.88  |   0.87   |

**Confusion Matrix:**
-   **True Positives (Pneumonia):** 357
-   **True Negatives (Normal):** 206
-   **False Positives (Type I Error):** 28
-   **False Negatives (Type II Error):** 33

The high recall for the 'Pneumonia' class (92%) is particularly important, as it indicates the model is very effective at correctly identifying patients with the condition, minimizing the risk of missed diagnoses.

---

## 🚀 How to Use

To run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone git clone https://github.com/Harish-lvrk/Pneumonia-Detection.git
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *Note: A `requirements.txt` file should be created containing libraries like `tensorflow`, `numpy`, `pandas`, `matplotlib`, and `opencv-python`.*

3.  **Download the dataset:**
    Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place the `chest_xray` folder in the project's root directory or update the paths in the notebook.

4.  **Run the Jupyter Notebook:**
    Launch Jupyter and open the `own-architecture-pneumonia-detection-v2.ipynb` notebook.
    ```bash
    jupyter notebook
    ```

---

## 🙏 Acknowledgments

-   The dataset was provided by Kermany et al. and is available on Kaggle.
-   This project was developed as part of my summer internship at RGUKT Nuzvid.
-   Special thanks to the open-source community for providing the tools and frameworks (TensorFlow, Keras, etc.) that made this research possible.
