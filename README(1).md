# Pneumonia Detection from Chest X-Rays using Custom CNN

## Project Overview
This project implements a deep learning-based medical image classification system to detect pneumonia from pediatric chest X-ray images.

A custom Convolutional Neural Network (CNN) was built using TensorFlow and Keras to classify X-ray images into:

- Normal
- Pneumonia

The model helps assist early disease screening using medical imaging.

---

## Dataset
The model was trained using the Chest X-Ray Images (Pneumonia) dataset containing 5,863 chest X-ray images.

Dataset contains:
- Training Set
- Validation Set
- Test Set

---

## Methodology

### Data Preprocessing
- Image resizing to 150×150
- Grayscale conversion
- Normalization
- Data augmentation

### Model Architecture
- Custom CNN
- Multiple convolution layers
- Max pooling
- Dense layers
- Dropout regularization

---

## Results
- Binary classification model
- Achieved approximately 90.4% test accuracy
- Reduced overfitting using augmentation

---

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Jupyter Notebook

---

## Project Structure

```text
dataset/
notebooks/
requirements.txt
README.md
```

---

## Future Improvements
- Transfer Learning
- Explainable AI
- Real-time deployment
- Mobile healthcare integration
