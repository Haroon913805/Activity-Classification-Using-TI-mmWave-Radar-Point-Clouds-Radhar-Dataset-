# Activity-Classification-Using-TI-mmWave-Radar-Point-Clouds-Radhar-Dataset-
This project focuses on classifying human activities (Boxing, Jump, Squat, Jack, Walk) using raw point-cloud data from TI mmWave Radar. The methodology involves several stages, including signal processing, feature extraction, transformation into spectrograms, preprocessing, and classification using a CNN.

Workflow
Data Source

Utilizes raw .txt point-cloud files obtained from TI mmWave Radar.
Signal Processing

Extracts features from point-clouds, converting them into 1D Doppler-time curves.
Transformation

Applies Short-Time Fourier Transform (STFT) to generate 2D Micro-Doppler Spectrograms.
Preprocessing

Implements noise filtering and normalization techniques to enhance rhythmic motion patterns.
Architecture

A PyTorch-based Convolutional Neural Network (CNN) is designed to classify the generated spectrograms.
Training

The model is trained over 50 epochs using the Adam optimizer, with a 60/40 data split to maximize testing support.
Performance Metrics

Evaluates model performance through Precision, Recall, and F1-Score for each activity class.
Final Output

Generates a heatmap confusion matrix to visually assess prediction accuracy and identify misclassified movements.
Installation
To run this project, ensure you have the following dependencies installed:

PyTorch
NumPy
Matplotlib
SciPy
Usage
Clone the repository.

Prepare your data in the specified format.
Run the training script to start the model training process.
Evaluate the model using the provided metrics and visualize results with the confusion matrix.
