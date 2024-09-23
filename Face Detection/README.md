# ACTIVE PROJECT - EXPECT CHANGES
All credit to https://www.youtube.com/watch?v=N_W4EYtsa10 
# Project Overview

This project focuses on building a facial detection and tracking model using deep learning. The main objective is to identify faces in images and predict the bounding box around each face. The model uses a Convolutional Neural Network (CNN), specifically a pre-trained VGG16 architecture, to perform the facial classification and localization tasks.

## Key Features:

	1.	Data Collection: Images are collected using a connected camera and saved for further processing.
	2.	Data Augmentation: Using the Albumentations library, various transformations like flipping, brightness adjustments, and cropping are applied to enhance the dataset.
	3.	Bounding Box Prediction: The model predicts the bounding box coordinates around faces, allowing accurate localization.
	4.	Classification: Along with localization, the model classifies whether a face is present in an image.
	5.	Deep Learning: A custom model built on TensorFlow utilizing the VGG16 backbone for feature extraction.
	6.	Training Pipeline: A comprehensive data loading, augmentation, and model training pipeline is implemented to train the model on large datasets.

## Code Breakdown

### 1. Image Collection and Storage

The code collects images from a video capture device (like a MacBook camera or an iPhone) and stores them as .jpeg files in a designated folder. A loop runs to collect 30 images, and each image is displayed for a brief period.

### 2. Data Augmentation

The Albumentations library is used to apply several transformations to the images:

	•	Random cropping
	•	Horizontal and vertical flipping
	•	Adjusting brightness, contrast, and gamma
	•	Shifting the RGB color channels

This increases the dataset size and variety, helping the model generalize better during training.

### 3. Image Loading and Preprocessing

The images are loaded using TensorFlow’s tf.data.Dataset API. Each image is resized to 120x120 pixels and normalized by dividing pixel values by 255. Labels for bounding boxes and class (face or no face) are also loaded from JSON files, which contain the bounding box coordinates and class labels.

### 4. Model Architecture

A deep learning model is built using TensorFlow’s Keras Functional API:

	•	VGG16 Backbone: The pre-trained VGG16 model (without its top classification layers) is used as the feature extractor.
	•	Bounding Box Prediction: The model outputs a set of 4 coordinates, representing the bounding box around the detected face.
	•	Classification Output: The model also outputs a binary classification (face/no face) for each image.

### 5. Loss Functions

Two loss functions are used during training:

	•	Classification Loss: Binary cross-entropy to handle the face/no face classification task.
	•	Localization Loss: A custom loss function to calculate the difference between the predicted and true bounding box coordinates.

### 6. Training Process

The model is compiled with the Adam optimizer and a learning rate schedule that gradually reduces the learning rate after each epoch to ensure stable convergence. The model is trained over several epochs, with TensorFlow’s Model.fit function, using training and validation datasets.

### 7. Results Visualization

After each epoch, the model’s performance is evaluated by comparing the predicted bounding boxes with ground truth values. Results are visualized by drawing the predicted bounding boxes on sample images.

## Virtual Environment Setup for Mac

To run this project on a Mac, it’s highly recommended to create a virtual environment to isolate dependencies and ensure smooth project execution. Follow the steps below:

### 1. Create a Virtual Environment:

Open your terminal and navigate to your project folder. Run the following command to create a virtual environment named venv:

```
python3 -m venv venv
```
### 2. Activate the Virtual Environment:

To activate the virtual environment, run:

```
source venv/bin/activate
```
Once activated, your terminal prompt will change, indicating that you’re inside the virtual environment.

### 3. Install Dependencies:

With the environment activated, install the required Python packages (like TensorFlow, OpenCV, Albumentations, etc.) by running:
```
pip install -r requirements.txt
```
If you don’t have a requirements.txt file, you can manually install the libraries:
```
pip install tensorflow opencv-python albumentations matplotlib
```
## Final Notes

This project demonstrates a robust approach to facial detection using deep learning techniques, combined with image augmentation and bounding box regression. It uses TensorFlow for model building and training, making it scalable and suitable for GPU-accelerated tasks. This project can serve as a foundation for more advanced computer vision tasks such as facial recognition or tracking in video streams.
