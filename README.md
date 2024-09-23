Welcome to my GitHub repository of passion projects! This directory showcases a variety of personal projects that I've created to explore new technologies, improve my skills, and have fun with coding. Each project is unique and represents a different area of interest or a new challenge I've taken on. Below, you'll find a brief overview of each project along with links to their respective folders.

# Projects

- Credit Risk Model
- Homemade Neural Network
- Plant Monitoring system
- Quasar Detector
- Pulsar Detector
- Face Detection (ML)

# Project Overviews

## Credit risk classifier
A credit risk model is a financial tool used to estimate the likelihood of a borrower defaulting on a loan or credit obligation. These models are essential for financial institutions as they help in assessing the risk associated with lending and making informed decisions about credit issuance. Here’s a short overview of a typical credit risk model:

Purpose

The primary purpose of a credit risk model is to predict the probability of default (PD) and to estimate the potential loss (Loss Given Default, LGD) in the event of a default. This allows lenders to manage their risk exposure and set appropriate interest rates and credit limits.

## Simple Neural Network Framework

This repository contains an implementation of a neural network framework in Python. It includes classes for creating neural network layers and the neural network itself, functions for training the network using Stochastic Gradient Descent (SGD), and methods for evaluating and visualizing performance. It can serve as guide for how neural networks function and their parts.

### Features

- **Layer Class**: Represents a single neural network layer with forward and backward propagation methods.
- **NeuralNetwork Class**: Represents a neural network composed of multiple layers.
- **Validation Curve**: Function to plot training and testing accuracy over epochs.
- **Layer Testing**: Method to test different layer configurations and save performance data to a CSV file.

### Installation

To run this script, you need to have Python and the following libraries installed:

- NumPy
- Matplotlib
- Pandas

You can install the required libraries using pip:

```bash
pip install numpy matplotlib pandas
```
## Plant Monitoring System with OLED Display

This project implements a plant monitoring system using an ESP32 microcontroller and an SSD1306 OLED display. The system continuously monitors three key parameters crucial for plant health: light intensity, temperature, and soil moisture. Based on predefined thresholds, it provides real-time feedback on the environmental conditions and alerts the user if any parameter goes beyond the desired range.

### Components Used

	•	ESP32 microcontroller
	•	SSD1306 OLED display (128x64 pixels)
	•	Light Dependent Resistor (LDR) for light intensity
	•	Temperature sensor for measuring ambient temperature
	•	Soil moisture sensor for monitoring soil moisture levels

### Functionality

	•	Real-time Monitoring: Continuously reads and displays light intensity, temperature, and soil moisture levels on the OLED display.
	•	Threshold Alerts: If any parameter (light intensity, temperature, or soil moisture) exceeds predefined thresholds, the system displays a warning message indicating the nature of the issue (e.g., too much light, low temperature).
	•	User Feedback: Displays actionable messages based on sensor readings, suggesting whether the plant needs more or less light, water, or if the temperature needs adjustment.
	•	Power Efficiency: Utilizes deep sleep mode of the ESP32 to conserve power between display updates, maximizing battery life.

### Libraries Used

	•	Wire.h for I2C communication
	•	Adafruit_GFX.h and Adafruit_SSD1306.h for OLED display control
	•	esp_sleep.h for ESP32 deep sleep mode functionality

# How to Use

Each project folder contains a `README.md` file with detailed information on the project's purpose, technologies used, setup instructions, and usage guidelines. Feel free to explore the code, provide feedback, and contribute if you're interested!


# Contact

You can reach me at [liam.stothart@gmail.com] I'm always excited to connect with fellow developers and discuss new ideas.

Thank you for visiting my repository. Happy coding!

---

This directory serves as a testament to my passion for coding and continuous learning. I hope you find these projects as enjoyable to explore as they were for me to create.
