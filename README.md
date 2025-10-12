# MalariaNet
Overview

This repository presents the implementation of a Convolutional Neural Network (CNN)–based system for Plasmodium species classification using microscopic blood smear images. The project integrates a digital microscope interface, preprocessing pipeline, model training, validation, and a graphical user interface (GUI) to support semi-automated malaria diagnosis in laboratory and field settings.

The system classifies microscopic images into three categories:

-Plasmodium falciparum

-Plasmodium vivax

-Uninfected red blood cells

By combining image enhancement, deep learning, and a Python-based GUI, this system enables real-time classification directly from microscope image feeds.

System Architecture
1. Image Preprocessing

Implemented in detection_program_v2.py, the preprocessing module transforms RGB microscope images into seven-channel enhanced representations, consisting of:

-Original RGB channels

-Contrast, saturation, and sharpness enhancement

-Canny Edge Detection channel for morphological contour highlighting

These enhanced features significantly improve the CNN’s ability to distinguish between visually similar parasite species.

2. CNN Model (MalariaNet)

The CNN model architecture (MalariaNet) consists of:

-4 convolutional blocks: Conv–BatchNorm–LeakyReLU–MaxPool

-Fully connected layers with dropout (0.5)

-Softmax classifier for 3 output classes

Model training was performed using 5-fold cross-validation, achieving an average accuracy of ~99% with precision, recall, and F1-score above 0.98.
Each fold’s trained model is stored as a .pth file within the Model folder. Among these, Fold 4 demonstrated the best performance balance (accuracy = 99.17%, AUC = 1.00, stable training/validation loss convergence). Therefore, Fold 4 serves as the default model for inference and GUI-based real-time classification.

3. Graphical User Interface (GUI)

The GUI, developed in Python Tkinter and implemented within detection_program_v2.py, provides real-time interaction between the user and the CNN model.
Key functionalities include:
-Live image capture from the microscope camera

-On-screen classification results with confidence scores

-Manual annotation via bounding boxes

-Image enhancement controls (zoom, rotate, fit-to-window, pan)

-Result export in JSON format for dataset generation and validation

Modules such as image_viewer_enhancement.py add extended interaction features, while smart_batch_loader.py and model_val.py handle dataset evaluation and batch comparison tasks.

Dependencies

All required Python packages are listed in requirements.txt

Usage Instructions

1. Set up environment:

        pip install -r requirements.txt

2. Run the classification system:

        python detection_program_v2.py

Then load the Fold 4 model from the /Model directory and initialize the GUI interface.

Optional:

To perform validation or batch processing using saved annotations, execute:

    python model_val.py

Evaluation and Results

-5-Fold Cross Validation: Average accuracy = 98.89%; F1-score = 0.99

-Fold 4: Accuracy = 99.17%; AUC = 1.00; minimal validation loss (~0.018)

-Independent Dataset (25 images): Accuracy >95% on all classes, 100% for uninfected class

These results demonstrate that the model is both accurate and generalizable for real-world microscopy data.

Citation

William Setyawan Parulian (2025). Digitization of Conventional Microscopes for Plasmodium Classification Using Convolutional Neural Network (CNN) and Cameras. Undergraduate Thesis, Institut Teknologi Sumatera.
