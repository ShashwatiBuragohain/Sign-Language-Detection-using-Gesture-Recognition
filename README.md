# Sign-Language-Detection-using-Gesture-Recognition
A real-time CNN-based sign language recognition system with webcam-based gesture detection, background subtraction, and skin segmentation.
## Overview
Accurate sign language recognition is fundamental for facilitating accessibility and interaction for people with hearing impairments. 
This project uses **Convolutional Neural Networks (CNNs)** to classify **36 gesture classes (A–Z and 0–9)** from the American Sign Language dataset.  

This pipeline includes:  
- **Preprocessing**: resizing, grayscale conversion, normalization, and one-hot encoding  
- **Data Augmentation**: rotations, shifts, brightness, and zoom to improve generalization  
- **CNN Architecture**: 2 convolutional layers, max-pooling, dropout, and softmax classifier (1.6M params)  
- **Training**: Adam optimizer, categorical cross-entropy, early stopping  
- **Evaluation**: Accuracy, confusion matrix, classification report  
- **Real-Time Recognition**: webcam input with background subtraction + skin segmentation

  *Snippet of the real-time processing code, showing the combination of background subtraction and HSV-based skin segmentation to isolate the hand.*

## Features
- Real-Time Webcam Integration: Processes live video feed for instant gesture recognition.
- Advanced Preprocessing: Employs background subtraction and HSV-based skin segmentation to isolate hands in complex environments.
- Robust Feature Extraction: Utilizes ORB keypoints to enhance gesture features before classification.
- Data Augmentation Pipeline: Improves model generalization and reduces overfitting using rotational shifts, brightness adjustments, and zooms.
- Prevention of Overfitting: Implements Early Stopping and Dropout layers for a more reliable model.
 ## Dataset
 ASL Kaggle dataset containing 36 classes (A–Z and 0–9). Each class has ~70 images.

## Model Pipeline 
### 1. Data Collection & Preprocessing

- Grayscale conversion: All images are converted to grayscale to reduce computational complexity by eliminating the RGB color channels,  while retaining essential structural information of the gestures.
- Resizing: Each image is resized to 64 × 64 pixels, ensuring uniform input dimensions for the CNN.
- Normalization: Pixel values originally in the range [0, 255], are scaled down to the range [0, 1], which speeds up training and improves convergence by providing smaller, consistent input values for the optimizer.
- Reshaping: Each processed image is reshaped into (64, 64, 1) to include the single grayscale channel matching the expected input shape of the model.
- Label Encoding:  String labels (e.g., 'A', 'B', '1') are mapped to numerical values and then one-hot encoded vectors (e.g., 'A' becomes [1, 0, 0, ..., 0]) to be compatible with the categorical cross-entropy loss function for multi-class classification.
- Train/Test Split: The dataset is divided into 80% training data and 20% test data, ensuring fair evaluation of the model’s performance.

### 2. Data Augmentation

The following transformations were applied randomly during each epoch:

- Rotation: Random rotations within a range of ±15 degrees to account for slight hand tilts.
- Width & Height Shifts: Random horizontal and vertical shifts to simulate the hand not being perfectly centered.
- Brightness Adjustments: Randomly darkening or brightening images to improve robustness against changing lighting conditions.
- Zoom Variations: Randomly zooming in or out slightly on the image.

Here we have used ImageDataGenerator to artificially expand dataset.

### 3. CNN Model Architecture

The proposed model is a Convolutional Neural Network (CNN) designed to classify hand gestures efficiently. Its architecture contains approximately 1.63M trainable parameters:
- Conv2D Layer: 32 filters, kernel size 3×3, ReLU activation → followed by MaxPooling (2×2)
- Conv2D Layer: 64 filters, kernel size 3×3, ReLU activation → followed by MaxPooling (2×2)
- Flatten Layer: Converts the feature maps into a 1D vector
- Dense Layer: 128 neurons, ReLU activation
-Dropout Layer: Dropout rate = 0.5, randomly deactivates neurons during training to prevent overfitting
- Dense (36 outputs, Softmax) → Probability distribution across gesture classes
### 4. Model Compilation & Training
Input pipeline includes augmented data
- Loss Function: Categorical Crossentropy - ideal for measuring the performance of a model where the output is a probability distribution over multiple classes.
- Optimizer: Adam (Adaptive Moment Estimation(lr = 0.001)) - an efficient optimizer that adapts the learning rate during training, leading to faster convergence.
- Metric: Accuracy - the primary metric for monitoring training and evaluation performance.
To ensure efficient and effective training, we implemented a callback:
- Early Stopping: Monitored the validation loss with a patience of 5 epochs. This halts training if the validation loss does not improve for 5 consecutive epochs and restores the model weights from the best epoch observed.

The model was trained for 20 epochs using batches of 32 images, with the augmented data streamed directly from the ImageDataGenerator.

### 5. Evaluation
The trained model is evaluated on the held-out test set, achieving a test accuracy of 95.43%. Accuracy and Loss curves are plotted over epochs to analyze training and validation trends.A Confusion Matrix and A Classification Report with Precision, Recall, and F1-Score is provided for a detailed view of per-class predictions, highlighting strengths and weaknesses across different gestures.
and to ensure balanced evaluation of all classes.

### Real-Time Gesture Detection

Here we have used OpenCV + Webcam for real time detection.

- Frame Capture: OpenCV captures continuous frames from the webcam.
- Background Subtraction: A background subtractor is applied to isolate moving objects (the hand) from the static background, making the system less sensitive to clutter.
- Region of Interest (ROI): A fixed rectangle is defined on the screen where the user is expected to place their hand, simplifying the search area.
- Skin Segmentation: The ROI is converted to the HSV color space. A predefined range of Hue and Saturation values is used to create a mask that isolates skin-toned pixels, effectively separating the hand from other objects.
- Feature Enhancement (ORB): The ORB (Oriented FAST and Rotated BRIEF) algorithm detects keypoints and computes descriptors on the segmented hand image. These keypoints are drawn onto the frame, providing visual feedback and helping to highlight distinctive features of the gesture.
- Preprocessing for Model: The ROI is then processed identically to the training data: converted to grayscale, resized to 64x64, normalized, and reshaped.
- Prediction & Display: The processed image is fed into the trained CNN model. The model outputs a probability distribution, and the class with the highest probability is selected as the prediction. This predicted label is then overlayed directly onto the live video feed.
