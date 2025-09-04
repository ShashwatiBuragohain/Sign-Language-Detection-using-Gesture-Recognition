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
<p align="center">
  <img src="https://github.com/ShashwatiBuragohain/Sign-Language-Detection-using-Gesture-Recognition/blob/4ee3d558064fa6d6c0dbe0fd7813729728057c03/ASL%20OUTPUT.png?raw=true" width="60%" />
</p>

  *Snippet of the real-time processing code, showing the combination of background subtraction and HSV-based skin segmentation to isolate the hand.*

## Features
- Real-Time Webcam Integration: Processes live video feed for instant gesture recognition.
- Advanced Preprocessing: Employs background subtraction and HSV-based skin segmentation to isolate hands in complex environments.
- Robust Feature Extraction: Utilizes ORB keypoints to enhance gesture features before classification.
- Data Augmentation Pipeline: Improves model generalization and reduces overfitting using rotational shifts, brightness adjustments, and zooms.
- Prevention of Overfitting: Implements Early Stopping and Dropout layers for a more reliable model.
 ## Dataset
 We used the [ASL dataset from Kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset), which contains **36 classes** representing the letters **A–Z** and digits **0–9**, with approximately **70 images per class**.

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

### 6. Real-Time Gesture Detection

Here we have used OpenCV + Webcam for real time detection.

- Frame Capture: OpenCV captures continuous frames from the webcam.
- Background Subtraction: A background subtractor is applied to isolate moving objects (the hand) from the static background, making the system less sensitive to clutter.
- Region of Interest (ROI): A fixed rectangle is defined on the screen where the user is expected to place their hand, simplifying the search area.
- Skin Segmentation: The ROI is converted to the HSV color space. A predefined range of Hue and Saturation values is used to create a mask that isolates skin-toned pixels, effectively separating the hand from other objects.
- Feature Enhancement (ORB): The ORB (Oriented FAST and Rotated BRIEF) algorithm detects keypoints and computes descriptors on the segmented hand image. These keypoints are drawn onto the frame, providing visual feedback and helping to highlight distinctive features of the gesture.
- Preprocessing for Model: The ROI is then processed identically to the training data: converted to grayscale, resized to 64x64, normalized, and reshaped.
- Prediction & Display: The processed image is fed into the trained CNN model. The model outputs a probability distribution, and the class with the highest probability is selected as the prediction. This predicted label is then overlayed directly onto the live video feed.

## Challenges and Limitations
### Environmental Challenges

#### Lighting Sensitivity

**Problem:**  
The performance of the skin segmentation algorithm is highly dependent on consistent and neutral lighting.  
Under strong yellow/warm light, the HSV range for skin can shift toward lower values, causing the algorithm to fail to detect the hand.  
Conversely, under very cool or blue-tinted light, the skin tone can be misclassified as outside the defined range.  
Shadows cast on the hand also create drastic changes in pixel values, which can break up the hand mask or be misinterpreted as part of the gesture.

**Observed Effect:**  
In suboptimal lighting, the system either fails to recognize a hand is present or produces a corrupted, fragmented mask.  
This corrupted input is then passed to the CNN, which almost always results in an incorrect or low-confidence prediction.

---

#### Background Complexity

**Problem:**  
While background subtraction helps isolate moving objects, it is not foolproof. The current implementation struggles with:

- **Cluttered Backgrounds:** Busy backgrounds with many edges and colors (e.g., a bookshelf, another person moving) can introduce noise that is not fully subtracted away, contaminating the Region of Interest (ROI).
- **Skin-Colored Objects:** If an object in the background (e.g., a wooden table, a poster) has a color similar to the predefined skin tone range, it may be incorrectly included in the mask. This "false positive" segmentation adds significant noise to the input image.

**Observed Effect:**  
Noisy inputs from complex backgrounds confuse the CNN, which was trained on clean, centered images of hands.  
This leads to a sharp drop in prediction accuracy.

---

#### Camera Quality and Auto-Adjustments

**Problem:**  
Consumer-grade webcams often have auto-adjustment features for white balance, exposure, and focus.  
These automated corrections can change the image properties from frame to frame without user input.  
For instance, if the hand moves into the frame, the camera might automatically adjust its exposure, altering the apparent skin color and violating the assumptions of our static HSV range.

**Observed Effect:**  
This creates an inconsistent input stream for the model, where the same hand under the same conditions can look different from one moment to the next, leading to unpredictable performance.

---

##  Impact on the Pipeline

These environmental factors primarily degrade the quality of the input before it even reaches the CNN model.  
The model itself is powerful and accurate, but its performance is entirely dependent on receiving clean, well-segmented input that resembles the training data.

## Acknowledgment

This project was completed as part of the Machine Learning lab (EE 524) course at Indian Institute of Technology Guwahati (IITG).


