# Facial Authenticated Folder Locking System

## Aim

The aim of this project is to create a facial authentication system for securing folders. This system uses a pre-trained convolutional neural network (CNN) to recognize faces through a webcam. Upon successful facial authentication, users can choose to hide or unhide a specified folder.

## Algorithm

1. **Face Detection:**
   - Utilizes Haar Cascades for face detection in webcam frames.

2. **Facial Authentication:**
   - Captures frames from the webcam.
   - Converts frames to grayscale and detects faces.
   - Resizes the detected face region to a target size.
   - Normalizes pixel values and uses a pre-trained CNN model for facial authentication.

3. **Folder Manipulation:**
   - Provides options to hide or unhide a user-specified folder.
   - Uses OS commands to set or remove the hidden attribute.

## Code

The code is written in Python and uses the following libraries:

- OpenCV (`cv2`) for webcam interaction and image processing.
- NumPy for numerical operations.
- TensorFlow and Keras for loading the pre-trained CNN model.
- Streamlit for creating the web application.

Make sure to replace `'model.h5'` with the actual path to your pre-trained model before running the script.

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import streamlit as st

# Function to hide a folder
def hide_folder(folder_path):
    try:
        # Create a hidden attribute for the folder
        os.system(f"attrib +h \"{folder_path}\"")
        return f"The folder '{folder_path}' is now hidden."
    except Exception as e:
        return f"Error: {e}"

# Function to unhide a folder
def unhide_folder(folder_path):
    try:
        # Remove the hidden attribute from the folder
        os.system(f"attrib -h \"{folder_path}\"")
        return f"The folder '{folder_path}' is now visible."
    except Exception as e:
        return f"Error: {e}"

# Function to load and preprocess the single image from the webcam
def load_and_preprocess_webcam_image(target_size):
    try:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame is not empty
        if not ret or frame is None:
            print("Error capturing frame from webcam.")
            return None, None

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use a face detection model (e.g., Haar Cascades) to detect faces in the frame
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = gray_frame[y:y + h, x:x + w]

            # Resize the face region to the target size
            resized_face = cv2.resize(face_roi, target_size)

            # Convert the resized face to a NumPy array
            img_array = img_to_array(resized_face)

            # Reshape the array to (1, height, width, channels) as there is only one image
            img_array = img_array.reshape((1,) + img_array.shape)

            # Normalize the pixel values
            img_array /= 255.0

            # Use your pre-trained model to predict if the face belongs to you
            prediction = face_model.predict(img_array)

            return frame, prediction[0][0]

    except Exception as e:
        print("Error capturing frame from webcam:", e)
        return None, None

# Load your pre-trained CNN model
model_path = 'model.h5'  # Replace with the actual path to your model
face_model = load_model(model_path)

# Set the target size for the image
target_size = (64, 64)

# Streamlit title
st.title("Facial Authenticated Folder Locking System")

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Streamlit placeholder for displaying webcam image
image_placeholder = st.empty()

try:
    authenticated = False

    while not authenticated:
        # Capture, preprocess, and recognize faces in the webcam frame
        processed_frame, prediction = load_and_preprocess_webcam_image(target_size)

        # Check if the processed frame is not empty and prediction is not None
        if processed_frame is not None and prediction is not None:
            # Display the processed frame
            image_placeholder.image(processed_frame, channels="BGR")

            # Allow access to hide/unhide the folder only if the recognized face is positive
            if prediction > 0.5:
                authenticated = True
                st.sidebar.markdown("### Facial Authentication Successful")
            else:
                st.sidebar.markdown("### Facial Authentication Failed")
except KeyboardInterrupt:
    # Release the webcam when the script is interrupted
    cap.release()
    st.sidebar.markdown("### Script Interrupted")

finally:
    if authenticated:
        # Prompt the user for the folder path
        folder_path = st.text_input("Enter the path to the folder you want to hide/unhide:")

        # Ask the user whether to hide or unhide the folder
        action = st.radio("Do you want to hide or unhide the folder?", ('Hide', 'Unhide'))

        # Perform the selected action on the folder
        if action == 'Hide':
            hide_folder(folder_path)
            st.write(f"The folder '{folder_path}' is now hidden.")
        elif action == 'Unhide':
            st.warning("Warning: Unhiding the folder will only occur after you press the 'Unhide Folder' button.")
        else:
            st.write("Invalid action. Please select 'Hide' or 'Unhide.")

        # Streamlit button to trigger folder unhiding, visible only when 'Unhide' is selected
        if action == 'Unhide':
            if st.button("Unhide Folder"):
                unhide_folder(folder_path)
                st.success(f"The folder '{folder_path}' is now visible.")
```
## OUTPUT:
###INTERFACE
<img width="960" alt="2" src="https://github.com/Surendar14/Facial-Authenticated-Folder-Locking-System/assets/75235759/37f8f9ac-ee23-4fb3-85d9-1f6359e23361">
### AUTHENTICATION
<img width="960" alt="3" src="https://github.com/Surendar14/Facial-Authenticated-Folder-Locking-System/assets/75235759/15339b18-cf90-480e-935b-9d728d435cb8">
### FOLDER SELECTION
<img width="960" alt="4" src="https://github.com/Surendar14/Facial-Authenticated-Folder-Locking-System/assets/75235759/e5ecec22-b540-422b-9049-352771aec408">
### FOLDER HIDING
<img width="960" alt="5" src="https://github.com/Surendar14/Facial-Authenticated-Folder-Locking-System/assets/75235759/2df9cc0f-0318-4920-8e12-970672d4e8c4">
### FOLDER HIDDEN
<img width="960" alt="8" src="https://github.com/Surendar14/Facial-Authenticated-Folder-Locking-System/assets/75235759/d164729a-7117-454e-ac47-e773d6faac2b">
### FOLDER UNHIDING
<img width="960" alt="10" src="https://github.com/Surendar14/Facial-Authenticated-Folder-Locking-System/assets/75235759/a81d990b-fb39-45c4-87f3-d3277aa67a44">
### RESULT
<img width="960" alt="11" src="https://github.com/Surendar14/Facial-Authenticated-Folder-Locking-System/assets/75235759/7b9591fa-d452-4d42-9563-4f72f1e7f8a3">




## RESULT:
The main script, when executed, opens a Streamlit web application. It captures frames from the webcam, performs facial authentication, and allows the user to hide or unhide a specified folder based on the authentication result.

The final result is a facial authentication system with folder locking capabilities.
