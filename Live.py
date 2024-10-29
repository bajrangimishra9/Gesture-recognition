import cv2
import imutils
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import time

# Global variables
bg = None

# Load model and categories
def _load_weights():
    try:
        model = load_model("hand_gesture_recognition_model_b.h5")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Categories for gestures
CATEGORIES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Function to accumulate background average
def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

# Function to segment the hand region
def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    segmented = max(cnts, key=cv2.contourArea)
    return thresholded, segmented

# Function to predict the gesture
def getPredictedClass(model):
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (50, 50))  # Resize to match model input shape
    gray_image = gray_image.reshape(1, 50, 50, 1) / 255.0  # Normalize
    prediction = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction)
    return CATEGORIES[predicted_class] if predicted_class < len(CATEGORIES) else "Unknown"

# Matplotlib display function
def display_with_matplotlib(frame):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)
    plt.clf()

# Main function
if __name__ == "__main__":
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False
    model = _load_weights()
    k = 0
    predictedClass = "Calibrating..."

    plt.ion()  # Turn on interactive mode for Matplotlib

    while True:
        start_time = time.time()

        # Read the frame from the camera
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        # Resize the frame and flip it
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        # Define the region of interest
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Calibration for the background
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successful...")
        else:
            hand = segment(gray)
            if hand is not None:
                thresholded, segmented = hand
                if k % 5 == 0:  # Classify every 5 frames
                    cv2.imwrite('Temp.png', thresholded)
                    predictedClass = getPredictedClass(model)

                # Display the thresholded image on the main feed
                thresholded_resized = cv2.resize(thresholded, (150, 150))  # Resize for display
                clone[10:160, 10:160] = cv2.cvtColor(thresholded_resized, cv2.COLOR_GRAY2BGR)

        # Draw rectangle around ROI and display prediction text
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(clone, "Prediction: " + predictedClass, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with Matplotlib
        display_with_matplotlib(clone)

        # Adjust frame rate for smoothness
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.001:
            time.sleep(0.001 - elapsed_time)

        # Break the loop if any key is pressed in matplotlib
        if plt.waitforbuttonpress(timeout=0.001):
            break

        num_frames += 1
        k += 1

    camera.release()
    plt.close()  # Close the Matplotlib window
