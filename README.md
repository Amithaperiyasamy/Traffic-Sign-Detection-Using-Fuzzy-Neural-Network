TRAFFIC SIGN DETECTION USING FUZZY NEURAL NETWORK

Creating a Traffic Sign Detection using a Fuzzy Neural Network (FNN) involves a combination of image processing, machine learning, and fuzzy logic. Below is a basic outline and an initial code framework to get you started with such a project using Python, OpenCV, and a fuzzy neural network approach.

🧠 Project Overview
1. Input: Images or video feed containing traffic signs
2. Process:
Preprocessing images

Detecting traffic signs (e.g., using color + shape filtering or CNN)

Classifying using a Fuzzy Neural Network

3. Output: Detected and labeled traffic signs
🔧 Tools Required
Python 3.x

OpenCV

NumPy

scikit-fuzzy (for fuzzy logic)

TensorFlow/Keras (optional: if integrating CNN or deep learning)

Step 1: Import Required Libraries
import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
Step 2: Preprocessing the Image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (600, 400))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv, resized
Step 3: Traffic Sign Detection (Color and Shape Based)
def detect_red_signs(hsv_img):
    # Red color range in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = mask1 | mask2

    return mask
Step 4: Feature Extraction for Fuzzy Neural Network Input
def extract_features(mask, original_img):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter small objects
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            roi = original_img[y:y+h, x:x+w]
            aspect_ratio = float(w)/h
            features.append((area, len(approx), aspect_ratio))  # e.g., area, number of sides, aspect ratio
    return features
Step 5: Define Fuzzy Inference System
# Define fuzzy variables
area = ctrl.Antecedent(np.arange(0, 5000, 1), 'area')
sides = ctrl.Antecedent(np.arange(3, 10, 1), 'sides')
aspect = ctrl.Antecedent(np.arange(0.5, 2.0, 0.1), 'aspect')
sign_type = ctrl.Consequent(np.arange(0, 4, 1), 'sign_type')  # 0: Unknown, 1: Stop, 2: Yield, 3: Speed Limit

# Membership functions
area['small'] = fuzz.trimf(area.universe, [0, 500, 1500])
area['medium'] = fuzz.trimf(area.universe, [1000, 2000, 3000])
area['large'] = fuzz.trimf(area.universe, [2500, 4000, 5000])

sides['triangle'] = fuzz.trimf(sides.universe, [3, 3, 4])
sides['circle'] = fuzz.trimf(sides.universe, [6, 7, 8])
sides['octagon'] = fuzz.trimf(sides.universe, [8, 8, 8])

aspect['narrow'] = fuzz.trimf(aspect.universe, [0.5, 0.75, 1])
aspect['square'] = fuzz.trimf(aspect.universe, [0.9, 1.0, 1.1])
aspect['wide'] = fuzz.trimf(aspect.universe, [1.1, 1.5, 2.0])

sign_type['unknown'] = fuzz.trimf(sign_type.universe, [0, 0, 1])
sign_type['stop'] = fuzz.trimf(sign_type.universe, [1, 1, 2])
sign_type['yield'] = fuzz.trimf(sign_type.universe, [2, 2, 3])
sign_type['speed'] = fuzz.trimf(sign_type.universe, [3, 3, 4])

# Fuzzy rules
rules = [
    ctrl.Rule(area['large'] & sides['octagon'] & aspect['square'], sign_type['stop']),
    ctrl.Rule(area['medium'] & sides['triangle'], sign_type['yield']),
    ctrl.Rule(area['medium'] & sides['circle'] & aspect['square'], sign_type['speed']),
]

sign_ctrl = ctrl.ControlSystem(rules)
sign_detector = ctrl.ControlSystemSimulation(sign_ctrl)
Step 6: Apply Fuzzy Inference
def classify_signs(features):
    for f in features:
        area_val, sides_val, aspect_val = f
        sign_detector.input['area'] = area_val
        sign_detector.input['sides'] = sides_val
        sign_detector.input['aspect'] = aspect_val
        sign_detector.compute()
        output = sign_detector.output['sign_type']
        if output < 1.5:
            print("Detected: STOP Sign")
        elif output < 2.5:
            print("Detected: YIELD Sign")
        elif output < 3.5:
            print("Detected: SPEED LIMIT Sign")
        else:
            print("Unknown Sign")
Step 7: Putting It All Together
if __name__ == "__main__":
    hsv_img, original_img = preprocess_image("traffic.jpg")
    red_mask = detect_red_signs(hsv_img)
    features = extract_features(red_mask, original_img)
    classify_signs(features)





