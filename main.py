import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import util

# 1. Define constants with universal, relative paths
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

# FIX: Changed from absolute local Windows path to a clean relative path
img_path = os.path.join('.', 'test2.jpg') 

# Load class names
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

# Load model
net = cv2.dnn.readNet(model_cfg_path, model_weights_path)

# Load image
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Test image not found at {img_path}. Please make sure 'test2.jpg' is in the root directory.")

img = cv2.imread(img_path)
H, W, _ = img.shape

# Convert image to blob
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

# Get detections
net.setInput(blob)
detections = util.get_outputs(net)

# bboxes, class_ids, confidences
bboxes = []
class_ids = []
scores = []

for detection in detections:
    bbox = detection[:4]
    xc, yc, w, h = bbox
    bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

    class_id = np.argmax(detection[5:])
    score = np.amax(detection[5:])

    bboxes.append(bbox)
    class_ids.append(class_id)
    scores.append(score)

# Apply NMS
bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

# Initialize EasyOCR reader (Recommended to keep GPU=True if you have CUDA setup)
reader = easyocr.Reader(['en'], gpu=True)

license_plate = None
license_plate_gray = None
license_plate_thresh = None

for bbox_, bbox in enumerate(bboxes):
    xc, yc, w, h = bbox

    # Draw bounding box
    img = cv2.rectangle(img,
                        (int(xc - (w / 2)), int(yc - (h / 2))),
                        (int(xc + (w / 2)), int(yc + (h / 2))),
                        (0, 255, 0),
                        10)
    
    # Safe boundary cropping
    ymin, ymax = max(0, int(yc - (h / 2))), min(H, int(yc + (h / 2)))
    xmin, xmax = max(0, int(xc - (w / 2))), min(W, int(xc + (w / 2)))
    
    license_plate = img[ymin:ymax, xmin:xmax, :].copy()
    license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY)

    output = reader.readtext(license_plate_thresh)
    
    print(f"--- Detections for Plate #{bbox_+1} ---")
    for out in output:
        text_bbox, text, text_score = out
        if text_score > 0.4:
            print(f"Text: {text} | Confidence: {text_score:.2f}")

# Plotting results safely
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected Frame")

if license_plate is not None:
    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Color Plate")

    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))
    plt.title("Grayscale Plate")

    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))
    plt.title("Thresholded Plate")
else:
    print("No plates detected to display processing stages.")

plt.show()