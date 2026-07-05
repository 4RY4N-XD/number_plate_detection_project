# yolov3-from-opencv-object-detection
Markdown
# YOLOv3 Number Plate Detection Project

A computer vision application built using Python, OpenCV (DNN module), and EasyOCR to detect vehicle license plates and extract registration characters from images.

---

##  Project Structure

This repository is structured exactly as follows. Please note that heavy deep-learning weights are excluded from Git synchronization to prevent upload failures and repository bloating.

```text
NUMBER_PLATE_DETECTIO...
└── yolov3-from-opencv-object-detection/
    ├── __pycache__/
    │   └── util.cpython-313.pyc
    │
    ├── model/
    │   ├── cfg/
    │   │   ├── .gitignore
    │   │   └── darknet-yolov3.cfg     # YOLOv3 network topology
    │   │
    │   ├── weights/
    │   │   ├── .gitignore
    │   │   └── model.weights          #  EXCLUDED FROM GITHUB (Add manually)
    │   │
    │   └── class.names                # Dataset labels
    │
    ├── .gitignore                     # Git tracking exclusions
    ├── LICENSE.md                     # Project license file
    ├── main.py                        # Main Python inference pipeline
    ├── README.md                      # Documentation setup guide
    ├── requirements.txt               # Package manifest
    ├── tempCodeRunnerFile.py          # Script runner log
    ├── test2.jpg                      # Sample target test image
    └── util.py                        # Output parsing and Non-Maximum Suppression (NMS)
```
# Complete Setup Instructions


1. Clone the Repository
Open your command prompt or terminal and download the project files:

```Bash
git clone [https://github.com/4RY4N-XD/number_plate_detection_project.git](https://github.com/4RY4N-XD/number_plate_detection_project.git)
cd yolov3-from-opencv-object-detection
```
2. Configure Your Environment & Dependencies
This project relies on OpenCV 4.x because native parsing support for Darknet format configs and weights has been entirely removed from OpenCV 5.0+. Run this exact installation sequence:

```DOS
"C:\Program Files\Python313\python.exe" -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
"C:\Program Files\Python313\python.exe" -m pip install opencv-python==4.10.0.84 easyocr numpy matplotlib
```
3. Restore the Excluded Model Weights File (Crucial) 
The network model weights file (model.weights) is approximately 234 MB and has been safely untracked by the local repository's .gitignore rules to keep the codebase clean.

To add this file back to your local setup so the script can execute successfully:

Obtain the required model.weights file from your cloud storage backup or download resource.

In your file explorer, open the project directory: yolov3-from-opencv-object-detection/

Navigate into the model folder, then open the weights subfolder.

Place your downloaded weights file explicitly in this folder.

Make sure the file is named exactly: model.weights
(Absolute Path verification: yolov3-from-opencv-object-detection/model/weights/model.weights)

# Running the Inference Application
After placing the weights file into the correct sub-directory, run the detector via your terminal interface:

```DOS
"C:\Program Files\Python313\python.exe" main.py
```

# Enabling Hardware/GPU Acceleration
If your desktop or laptop setup contains an NVIDIA GPU alongside an active CUDA environment, you can shift the text evaluation workload from your CPU to your graphic processor. Open main.py and ensure the OCR block initialization reads:

```python
reader = easyocr.Reader(['en'], gpu=True)
```
---

### How to upload this updated documentation to GitHub now:
Save the file as `README.md`, open your terminal window, and run these synchronization flags to force-push the complete code and your `test2.jpg` file online:

```cmd
git add README.md main.py test2.jpg
git commit -m "Update README directory tree map and operational guide"
git push -f origin main
