# Object Detection using SSD MobileNet v3
This project demonstrates object detection using SSD MobileNet v3 with pre-trained models on the COCO dataset. The code captures video from your webcam, detects objects in real time, and labels them with bounding boxes and confidence scores.

### Table of Contents
Overview
Requirements
Setup
Usage
COCO Classes
References

### Overview
This project uses OpenCV's DNN module to detect objects in real-time using a pre-trained model on the COCO dataset. The detection process is powered by the SSD MobileNet v3 architecture.

### Key Features:

Real-time object detection from a webcam feed.
Bounding boxes and confidence scores are displayed for detected objects.
Utilizes pre-trained models for quick and accurate detection.
Requirements
Python 3.x
OpenCV (cv2)
Pre-trained weights and config files for SSD MobileNet v3
COCO class names file
Install the required packages:
```
pip install opencv-python
```


### Setup
Clone the repository or download the necessary files:
```
OD.py (Main Python script)
coco.names (COCO class labels)
frozen_inference_graph.pb (Pre-trained weights)
ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt (Model configuration)
Place all the files in the same folder.
```
Make sure your webcam is connected and working.

Usage
Run the Python script:
```
python OD.py
```
The script will open a window showing the webcam feed, and detected objects will be highlighted with bounding boxes along with their labels and confidence scores.

Press q to quit the window.

COCO Classes

### The model is trained on the COCO dataset, which contains 80 common object classes such as:

Person
Bicycle
Car
Dog
etc.
###The full list of classes is in the coco.names file included in this project.

#References
SSD MobileNet v3
COCO Dataset
OpenCV DNN module
