# -Real-time-Object-Detection-using-TensorFlow-and-OpenCV



## Overview

This project provides a Python script for real-time object detection using TensorFlow and OpenCV. It leverages a pre-trained deep learning model to identify and visualize objects in a live webcam feed, displaying bounding boxes, class labels, and confidence scores for detected objects.

## Features

- Integration of a pre-trained object detection model
- Real-time processing of webcam feed
- Visualization of detected objects with bounding boxes, class labels, and confidence scores

## Requirements

- TensorFlow
- OpenCV
- NumPy

## Usage

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/Real-time-Object-Detection-using-TensorFlow-and-OpenCV.git
    cd Real-time-Object-Detection-using-TensorFlow-and-OpenCV
    ```

2. **Replace the `model_path` Variable**
   Update the `model_path` variable in `od.py` with the path to your pre-trained model.

3. **Modify the Camera Source**
   Change the value `1` to `0` in the `cap` variable to use the default integrated webcam:
   ```python
   cap = cv2.VideoCapture(0)


Run the Script
    ```bash
       python od.py
    ```
Customization
Feel free to customize and extend the code to suit your specific use case. Contributions are welcome!

Files
    ```bash
      od.py: Main script for real-time object detection.
      coco.names: List of object class names.
      frozen_inference_graph.pb: Pre-trained model weights.
      ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt: Model configuration file.
    ```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements and bug fixes.

## Acknowledgments
Thanks to the TensorFlow and OpenCV communities for their excellent tools and libraries.

Thank you for using this script! If you have any questions or suggestions, please feel free to reach out:
```bash
shrinkhalshrinkhal@gmail.com
```
