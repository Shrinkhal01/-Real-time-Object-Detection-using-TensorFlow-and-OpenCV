import cv2

#sets a threshold value (minnimum value) for the model to decide what it is 
thres = 0.50

#tcaptures the video from the webcam
cap = cv2.VideoCapture(1)
#Now , here we use cap to set the various aspects of the video being rendered
cap.set(3, 1850)
cap.set(4, 1850)
cap.set(10, 70)

#now we make a list and store the names of the classes from the coco.names      
classNames = []
classFile = "coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# following are config files for the model(they are actually binary files with the pre trained models)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'#how to process input data
weightsPath = 'frozen_inference_graph.pb'#learned weights


#Now,we load the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)

#Now, we set the model dimensions
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#Now starts the reading of the video and detection of the objects as per the threshold and the confidence value set by us   
while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 <= classId - 1 < len(classNames):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Output', img)
    cv2.waitKey(1)