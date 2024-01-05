import os
import numpy as np
import cv2
import tensorflow as tf

# Replace with your actual model path
model_path = "O:/OD/P7"

pb_file = os.path.join(model_path, 'frozen_inference_graph.pb')

# Read the graph
try:
    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
except tf.errors.NotFoundError:
    print(f"Error: Unable to find the file {pb_file}. Make sure the path is correct.")
    exit()

# Configuring GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.compat.v1.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame")
            break

        # Run the model on the frame
        img_in = frame[:, :, [2, 1, 0]]  # BGR2RGB
        outputs = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': img_in.reshape(1, img_in.shape[0], img_in.shape[1], 3)})

        # Visualize the results
        font = cv2.FONT_HERSHEY_SIMPLEX

        num_detections = int(outputs[0][0])
        for i in range(num_detections):
            classId = int(outputs[3][0][i])
            score = float(outputs[1][0][i])
            bbox = [float(v) for v in outputs[2][0][i]]

            x = bbox[1] * frame.shape[1]
            y = bbox[0] * frame.shape[0]
            right = bbox[3] * frame.shape[1]
            bottom = bbox[2] * frame.shape[0]

            cv2.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (225, 255, 0), thickness=2)
            cv2.putText(frame, str(classId), (int(x), int(y)), font, 1, (200, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
