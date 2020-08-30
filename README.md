# detect-bubble
Classify & detect dialogue on manga pages

## requirements
- opencv-python
- tensorflow2

## Image Classifier & Custom Dataset
Custom Dataset, contains 3 labels: 
- Cloud
- Oval
- Square

Link : https://drive.google.com/drive/folders/15hcGAZtAKd9UMmt-kLcD-3nT99uLC7Qv?usp=sharing

Credits for this dataset to https://github.com/Shiroechi

## Darknet YoloV3 Object Detection & Manga109 Dataset
Download the [yolov3_manga109_weights](https://drive.google.com/file/d/1-8A9wdYlCb5V6nX5HzYS_FByXTR1bD9X/view?usp=sharing) and the [yolov3_configuration_file](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.cfg)

## Use cv2 for detection

```py
import cv2
import numpy as np
import glob
import random
import matplotlib.pyplot as plt

# Load yolov3 model configuration & the weights
net = cv2.dnn.readNet("yolov3_manga109_v2_5000.weights", "yolov3.cfg")

# Get all the image path from the test folder.
images_path = glob.glob(r"test\*.jpg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# For each image in test folder
for img_path in images_path:
    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (512, 512), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.25:
                # Detection output is `nomralized` (center_x, center_y, width, height)
                # Convert back, multiply them by the page width/height.
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate (x,y) to get (x,y,w,h) bbox format
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.45)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(img,(x, y),(x + w, y + h),(0, 0, 255), 2)
            cv2.putText(img, 'text', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    plt.imshow(img)
    plt.show()
```
