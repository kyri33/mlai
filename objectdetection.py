import numpy as np
from cv2 import cv2
import time
import os

LABELS = open("yolo/coco.names").read().strip().split('\n')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet("yolo/yolov3.cfg", "yolo/yolov3.weights")

image = cv2.imread("yolo/images/basketball2.jpg")

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

(H, W) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),
        swapRB=True, crop=False)

CONFIDENCE = 0.4
THRESHOLD = 0.3

net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        best = np.argmax(scores)
        confidence = scores[best]

        if confidence > CONFIDENCE:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype('int')
            x = int(centerX - (width / 2))
            y = int(centerY - (width / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(best)

#idxs = np.array(boxes)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, THRESHOLD)
if len(idxs) > 0:
    for i in range(len(boxes)):
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

cv2.imshow("IMage", image)
cv2.waitKey(0)