from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import sys
import os

sub_data = sys.argv[1]
datasets = 'datasets'

confidence_level = 0.5
print("loading..")
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"


def detect_face(frame, faceNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= confidence_level:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            locs.append((startX, startY, endX, endY))

    return locs


faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("starting video stream...")
vs = VideoStream(src=0).start()


def start_training(count):
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        locs = detect_face(frame, faceNet)

        for box in zip(locs):
            for face in box:
                (startX, startY, endX, endY) = face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = gray[startY:endY, startX:endX]
                cv2.imwrite('%s/%s.png' % (path, count), face)
                count += 1

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            break


if not os.path.isdir(datasets):
    os.mkdir(datasets)
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)
    start_training(0)
else:
    num = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            num.append(int(file.split(".")[0]))
    start_training(max(num) + 1)
