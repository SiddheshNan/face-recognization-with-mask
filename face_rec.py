from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os

datasets = 'datasets'

confidence_level = 0.5
prediction_level = 60

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

blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)


def start():
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        total_boxes = detect_face(frame, faceNet)

        for box in zip(total_boxes):
            for face in box:
                (startX, startY, endX, endY) = face
                cv2.rectangle(frame, (startX, startY), (endX, endY), green_color, 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = gray[startY:endY, startX:endX]

                face_resize = cv2.resize(face, (width, height))
                prediction = model.predict(face_resize)

                if prediction[1] >= prediction_level:
                    print('%s - %.0f' % (names[prediction[0]], prediction[1]))
                    cv2.rectangle(frame, (startX, startY), (endX, endY), green_color, 2)
                    cv2.putText(frame, '%s - %.0f' % (names[prediction[0]], prediction[1]), (startX - 10, startY - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.3, green_color, 1)
                else:
                    print('face not recognized')
                    cv2.rectangle(frame, (startX, startY), (endX, endY), red_color, 2)
                    cv2.putText(frame, 'not recognized', (startX - 10, startY - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.3, red_color, 1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            break


(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (100, 130)
(images, labels) = [np.array(lis) for lis in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
start()
