# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:47:36 2019

@author: hp
"""

import os
from PIL import Image
import time
import imutils
import cv2
import numpy as np
from numpy import newaxis
from imutils.video import VideoStream
from keras.models import model_from_yaml


class FaceDetectorPredictor:

    def __init__(self):
        '''
        # Creating 'object' for Database Updation
        self.DbU = DatabaseUpdater()
        # Creating Connection with Database
        self.DbU.createConnection("attendance")
        # Obtaing Roll Numbers from the database
        self.DbU.getRollNos("Sub1")
        '''

        # Initializing paths for 'prototxt', 'model' and 'confidence'
        prototxt_path = "TrainingEntities/deploy.prototxt"
        model_path = "TrainingEntities/res10_300x300_ssd_iter_140000.caffemodel"
        self.threshold_confidence = 0.14

        # Load Serialized Model from Disk
        self.modal = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Starting WebCam
        self.vCap = VideoStream(src=0).start()
        time.sleep(2.0)

        # load YAML and create model
        yaml_file = open('TrainedEntities/modal.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.loaded_model = model_from_yaml(loaded_model_yaml)

        # load weights into new model
        self.loaded_model.load_weights("TrainedEntities/modal.h5")
        print("Loaded model from disk")
        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        '''
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        '''

    def FaceDetector(self):
        self.flag = 0
        # newFrame = cv2.imread("hhh.jpg")
        # Loop over the Frames from the Video Stream
        while True:

            # Reading returned captured frame Image
            capturedFrame = self.vCap.read()

            a, b, c = np.mean(capturedFrame, axis=(0, 1))

            a = int(a) * 1.0
            b = int(b) * 1.0
            c = int(c) * 1.0

            newFrame = imutils.resize(capturedFrame, width=700)  # some edititng

            # Grab the Frame Dimensions and convert it to a 'Blob'
            (h, w) = newFrame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(newFrame, (300, 300)), 1.0, (300, 300), (a, b, c))

            # Pass the 'Blob' through the network and obtain the 'Detections' and 'Predictions'
            self.modal.setInput(blob)
            detections = self.modal.forward()

            # Loop over all the Detections
            for i in range(0, detections.shape[2]):
                # Extract the 'confidence' (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
                if confidence < self.threshold_confidence:
                    continue

                # Compute the (X, Y)-coordinates of the Bounding Box for the Object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (self.startX, self.startY, self.endX, self.endY) = box.astype("int")

                # Draw the Bounding Box of the face along with the Associated Probability
                confidencePercentage = "{:.2f}%".format(confidence * 100)
                if ((self.startY - 10) > 10):
                    y = self.startY - 10
                else:
                    y = self.startY + 10

                grayFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
                print("Values", self.startX, self.startY, self.endX, self.endY, "shown,", h, w, "hw shown")
                if (self.startX > 0 and self.startY > 0):
                    if (self.endX < 700 and self.endY < 525):
                        faceFrame = grayFrame[self.startY:self.endY, self.startX:self.endX];

                        faceFrame = cv2.resize(faceFrame, (224, 224))
                        cv2.imshow("Frame", faceFrame)
                        faceFrame = faceFrame[newaxis, :224, :224, newaxis]

                        Id = "helo"

                        Id = self.loaded_model.predict(faceFrame)
                        print(Id)
                        max = Id[0][0]
                        j = 0
                        for i in Id[0]:
                            j += 1
                            if i > max:
                                Id = str(j - 1)
                                break

                cv2.rectangle(newFrame, (self.startX, self.startY), (self.endX, self.endY), (0, 255, 0), 2)
                cv2.putText(newFrame, Id, (self.startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # Showing the captured frame image
            cv2.imshow("Frame", newFrame)
            # cv2.imwrite("output.jpg", newFrame)

            # Checking which key was pressed
            keyPressed = cv2.waitKey(1)

            if (keyPressed == ord("q") or cv2.waitKey(1) == ord("Q")):
                break

                # Exiting Camera
        self.vCap.stop()

        # Destroying All Windows
        cv2.destroyAllWindows()


def main():
    # Creating object of the class
    DG = FaceDetectorPredictor()

    # Detecting the faces in the frame
    DG.FaceDetector()


if __name__ == "__main__":
    main()