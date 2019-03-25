import os
import time
import imutils
import cv2
import numpy as np
from imutils.video import VideoStream


class FaceDataSetGenerator:

    def __init__(self):
        # Initializing paths for 'prototxt', 'model' and 'confidence'
        prototxt_path = "TrainingEntities/deploy.prototxt"
        model_path = "TrainingEntities/res10_300x300_ssd_iter_140000.caffemodel"
        self.threshold_confidence = 0.5

        # Load Serialized Model from Disk
        self.modal = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Starting WebCam
        self.vCap = VideoStream(src=0).start()
        time.sleep(2.0)

        # Setting sampleIndex and initial path
        self.sampleIndex = 1
        self.path = "DataSet/"

    def GetUID(self):
        # Inputting User ID
        self.U_ID = input("Enter User ID: ")

    def SetPath(self):
        # Setting path according to User ID
        self.path = self.path + str(self.U_ID) + "/"

        # Creating U_ID directory if not present
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def SaveFrame(self, newFrame, frameParameters):
        # Convering the RGB image to GrayScale Image
        grayFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)

        # Saving the face portion of the captures image frame
        cv2.imwrite(self.path + "User_" + self.U_ID + "_" + str(self.sampleIndex) + ".jpg",
                    grayFrame[frameParameters[1]:frameParameters[3], frameParameters[0]:frameParameters[2]])

    def FaceDetector(self):
        self.flag = 0

        # Loop over the Frames from the Video Stream
        while True:
            # Reading returned captured frame Image
            capturedFrame = self.vCap.read()
            newFrame = imutils.resize(capturedFrame, width=700)  # some edititng

            # Grab the Frame Dimensions and convert it to a 'Blob'
            (h, w) = newFrame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(newFrame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

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
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the Bounding Box of the face along with the Associated Probability
                confidencePercentage = "{:.2f}%".format(confidence * 100)
                if ((startY - 10) > 10):
                    y = startY - 10
                else:
                    y = startY + 10

                cv2.rectangle(newFrame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(newFrame, confidencePercentage, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                # Checking conditions for saving the captured frame
                if (self.flag == 1):
                    # Checking if the captured frame is good enough for saving?
                    if ((float(confidence * 100)) > 75.00):
                        frameParameters = []
                        frameParameters.append(startX)
                        frameParameters.append(startY)
                        frameParameters.append(endX)
                        frameParameters.append(endY)

                        # Saving the captured frame
                        self.SaveFrame(newFrame, frameParameters)

                        # Incrementing the sample index
                        self.sampleIndex += 1

                        # Showing the captured frame image
            cv2.imshow("Frame", newFrame)

            # Checking which key was pressed
            keyPressed = cv2.waitKey(1)

            if (keyPressed == ord("s") or keyPressed == ord("S")):
                self.sampleIndex = 1
                self.flag = 1
            elif (self.sampleIndex == 151):
                break
            elif (keyPressed == ord("q") or cv2.waitKey(1) == ord("Q")):
                break

                # Exiting Camera
        self.vCap.stop()

        # Destroying All Windows
        cv2.destroyAllWindows()


def main():
    # Creating object of the class
    DG = FaceDataSetGenerator()

    # Getting Id of the user
    DG.GetUID()

    # Setting Path for saving the captured frames
    DG.SetPath()

    # Detecting the faces in the frame
    DG.FaceDetector()


if __name__ == "__main__":
    main()