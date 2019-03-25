import sys
import os
sys.path.append(os.path.abspath(__file__))

import cv2
import numpy as np
from PIL import Image
import keras
from VGG_CNN import *


class FaceDataSetTrainer:
    
    def __init__(self):
        
        obVGG_CNN = VGG_CNN()
        
        self.modal = obVGG_CNN.Modal()
        
        # Setting initial path
        self.path = "DataSet"     


    def getAllDirectoryPaths(self):
        self.allDirectoryPaths = []

        # Looping through all sub directories to generate all the directory paths
        for directory in os.listdir(self.path):
            self.allDirectoryPaths.append(self.path + ('//') + directory)

            

    def getAllImagePaths(self):        
        self.allImagePaths = []

        # Looping through all sub directory paths to gain the path of all images in the subdirectories
        for directoryPath in self.allDirectoryPaths:
            for directory in os.listdir(directoryPath):
                self.allImagePaths.append(directoryPath + ('//') + directory)
            
        
    def getImagesAndLabels(self):
        self.train_faceSamples = []
        self.train_faceIDs = []
        
        self.validation_faceSamples = []
        self.validation_faceIDs = []
        
        self.no_of_students = 4


        self.i = 0
        self.j = 0
        self.flag = 0
        self.c = 0
        # Looping through all the image paths to select all face images for training
        for imagePath in self.allImagePaths:
            # Converting image to gray scale
            gray = Image.open(imagePath).convert('L')
            grayImage = gray.resize((224, 224), Image.ANTIALIAS)
            
            # Converting grayscale image to numpy array
            npImage = np.array(grayImage, 'uint8')               
            
            # Extracting Roll Number
            rNo = (imagePath.split("_")[-2])
            
            
            if(self.flag == 0):
                self.i += 1
            
                # Creating array of face samples found
                self.train_faceSamples.append(npImage)
                
                # Creating array of face IDs using 'roll number and serial number'
                self.train_faceIDs.append(int(rNo))
            
                
            if(self.flag == 1):
                self.j += 1
                
                # Creating array of face samples found
                self.validation_faceSamples.append(npImage)
            
                # Creating array of face IDs using 'roll number and serial number'
                self.validation_faceIDs.append(int(rNo))
                
            
            if(self.i == 100):
                self.flag = 1
                self.i = 0
                self.j = 0
        
                
            if(self.j == 50):
                self.i = 0
                self.j = 0
                self.flag = 0
                
        
        
        # Converting list to n.array        
        self.train_faceSamples = np.array(self.train_faceSamples, 'uint8')
        self.validation_faceSamples = np.array(self.validation_faceSamples, 'uint8')
        
        
        self.train_faceSamples = self.train_faceSamples.reshape(len(self.train_faceIDs),224,224,1)
        self.validation_faceSamples = self.validation_faceSamples.reshape(len(self.validation_faceIDs),224,224,1)
    
    
        self.train_faceIDs = keras.utils.to_categorical(self.train_faceIDs, self.no_of_students)
        self.validation_faceIDs = keras.utils.to_categorical(self.validation_faceIDs, self.no_of_students)
    


    def datasetTrainer(self):
        self.getAllDirectoryPaths()
        self.getAllImagePaths()

        print("Training Started...")
        self.getImagesAndLabels()

        # Fitting the VGG Modal
        self.modal.fit(self.train_faceSamples, np.array(self.train_faceIDs), epochs=1, batch_size=4, verbose=2, validation_data=(self.validation_faceSamples, np.array(self.validation_faceIDs)))
        
        # Evaluating the VGG Modal for errors        
        score = self.modal.evaluate(self.validation_faceSamples, np.array(self.validation_faceIDs), verbose=0)

        # Printing the Loss
        print('Validation loss:', score)

        print("Training Successfully Completed!")
        
        
        # Serialize modal to YAML
        self.modal_yaml = self.modal.to_yaml()
        with open("TrainedEntities/modal.yaml", "w") as yaml_file:
            yaml_file.write(self.modal_yaml)
            
        # Serialize weights to HDF5
        self.modal.save_weights("TrainedEntities/modal.h5")
        print("Modal saved to disk")
 
      


DT = FaceDataSetTrainer()
DT.datasetTrainer()

