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
        # Declaring number of classes
        self.no_of_students = 4
        
        # Setting initial path
        self.path = "DataSet"
        
        obVGG_CNN = VGG_CNN()        
        self.modal = obVGG_CNN.Modal()            


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
     
    def consequtiveShuffler(self,old_list,frequency):
        # Creating copy of given list
        new_list = old_list[:]
        # Creating new consequtive list
        k = 0
        for i in range(0, frequency):
            for j in range(0, self.no_of_students):
                new_list[k] = old_list[(frequency*j)+i]
                k += 1
                
        return new_list
    
        
    def getImagesAndLabels(self):            
        # Creating empty list for Training enitities
        self.train_faceSamples = []
        self.train_faceIDs = []
        # Creating empty list for Validation entities
        self.validation_faceSamples = []
        self.validation_faceIDs = []
        

        self.i = 0
        self.j = 0
        self.flag = 0
        self.c = 0
        
        # Looping through all the image paths to select all face images for training
        for imagePath in self.allImagePaths:
            # Reading images from path_list
            image = cv2.imread(imagePath) 
            # Converting Read image to Grayscale image
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resizing the Grayscale image
            image = cv2.resize(grayImage, (224, 224))
            
            # Extracting Roll Number
            rNo = (imagePath.split("_")[-2])
            
            
            if(self.flag == 0):
                self.i += 1            
                # Creating array of face samples found
                self.train_faceSamples.append(image)
                # Creating array of face IDs using 'roll number and serial number'
                self.train_faceIDs.append(int(rNo))            
                
            if(self.flag == 1):
                self.j += 1                
                # Creating array of face samples found
                self.validation_faceSamples.append(image)            
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
        
        
        # Shuffling the Training and Validation lists        
        self.train_faceSamples = self.consequtiveShuffler(self.train_faceSamples,100)
        self.train_faceIDs = self.consequtiveShuffler(self.train_faceIDs,100) 
        self.validation_faceSamples = self.consequtiveShuffler(self.validation_faceSamples,50)
        self.validation_faceIDs = self.consequtiveShuffler(self.validation_faceIDs,50)      
                
        
        # Converting list to n.array   
        self.train_faceSamples = np.array(self.train_faceSamples)
        self.validation_faceSamples = np.array(self.validation_faceSamples)        
        
        # Reshaping Image Samples
        self.train_faceSamples = self.train_faceSamples.reshape((self.train_faceSamples.shape[0]),224,224,1)
        self.validation_faceSamples = self.validation_faceSamples.reshape((self.validation_faceSamples.shape[0]),224,224,1)
    
        # Converting ImageIds to Categorical
        #self.train_faceIDs = np.array(self.train_faceIDs) / 255
        #self.validation_faceIDs = np.array(self.validation_faceIDs) / 255
        
        print(self.train_faceIDs)
        self.train_faceIDs = keras.utils.to_categorical(self.train_faceIDs, self.no_of_students)
        self.validation_faceIDs = keras.utils.to_categorical(self.validation_faceIDs, self.no_of_students)  
        print(self.train_faceIDs)


    def datasetTrainer(self):
        self.getAllDirectoryPaths()
        self.getAllImagePaths()
        self.getImagesAndLabels()
    
        print("Training Started...")
        # Fitting the VGG Modal
        self.modal.fit(self.train_faceSamples, np.array(self.train_faceIDs), epochs=1, batch_size=5, verbose=2, validation_data=(self.validation_faceSamples, np.array(self.validation_faceIDs)))
        
        # Evaluating the VGG Modal for errors        
        score = self.modal.evaluate(self.validation_faceSamples, np.array(self.validation_faceIDs), verbose=0)
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

