import os
import sys
import cv2
import math
import numpy as np
from Network import *
from MyEnums import *


def main():
    trainingImages = len(os.listdir("C:\\Data\\Training1000"))
    testImages = len(os.listdir("C:\\Data\\Test10000"))
    train = np.empty((trainingImages,28,28),dtype=np.float)
    trainY = np.zeros((trainingImages,10))
    test = np.empty((testImages,28,28),dtype=np.float)
    testY = np.zeros((testImages,10))
    
    #load images
    i = 0
    for filename in os.listdir("C:\\Data\\Training1000"):
        y = int(filename[0])
        trainY[i,y] = 1.0
        train[i] = cv2.imread('C:\\Data\\Training1000\\{0}'.format(filename),0)/255.0 #for color use 1
        i += 1

    j = 0
    for filename in os.listdir("C:\\Data\\Test10000"):
        y = int(filename[0])
        testY[j,y] = 1.0
        test[j] = cv2.imread('C:\\Data\\Test10000\\{0}'.format(filename),0)/255.0 
        j += 1

    trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2])
    testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2])

    numLayers = [100,10]
    dropOut = 1.0 #20% dropout
    LROptType = LROptimizerType.ADAM

    myNetwork = Network(trainX,trainY,numLayers,dropOut,ActivationType.SIGMOID,ActivationType.SOFTMAX,LROptimizerType.ADAM,BatchNormEnabled.Enabled)

    epochs = 10
    learningRate = 0.01  
    lambda1 = 0. #don't use. not sure why it's there.
    trainType = GradDescType.MiniBatch
    batchSize = 5
    doBatchNorm = True
    lropt = LROptimizerType.NONE
    myNetwork.Train(epochs,learningRate,lambda1,trainType,batchSize,doBatchNorm,lropt)

    print("Finished Training. \nTesting Begins")

    accuracyCount = 0
    for i in range(testY.shape[0]):
        a2 = myNetwork.Evaluate(testX[i],doBatchNorm,BatchNormMode.TEST)
        maxindex = a2.argmax(axis = 0)
        if (testY[i,maxindex] == 1):
            accuracyCount = accuracyCount + 1
    print("Accuracy count = " + str(accuracyCount/testY.shape[0]*100) + '%')

if __name__ == "__main__":
    sys.exit(int(main() or 0))