from FeatureMap import FeatureMap
import math
import numpy as np
from CNNEnums import *
from Pooling import *
from sklearn.utils import shuffle
from scipy.signal import convolve2d

class CNNLayer(object):
    def __init__(self,numFeatureMaps,numPrevLayerFeatureMaps,inputSize,kernelSize,poolingType,activationType,batchSize=1):
        self.batchSize = batchSize
        self.ConvolSums = np.empty((batchSize,numFeatureMaps),dtype=object)
        self.kernelSize = kernelSize
        self.numFeatureMaps = numFeatureMaps
        self.numPrevLayerFeatureMaps = numPrevLayerFeatureMaps
        self.ConvolResults = np.empty((batchSize,numPrevLayerFeatureMaps,numFeatureMaps),dtype=object)
        convOutputSize = inputSize - kernelSize + 1
        for i in range(0,batchSize):
            for j in range(0,numFeatureMaps):
                self.ConvolSums[i,j] = np.zeros((convOutputSize,convOutputSize))
        initRange = (numFeatureMaps/((numPrevLayerFeatureMaps + numFeatureMaps)*(kernelSize**2)))**0.5
        # self.Kernels = numPrevLayerFeatureMaps,numFeatureMaps,kernelSize
        # self.KernelGrads = np.zeros((numPrevLayerFeatureMaps,numFeatureMaps))
        self.Kernels = np.empty((numPrevLayerFeatureMaps,numFeatureMaps),dtype=object)
        self.KernelGrads = np.empty((numPrevLayerFeatureMaps,numFeatureMaps),dtype=object)
        # self.InitMatrix2DArray(self.Kernels,numPrevLayerFeatureMaps,numFeatureMaps,kernelSize)
        # self.InitMatrix2DArray(self.KernelGrads,numPrevLayerFeatureMaps,numFeatureMaps,kernelSize)
        self.InitializeKernels(initRange)
        self.featureMapList = []
        for i in range(0,numFeatureMaps):
            self.featureMapList.append(FeatureMap(convOutputSize,poolingType,activationType,batchSize))

    def Evaluate(self,PrevLayerOutputList,batchIndex):
        # inputs are from the previous layer (unless first layer)
        # convolve inputs with Kernels
        for i in range(0,self.numPrevLayerFeatureMaps):
            for j in range(0,self.numFeatureMaps):
                currentKernel = self.Kernels[i,j]
                self.ConvolResults[batchIndex,i,j] = convolve2d(PrevLayerOutputList[i],currentKernel,mode='valid',boundary='symm')
        # Add Convolution Results
        for q in range(0,len(self.featureMapList)):
            self.ConvolSums[batchIndex,q] = np.zeros((self.convOutputSize,self.convOutputSize))
            for p in range(0,len(PrevLayerOutputList)):
                self.ConvolSums[batchIndex,q] += self.ConvolResults[batchIndex,p,q]
        # Evaluate each feature map
        for i in range(0, len(self.featureMapList)):
            self.featureMapList[i].Evaluate(self.ConvolSums[batchIndex,i],batchIndex)

    
    def InitMatrix2DArray(self, Mat, dim1, dim2, matrixSize):
        Mat = np.empty((dim1,dim2))
        for i in range(0,dim1):
            for j in range (0,dim2):
                Mat[i,j] = np.zeros((matrixSize,matrixSize))
    
    def InitializeKernels(self,initRange):
        for i in range (0,self.Kernels.shape[0]):
            for j in range (0,self.Kernels.shape[1]):
                self.Kernels[i,j] = np.random.uniform(low=-initRange,high=initRange,size=(self.kernelSize,self.kernelSize))

