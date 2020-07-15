import math
import numpy as np
from numpy.core.defchararray import array
from Layer import *
from MyEnums import *
from CNNLayer import *
from CNNEnums import *
from sklearn.utils import shuffle

class Network(object):
    def __init__(self,X,Y,numCNNLayers,kernelSize,poolingType,numLayers,dropout = 1.0, activationFunction = ActivationType.SIGMOID, lastLayerAF = ActivationType.SOFTMAX,batchSize = 1):
        self.X = X
        self.Y = Y
        self.numCNNLayers = numCNNLayers # array in the form [4, 6] for 2 layers of 4 and 6 feature maps respectively
        self.numLayers = numLayers # array in the form [50, 10] for 2 layers of 50 and 10 neurons respectively
        self.dropout = 1.0 - dropout # 0.0 to 1.0 for 0 to 100 percent dropout. (0 to 100 percent zeros for the self.dropout variable)
        self.activationFunction = activationFunction
        self.lastLayerAF = lastLayerAF
        self.Layers = []
        self.myCNNLayers = []
        self.numOfLayers = len(numLayers)
        self.batchSize = batchSize
        self.NNInputSize = 0
        self.Flatten = np.zeros((batchSize), dtype=object)

        # Initialize the CNN Layers
        for j in range(0,len(numCNNLayers)):
            if (j == 0):
                inputSize = self.X.shape[1]
                self.myCNNLayers.append(CNNLayer(self.numCNNLayers[j],1,inputSize,kernelSize,poolingType,activationFunction,batchSize))
            else:
                inputSize = self.X.shape[1]
                for k in range(1, j+1):
                    inputSize = (int)((inputSize - kernelSize + 1)/2) # Make sure the 2nd layer input is 12
                self.myCNNLayers.append(CNNLayer(self.numCNNLayers[j],self.numCNNLayers[j-1],inputSize,kernelSize,poolingType,activationFunction,batchSize))

        # Initialize the Normal Layers
        for i in range(self.numOfLayers):
            if (i == 0):
                # First NN layer coming from a CNN
                prevFeatureMapSize = (self.myCNNLayers[len(self.myCNNLayers) - 1].poolOutputSize)**2
                flattenSize = (prevFeatureMapSize) * self.myCNNLayers[len(self.myCNNLayers) - 1].numFeatureMaps
                # make the layer
                layer = Layer(self.numLayers[i],flattenSize,False,self.dropout,self.activationFunction)
            elif (i == (self.numOfLayers - 1)):
                layer = Layer(self.Y.shape[1],self.numLayers[i-1],True,self.dropout,self.lastLayerAF)
            else:
                layer = Layer(self.numLayers[i],self.numLayers[i-1], False,dropout,self.activationFunction)
            self.Layers.append(layer)
        
        # Find the flattened expected size for the normal NN input
        self.NNInputSize = self.X.shape[1]
        for k in range(1, len(numCNNLayers)+1):
            self.NNInputSize = (self.NNInputSize - kernelSize + 1)/2 # Make sure the 3nd iteration is 4
        self.NNInputSize = (self.NNInputSize**2) * numCNNLayers[len(numCNNLayers)-1] #for 2 layers it should be 4 * 4 * 6 if six feature maps
    
    def Evaluate(self,batch,doBatchNorm=False,batchType=BatchNormMode.TEST):
        # Evaluate CNN Layers
        # Note: batch is in format [batchSize,prevFeatureMaps,FeatureMapWidth,FeatureMapHeight]
        # prevFeatureMaps may just be the single image. so will be 1 on first layer.
        for j in range(0, len(self.numCNNLayers)): # select Layer
            PrevOut = None
            if (j == 0):
                PrevOut = batch 
            else:
                PrevOut = np.empty((self.batchSize, self.myCNNLayers[j-1].numFeatureMaps),dtype=object)
                for k in range(0,len(self.myCNNLayers[j-1].featureMapList)): # select Feature Map
                    BatchFeatureMapOut = self.myCNNLayers[j - 1].featureMapList[k].OutputPool
                    for m in range(0, len(BatchFeatureMapOut)): # This puts the prevOut in the format [batch,featureMapOutput] for a batch of 5 and 4 feature maps it will be 5x4x12x12
                        PrevOut[m,k] = BatchFeatureMapOut[m]
                
            # For each item in batch evaluate
            for n in range(0,self.batchSize):
                batchIndex = n
                PreviousOutput = PrevOut[batchIndex]
                self.myCNNLayers[j].Evaluate(PreviousOutput,batchIndex)
        
        
        # Flatten the output
        # get the Feature Map Output Pools into the format [batch,OutputVectors] which should be a 5 x 6 for a batch size of 5 and a feature map size of 6.
        featureMapSize = (self.myCNNLayers[len(self.myCNNLayers) - 1].poolOutputSize)**2
        flattenSize = (featureMapSize) * self.myCNNLayers[len(self.myCNNLayers) - 1].numFeatureMaps
        self.Flatten = np.zeros((self.batchSize,flattenSize))
        for bIndex in range(0,self.batchSize):
            flatFM = []
            for fmIndex in range(0,self.myCNNLayers[len(self.myCNNLayers) - 1].numFeatureMaps):
                fm = self.myCNNLayers[len(self.myCNNLayers) - 1].featureMapList[fmIndex].OutputPool[bIndex]
                flatFM = np.append(flatFM,fm.reshape(featureMapSize))
                test = ""
            self.Flatten[bIndex] = flatFM


        # Normal NN Layers
        currBatch = self.Flatten
        self.Layers[0].Evaluate(currBatch,doBatchNorm,batchType)
        for i in range(1,self.numOfLayers):
            self.Layers[i].Evaluate(self.Layers[i-1].a,doBatchNorm,batchType)
        return self.Layers[self.numOfLayers - 1].a

    def BackProp(self,batch_x,batch_y,layerNumber,batchSize,doBatchNorm,batchType):
        layer = self.Layers[layerNumber]
        if (layerNumber == (self.numOfLayers - 1)): #last layer
            if (layer.activationType == ActivationType.SOFTMAX):
                layer.SoftMaxDeltaLL(batch_y, batchSize, doBatchNorm, batchType) #last layer
            else:
                layer.CalcDeltaLL(batch_y, batchSize, doBatchNorm, batchType) #last layer
        else:
            layer.CalcDelta(self.Layers[layerNumber + 1].deltabn, self.Layers[layerNumber + 1].w, batchSize, doBatchNorm, batchType)
            
    def Train(self,Epochs,LearningRate,doBatchNorm = False,lroptimization = LROptimizerType.NONE):

        for ep in range(0,Epochs):
            loss = 0
            itnum = 0
            self.X, self.Y = shuffle(self.X, self.Y, random_state=0)
            # shuffle(self.X, self.Y, random_state=0)

            # Evaluate CNN and Normal Layers
            for batch_i in range(0, self.X.shape[0], self.batchSize):
                batch_x = self.X[batch_i:batch_i + self.batchSize]
                batch_y = self.Y[batch_i:batch_i + self.batchSize]
                # Need to add the 1 into the array so that the CNN likes the shape
                batch_x = batch_x.reshape(batch_x.shape[0],1,batch_x.shape[1],batch_x.shape[2])
                #batch_y = batch_y.reshape(batch_y.shape[0],1,batch_y.shape[1],batch_y.shape[2])
                LLa = self.Evaluate(batch_x,doBatchNorm,BatchNormMode.TRAIN) # Last Layer 'a' value
                
                if (self.lastLayerAF == ActivationType.SOFTMAX):
                    # Use cross entropy loss
                    loss += (-batch_y*np.log(LLa)).sum()
                else:
                    # use mean square loss
                    loss += (0.5 * (batch_y - LLa)**2)

                layerNumber = self.numOfLayers - 1
                while (layerNumber >= 0):
                    self.BackProp(batch_x,batch_y,layerNumber,self.batchSize,doBatchNorm,BatchNormMode.TRAIN)
                    self.CalcGradients(layerNumber,batch_x)
                    layerNumber -= 1

                itnum += 1
                self.UpdateGradsBiases(LearningRate,self.batchSize,lroptimization,itnum,doBatchNorm)
            print("Epoch: " + str(ep) + ",   Loss: "+ str(loss))
    
    def CalcGradients(self,layerNumber,batch_x):
        if (layerNumber > 0):
            prevOut = self.Layers[layerNumber - 1].a
        else:
            prevOut = batch_x
        self.Layers[layerNumber].CalcGradients(prevOut)


    def UpdateGradsBiases(self, learningRate, batchSize, LROptimization, itnum, doBatchNorm):
        # update weights and biases for all layers
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for ln in range(len(self.numLayers)):
            if (LROptimization == LROptimizerType.NONE):
                self.Layers[ln].UpdateWb(learningRate, batchSize,doBatchNorm)
            elif (LROptimization == LROptimizerType.ADAM):
                self.Layers[ln].CalcAdam(itnum,learningRate,batchSize,doBatchNorm,beta1,beta2,epsilon)

