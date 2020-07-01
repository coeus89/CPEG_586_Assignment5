import math
import numpy as np
from Layer import *
from MyEnums import *
from sklearn.utils import shuffle

class Network(object):
    def __init__(self,X,Y,numLayers,dropout = 1.0, activationFunction = ActivationType.SIGMOID, lastLayerAF = ActivationType.SOFTMAX):
        self.X = X
        self.Y = Y
        self.numLayers = numLayers # array in the form [50, 10] for 2 layers of 50 and 10 neurons
        self.dropout = 1.0 - dropout #0.0 to 1.0 for 0 to 100 percent dropout. 
        self.activationFunction = activationFunction
        self.lastLayerAF = lastLayerAF
        self.Layers = []
        #self.LROptimizer = LROptimizerType
        #self.EnableBatchNorm = enableBatchNorm
        self.numOfLayers = len(numLayers)
        for i in range(self.numOfLayers):
            if (i == 0):
                layer = Layer(self.numLayers[i],self.X.shape[1],False,self.dropout,self.activationFunction)
            elif (i == (self.numOfLayers - 1)):
                layer = Layer(self.Y.shape[1],self.numLayers[i-1],True,self.dropout,self.lastLayerAF)
            else:
                layer = Layer(self.numLayers[i],self.numLayers[i-1], False,dropout,self.activationFunction)
            self.Layers.append(layer)
    
    def Evaluate(self,batch,doBatchNorm=False,batchType=BatchNormMode.TEST):
        self.Layers[0].Evaluate(batch,doBatchNorm,batchType)
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
            
    def Train(self,Epochs,LearningRate,lambda1,trainType,batchSize=1,doBatchNorm = False,lroptimization = LROptimizerType.NONE):

        for ep in range(0,Epochs):
            loss = 0
            itnum = 0
            self.X, self.Y = shuffle(self.X,self.Y,random_state=0)

            for batch_i in range(0, self.X.shape[0], batchSize):
                batch_x = self.X[batch_i:batch_i + batchSize]
                batch_y = self.Y[batch_i:batch_i + batchSize]
                LLa = self.Evaluate(batch_x,doBatchNorm,BatchNormMode.TRAIN) # Last Layer 'a' value
                
                if (self.lastLayerAF == ActivationType.SOFTMAX):
                    # Use cross entropy loss
                    loss += (-batch_y*np.log(LLa)).sum()
                else:
                    # use mean square loss
                    loss += (0.5 * (batch_y - LLa)**2)

                layerNumber = self.numOfLayers - 1
                while (layerNumber >= 0):
                    self.BackProp(batch_x,batch_y,layerNumber,batchSize,doBatchNorm,BatchNormMode.TRAIN)
                    self.CalcGradients(layerNumber,batch_x)
                    layerNumber -= 1

                itnum += 1
                self.UpdateGradsBiases(LearningRate,lambda1,batchSize,lroptimization,itnum,doBatchNorm)
            print("Epoch: " + str(ep) + ",   Loss: "+ str(loss))
    
    def CalcGradients(self,layerNumber,batch_x):
        if (layerNumber > 0):
            prevOut = self.Layers[layerNumber - 1].a
        else:
            prevOut = batch_x
        self.Layers[layerNumber].CalcGradients(prevOut)


    def UpdateGradsBiases(self, learningRate, lambda1, batchSize, LROptimization, itnum, doBatchNorm):
        # update weights and biases for all layers
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for ln in range(len(self.numLayers)):
            if (LROptimization == LROptimizerType.NONE):
                self.Layers[ln].UpdateWb(learningRate, batchSize,doBatchNorm)
            elif (LROptimization == LROptimizerType.ADAM):
                self.Layers[ln].CalcAdam(itnum,learningRate,batchSize,doBatchNorm,beta1,beta2,epsilon)

