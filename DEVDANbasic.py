import numpy as np
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import pdb
from collections import deque
import random
from scipy.stats.distributions import chi2
import pandas as pd
import warnings
from utilsDEVDAN import meanStdCalculator, probitFunc, deleteRowTensor, deleteColTensor, maskingNoise
# warnings.filterwarnings("ignore", category=RuntimeWarning)

class hiddenLayerBasicNet(nn.Module):
    def __init__(self, no_input, no_hidden):
        super(hiddenLayerBasicNet, self).__init__()
        
        # hidden layer
        self.linear = nn.Linear(no_input, no_hidden,  bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        
        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.Sigmoid()
        self.activationh= nn.ReLU(inplace=True)
        
        # decoder
        self.biasDecoder = nn.Parameter(torch.zeros(no_input))
        
    def forward(self, x):
        x = self.linear(x)
        x = self.activationh(x)                                        # encoded output

        with torch.no_grad():
            h2 = (x.clone().detach())**2
            r2 = F.linear(h2, self.linear.weight.t()) + self.biasDecoder
            r2 = self.activation(r2)                                  # reconstructed input for variance calculation

        # decoder
        r = F.linear(x, self.linear.weight.t()) + self.biasDecoder
        r = self.activation(r)                                        # reconstructed input
        
        return x, r, r2

class outputLayerBasicNet(nn.Module):
    def __init__(self, no_hidden, classes):
        super(outputLayerBasicNet, self).__init__()
        
        # softmax layer
        self.linearOutput = nn.Linear(no_hidden, classes,  bias=True)
        nn.init.xavier_uniform_(self.linearOutput.weight)
        self.linearOutput.bias.data.zero_()
        
    def forward(self, x):
        x = self.linearOutput(x)
        
        return x

class hiddenLayer():
    def __init__(self, no_input, no_hidden):
        self.network = hiddenLayerBasicNet(no_input,no_hidden)
        self.netUpdateProperties()

    def netUpdateProperties(self):
        self.nNetInput   = self.network.linear.in_features
        self.nNodes      = self.network.linear.out_features
        self.nParameters = (self.network.linear.in_features*self.network.linear.out_features +
                            len(self.network.linear.bias.data) + len(self.network.biasDecoder))

    def getNetProperties(self):
        print(self.network)
        print('No. of inputs :',self.nNetInput)
        print('No. of nodes :',self.nNodes)
        print('No. of parameters :',self.nParameters)

    def getNetParameters(self):
        print('Input weight: \n', self.network.linear.weight)
        print('Input bias: \n', self.network.linear.bias)
        print('Output decoder bias: \n', self.network.biasDecoder)

    def nodeGrowing(self, init = None, nNewNode = 1):
        nNewNodeCurr = self.nNodes + nNewNode
        # grow node
        # newWeight, newOutputWeight,_     = generateWeightXavInit(self.nNetInput,nNewNodeCurr,self.nOutputs,nNewNode)
        if init is None and nNewNode >= 1:
            newWeight                    = nn.init.xavier_uniform_(torch.empty(nNewNode, self.nNetInput))
        elif init is not None and nNewNode == 1:
            newWeight                    = init

        self.network.linear.weight.data  = torch.cat((self.network.linear.weight.data,newWeight),0)            # grow input weights
        self.network.linear.bias.data    = torch.cat((self.network.linear.bias.data,torch.zeros(nNewNode)),0)  # grow input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad
        
        self.netUpdateProperties()

    def nodePruning(self,pruneIdx,nPrunedNode = 1):
        nNewNodeCurr = self.nNodes - nPrunedNode  # prune a node
        
        # prune node for current layer, output
        self.network.linear.weight.data  = deleteRowTensor(self.network.linear.weight.data,
                                                           pruneIdx)  # prune input weights
        self.network.linear.bias.data    = deleteRowTensor(self.network.linear.bias.data,
                                                           pruneIdx)  # prune input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad
        
        self.netUpdateProperties()

    def inputGrowing(self,nNewInput = 1):
        nNewInputCurr = self.nNetInput + nNewInput

        # grow input weight
        # _,_,newWeightNext = generateWeightXavInit(nNewInputCurr,self.nNodes,self.nOutputs,nNewInput)
        newWeightNext                   = nn.init.xavier_uniform_(torch.empty(self.nNodes, nNewInput))
        self.network.linear.weight.data = torch.cat((self.network.linear.weight.data,newWeightNext),1)
        self.network.biasDecoder.data   = torch.cat((self.network.biasDecoder.data,torch.zeros(nNewInput)),0)
        
        del self.network.linear.weight.grad
        del self.network.biasDecoder.grad

        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()
        
    def inputPruning(self,pruneIdx,nPrunedNode = 1):
        nNewInputCurr = self.nNetInput - nPrunedNode

        # prune input weight of next layer
        self.network.linear.weight.data = deleteColTensor(self.network.linear.weight.data,pruneIdx)
        self.network.biasDecoder.data   = deleteRowTensor(self.network.biasDecoder.data,pruneIdx)
        
        del self.network.linear.weight.grad
        del self.network.biasDecoder.grad

        # update input features
        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()


class outputLayer():
    def __init__(self, no_hidden, classes):
        self.network = outputLayerBasicNet(no_hidden,classes)
        self.netUpdateProperties()

    def netUpdateProperties(self):
        self.nNetInput   = self.network.linearOutput.in_features
        self.nOutputs    = self.network.linearOutput.out_features
        self.nParameters = (self.network.linearOutput.in_features*self.network.linearOutput.out_features +
                            len(self.network.linearOutput.bias.data))

    def getNetProperties(self):
        print(self.network)
        print('No. of inputs :',self.nNetInput)
        print('No. of output :',self.nOutputs)
        print('No. of parameters :',self.nParameters)

    def getNetParameters(self):
        print('Output weight: \n', self.network.linearOutput.weight)
        print('Output bias: \n', self.network.linearOutput.bias)

    def inputGrowing(self,nNewInput = 1):
        nNewInputCurr = self.nNetInput + nNewInput

        # grow input weight
        # _,_,newWeightNext = generateWeightXavInit(nNewInputCurr,self.nNodes,self.nOutputs,nNewInput)
        newWeightNext = nn.init.xavier_uniform_(torch.empty(self.nOutputs, nNewInput))
        self.network.linearOutput.weight.data = torch.cat((self.network.linearOutput.weight.data,newWeightNext),1)
        del self.network.linearOutput.weight.grad

        self.network.linearOutput.in_features = nNewInputCurr
        self.netUpdateProperties()
        
    def inputPruning(self,pruneIdx,nPrunedNode = 1):
        nNewInputCurr = self.nNetInput - nPrunedNode

        # prune input weight of next layer
        self.network.linearOutput.weight.data = deleteColTensor(self.network.linearOutput.weight.data,pruneIdx)
        del self.network.linearOutput.weight.grad

        # update input features
        self.network.linearOutput.in_features = nNewInputCurr
        self.netUpdateProperties()


class DEVDAN():
    def __init__(self, nInput, nOutput, LR = 0.02):
        # initial network
        self.net = [hiddenLayer(nInput,nOutput),outputLayer(nOutput,nOutput)]

        # network significance
        self.averageBias    = meanStdCalculator()
        self.averageVar     = meanStdCalculator()
        self.averageInput   = meanStdCalculator()
        # self.averageFeature = meanStdCalculator()

        self.averageBiasGen = [meanStdCalculator()]
        self.averageVarGen  = [meanStdCalculator()]

        # hyper parameters
        self.lr           = LR
        self.lrGen        = LR/10
        self.criterion    = nn.CrossEntropyLoss()
        self.criterionGen = nn.MSELoss()            # loss for generative loss = 0.5*criterion(scores,minibatch_label)
        
        # Evolving
        self.growNode   = False
        self.pruneNode  = False
        self.hnGrowing  = True
        self.hnPruning  = True

        # properties
        self.nHiddenLayer = 1
        self.nHiddenNode  = nOutput
        self.nOutputs     = nOutput
        self.winLayerIdx  = 0
        
    def updateNetProperties(self):
        self.nHiddenLayer = len(self.net) - 1
        nHiddenNode = 0
        for iLayer in range(0,len(self.net)-1):
            nHiddenNode += self.net[iLayer].nNodes
        self.nHiddenNode = nHiddenNode

    def getNetProperties(self):
        for iLayer,nett in enumerate(self.net):
            print('\n',iLayer + 1,'-th layer')
            nett.getNetProperties()
        
    # ============================= Evolving mechanism =============================
    def hiddenNodeGrowing(self,layerIdx = -2):
        if layerIdx <= (len(self.net)-1):
            copyHiddenLayer = copy.deepcopy(self.net[layerIdx])
            copyHiddenLayer.nodeGrowing()
            self.net[layerIdx] = copy.deepcopy(copyHiddenLayer)

            if layerIdx == -2:
                # grow input for classifier
                copyOutputLayer = copy.deepcopy(self.net[layerIdx+1])
                copyOutputLayer.inputGrowing()
                self.net[-1] = copy.deepcopy(copyOutputLayer)
            else:
                # grow input for for next layer
                copyNextNet = copy.deepcopy(self.net[layerIdx+1])
                copyNextNet.inputGrowing()
                self.net[layerIdx+1] = copy.deepcopy(copyNextNet)
        
            # print('+++ GROW a hidden NODE +++')
            self.updateNetProperties()
        else:
            raise IndexError

    def hiddenNodeGrowingWithInit(self,init,layerIdx = -2):
        if layerIdx <= (len(self.net)-1):
            copyHiddenLayer = copy.deepcopy(self.net[layerIdx])
            copyHiddenLayer.nodeGrowing(init = init)
            self.net[layerIdx] = copy.deepcopy(copyHiddenLayer)

            if layerIdx == -2:
                # grow input for classifier
                copyOutputLayer = copy.deepcopy(self.net[layerIdx+1])
                copyOutputLayer.inputGrowing()
                self.net[-1] = copy.deepcopy(copyOutputLayer)
            else:
                # grow input for next layer
                copyNextNet = copy.deepcopy(self.net[layerIdx+1])
                copyNextNet.inputGrowing()
                self.net[layerIdx+1] = copy.deepcopy(copyNextNet)
        
            # print('+++ GROW a hidden NODE +++')
            self.updateNetProperties()
        else:
            raise IndexError
        
    def hiddenNodePruning(self,layerIdx = -2):
        if layerIdx <= (len(self.net)-1):
            copyHiddenLayer = copy.deepcopy(self.net[layerIdx])
            copyHiddenLayer.nodePruning(self.leastSignificantNode)
            self.net[layerIdx] = copy.deepcopy(copyHiddenLayer)

            if layerIdx == -2:
                # prune input for classifier
                copyOutputLayer = copy.deepcopy(self.net[layerIdx+1])
                copyOutputLayer.inputPruning(self.leastSignificantNode)
                self.net[-1] = copy.deepcopy(copyOutputLayer)
            else:
                # prune input for next layer
                copyNextNet = copy.deepcopy(self.net[layerIdx+1])
                copyNextNet.inputPruning(self.leastSignificantNode)
                self.net[layerIdx+1] = copy.deepcopy(copyNextNet)
        
            # print('+++ GROW a hidden NODE +++')
            self.updateNetProperties()
        else:
            raise IndexError

    # ============================= forward pass =============================
    def feedforwardTest(self,x,device = torch.device('cpu')):
        # feedforward to all layers
        with torch.no_grad():
            tempVar = x.to(device)
            tempVar = tempVar.type(torch.float)
            
            for iLayer in range(len(self.net)):
                currnet = self.net[iLayer].network
                obj     = currnet.eval()
                obj     = obj.to(device)
                if iLayer == len(self.net)-1:
                    # get output
                    tempVar     = obj(tempVar)
                else:
                    tempVar,_,_ = obj(tempVar)

            self.scoresTest            = tempVar
            self.multiClassProbability = F.softmax(tempVar.data,dim=1)
            self.predictedLabelProbability, self.predictedLabel = torch.max(self.multiClassProbability, 1)

    def feedforwardTrainDiscriminative(self,x,device = torch.device('cpu')):
        # feedforward to the winning layer
        tempVar = x.to(device)
        tempVar = tempVar.type(torch.float)
        
        # feedforward to all layers
        for iLayer in range(len(self.net)):
            currnet = self.net[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(device)
            if iLayer == len(self.net)-1:
                # get output
                tempVar     = obj(tempVar)
            else:
                tempVar,_,_ = obj(tempVar)

        self.scoresTrain = tempVar

    def feedforwardBiasVarDiscriminative(self,x,label_oneHot,device = torch.device('cpu')):
        # label_oneHot is label in one hot vector form, float, already put in device
        with torch.no_grad():
            tempVar = x.to(device)
            tempVar = tempVar.type(torch.float)
            
            hiddenNodeSignificance = []

            for iLayer in range(len(self.net)):
                currnet           = self.net[iLayer].network
                obj               = currnet.eval()
                obj               = obj.to(device)
                
                if iLayer == 0:
                    tempVar,_,_ = obj(tempVar)
                    tempVar2    = (tempVar.clone().detach())**2

                    # node significance
                    hiddenNodeSignificance.append(tempVar.clone().detach().squeeze(dim=0).tolist())

                else:
                    if iLayer == len(self.net)-1:
                        # get output
                        tempVar  = obj(tempVar)
                        tempVar2 = obj(tempVar2)
                    else:
                        tempVar,_,_  = obj(tempVar)
                        tempVar2,_,_ = obj(tempVar2)

                    if iLayer < len(self.net) - 1:
                        # node significance 
                        hiddenNodeSignificance.append(tempVar.clone().detach().squeeze(dim=0).tolist())
                    
            # bias variance
            tempY    = F.softmax(tempVar,dim=1)                 # y
            tempY2   = F.softmax(tempVar2,dim=1)                # y2
            bias     = torch.norm((tempY - label_oneHot)**2)    # bias
            variance = torch.norm(tempY2 - tempY**2)            # variance

            self.bias     = bias.item()
            self.variance = variance.item()
            self.hiddenNodeSignificance = hiddenNodeSignificance

    def feedforwardTrainGenerative(self, x, layerIdx=0, device = torch.device('cpu')):
        # feed forward only on winning layer
        minibatch_data = x.to(device)
        minibatch_data = minibatch_data.type(torch.float)

        currnet = self.net[layerIdx].network
        obj     = currnet.train()
        obj     = obj.to(device)
        _,self.scoresGenerativeTrain,_   = obj(x)

    def feedforwardBiasVarGenerative(self,x,layerIdx=0,device = torch.device('cpu')):
        with torch.no_grad():
            x = x.to(device)
            x = x.type(torch.float)
            
            hiddenNodeSignificance = []

            currnet = self.net[layerIdx].network
            obj = currnet.eval()
            obj = obj.to(device)

            _,reconstructedInput,_   = obj(x)
            _,_,reconstructedInput2  = obj(x)
            hiddenRepresentation,_,_ = obj(x)

            self.HS = hiddenRepresentation.clone().detach()
            hiddenNodeSignificance.append(hiddenRepresentation.clone().detach().squeeze(dim=0).tolist())

            bias     = torch.mean((reconstructedInput - x)**2)
            variance = torch.mean(reconstructedInput2 - x**2)

            self.bias     = bias.item()
            self.variance = variance.item()
            self.hiddenNodeSignificance = hiddenNodeSignificance

    # ============================= Network Evaluation =============================
    def updateBiasVariance(self):
        # calculate mean of bias
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.averageBias.updateMeanStd(self.bias)
        if self.averageBias.count < 1 or self.growNode:
            self.averageBias.resetMinMeanStd()
        else:
            self.averageBias.updateMeanStdMin()
        
        # calculate mean of variance
        self.averageVar.updateMeanStd(self.variance)
        if self.averageVar.count < 20 or self.pruneNode:
            self.averageVar.resetMinMeanStd()
        else:
            self.averageVar.updateMeanStdMin()

    def updateBiasVarianceGen(self,layerIdx):
        # calculate mean of bias
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.averageBiasGen[layerIdx].updateMeanStd(self.bias)
        if self.averageBiasGen[layerIdx].count < 1 or self.growNode:
            self.averageBiasGen[layerIdx].resetMinMeanStd()
        else:
            self.averageBiasGen[layerIdx].updateMeanStdMin()
        
        # calculate mean of variance
        self.averageVarGen[layerIdx].updateMeanStd(self.variance)
        if self.averageVarGen[layerIdx].count < 20 or self.pruneNode:
            self.averageVarGen[layerIdx].resetMinMeanStd()
        else:
            self.averageVarGen[layerIdx].updateMeanStdMin()

    def growNodeIdentification(self):
        dynamicKsigmaGrow = 1.3*np.exp(-self.bias) + 0.7
        growCondition1    = (self.averageBias.minMean + 
                             dynamicKsigmaGrow*self.averageBias.minStd)
        growCondition2    = self.averageBias.mean + self.averageBias.std

        if growCondition2 > growCondition1 and self.averageBias.count >= 1:
            self.growNode = True
        else:
            self.growNode = False
    
    def pruneNodeIdentification(self, layerIdx = -2):
        dynamicKsigmaPrune = 1.3*np.exp(-self.variance) + 0.7
        pruneCondition1    = (self.averageVar.minMean + 
                              2*dynamicKsigmaPrune*self.averageVar.minStd)
        pruneCondition2    = self.averageVar.mean + self.averageVar.std
        
        if (pruneCondition2 > pruneCondition1 and not self.growNode and 
            self.averageVar.count >= 20 and
            self.net[layerIdx].nNodes > self.nOutputs):
            self.pruneNode = True
            self.findLeastSignificantNode(layerIdx)
        else:
            self.pruneNode = False

    def growNodeIdentificationGen(self, layerIdx):
        dynamicKsigmaGrow = 1.3*np.exp(-self.bias) + 0.7
        growCondition1    = (self.averageBiasGen[layerIdx].minMean + 
                             dynamicKsigmaGrow*self.averageBiasGen[layerIdx].minStd)
        growCondition2    = self.averageBiasGen[layerIdx].mean + self.averageBiasGen[layerIdx].std

        if growCondition2 > growCondition1 and self.averageBiasGen[layerIdx].count >= 1:
            # print('growCondition2',growCondition2)
            # print('growCondition1',growCondition1)
            
            self.growNode = True
        else:
            self.growNode = False
    
    def pruneNodeIdentificationGen(self, layerIdx):
        dynamicKsigmaPrune = 1.3*np.exp(-self.variance) + 0.7
        pruneCondition1    = (self.averageVarGen[layerIdx].minMean + 
                              2*dynamicKsigmaPrune*self.averageVarGen[layerIdx].minStd)
        pruneCondition2    = self.averageVarGen[layerIdx].mean + self.averageVarGen[layerIdx].std
        
        if (pruneCondition2 > pruneCondition1 and not self.growNode and 
            self.averageVarGen[layerIdx].count >= 20 and
            self.net[layerIdx].nNodes > self.nOutputs):
            self.pruneNode = True
            self.findLeastSignificantNode(0)
        else:
            self.pruneNode = False

    def findLeastSignificantNode(self,layerIdx = -1):
        # find the least significant node in the winning layer
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.leastSignificantNode = torch.argmin(torch.abs(torch.tensor(self.hiddenNodeSignificance[layerIdx]))).tolist()

    # ============================= Training ============================= 
    # def generativeTraining(self, device = torch.device('cpu'), batchSize = 1, epoch = 1, generative = True):
    #     # shuffle the data
    #     batchDataGen = self.batchData
    #     nData        = batchDataGen.shape[0]
    #     firstEvolution = False

    #     for iHiddenLayer in range(self.nHiddenLayer):
    #         self.growNode = False
    #         self.pruneNode = False
        
    #         for iEpoch in range(0,epoch):

    #             shuffled_indices = torch.randperm(nData)
                
    #             # masked input
    #             maskedX = maskingNoise(batchDataGen.clone().detach())   # some of the input feature

    #             for iData in range(0,nData,batchSize):
    #                 # load data
    #                 indices          = shuffled_indices[iData:iData+batchSize]

    #                 minibatch_xTrain = maskedX[indices]
    #                 minibatch_xTrain = minibatch_xTrain.to(device)

    #                 minibatch_x_noNoise  = batchDataGen[indices]
    #                 minibatch_x_noNoise  = minibatch_x_noNoise.to(device)

    #                 if iEpoch == 0:
    #                     if batchSize > 1:
    #                         minibatch_x_noNoise_mean = torch.mean(minibatch_x_noNoise,dim=0).unsqueeze(dim=0)
    #                     else:
    #                         minibatch_x_noNoise_mean = minibatch_x_noNoise

    #                     if iHiddenLayer == 0:
    #                         # calculate mean of input, only done in the first hidden layer
    #                         self.averageInput.updateMeanStd(minibatch_x_noNoise_mean)

    #                         # get bias and variance
    #                         # outProbit = probitFunc(self.averageInput.mean,self.averageInput.std)
    #                         # self.feedforwardBiasVarGenerative(outProbit)  # for sigmoid activation function

    #                         self.feedforwardBiasVarGenerative(self.averageInput.mean)  # for ReLU activation function

    #                     if iHiddenLayer > 0:
    #                         # get bias and variance
    #                         outProbit = probitFunc(self.averageInput.mean,self.averageInput.std)
    #                         self.feedforwardBiasVarGenerative(outProbit)         # for sigmoid activation function

    #                     # update bias variance
    #                     self.updateBiasVarianceGen(iHiddenLayer)

    #                     # growing
    #                     if generative:
    #                         self.growNodeIdentificationGen(iHiddenLayer)
    #                         if self.growNode and self.hnGrowing:
    #                             firstEvolution = True
    #                             # pdb.set_trace()
    #                             # print('grow node generative')

    #                             # initiate the new weight
    #                             with torch.no_grad():
    #                                 self.feedforwardTrainGenerative(minibatch_xTrain)
    #                                 x_hat = self.scoresGenerativeTrain

    #                             WN = -(minibatch_x_noNoise - x_hat)

    #                             # self.hiddenNodeGrowing(iHiddenLayer)
    #                             # pdb.set_trace()
    #                             self.hiddenNodeGrowingWithInit(init = WN, layerIdx = iHiddenLayer)
    #                             # if iHiddenLayer == 0:
    #                             #     self.averageFeature = meanStdCalculator()

    #                         # pruning
    #                         if not self.growNode and self.hnPruning:
    #                             self.pruneNodeIdentificationGen(iHiddenLayer)
                                
    #                             if self.pruneNode:
    #                                 firstEvolution = True
    #                                 # pdb.set_trace()
    #                                 # print('prune node generative')
    #                                 self.hiddenNodePruning(iHiddenLayer)
    #                                 # if iHiddenLayer == 0:
    #                                 #     self.averageFeature = meanStdCalculator()

    #                 # declare parameters to be trained
    #                 if generative and firstEvolution:
    #                     self.getTrainableParameters(mode=0, layerIdx = iHiddenLayer)

    #                     # specify optimizer
    #                     optimizer = torch.optim.SGD(self.netOptim, lr = self.lrGen)

    #                     # forward pass
    #                     self.feedforwardTrainGenerative(minibatch_xTrain)
    #                     loss = self.criterionGen(self.scoresGenerativeTrain, minibatch_x_noNoise)

    #                     # backward pass
    #                     optimizer.zero_grad()
    #                     loss.backward()

    #                     # apply gradient
    #                     optimizer.step()

    #         if self.nHiddenLayer > 1:
    #             # prepare training data for the next hidden layer
    #             with torch.no_grad():
    #                 currnet = self.net[iHiddenLayer].network
    #                 obj     = currnet.eval()
    #                 obj     = obj.to(device)
    #                 batchDataGen,_,_ = obj(batchDataGen)

    def generativeTraining(self, device = torch.device('cpu'), batchSize = 1, epoch = 1, generative = True):
        # shuffle the data
        batchDataGen = self.batchData
        nData        = batchDataGen.shape[0]

        for iHiddenLayer in range(self.nHiddenLayer):
            self.growNode = False
            self.pruneNode = False
        
            for iEpoch in range(0,epoch):

                shuffled_indices = torch.randperm(nData)
                
                # masked input
                maskedX = maskingNoise(batchDataGen.clone().detach())   # some of the input feature

                for iData in range(0,nData,batchSize):
                    # load data
                    indices          = shuffled_indices[iData:iData+batchSize]

                    minibatch_xTrain = maskedX[indices]
                    minibatch_xTrain = minibatch_xTrain.to(device)

                    minibatch_x_noNoise  = batchDataGen[indices]
                    minibatch_x_noNoise  = minibatch_x_noNoise.to(device)

                    if iEpoch == 0:
                        if batchSize > 1:
                            minibatch_x_noNoise_mean = torch.mean(minibatch_x_noNoise,dim=0).unsqueeze(dim=0)
                        else:
                            minibatch_x_noNoise_mean = minibatch_x_noNoise

                        if iHiddenLayer == 0:
                            # calculate mean of input, only done in the first hidden layer
                            self.averageInput.updateMeanStd(minibatch_x_noNoise_mean)

                            # get bias and variance
                            # outProbit = probitFunc(self.averageInput.mean,self.averageInput.std)
                            # self.feedforwardBiasVarGenerative(outProbit)  # for sigmoid activation function

                            self.feedforwardBiasVarGenerative(self.averageInput.mean)  # for ReLU activation function

                        if iHiddenLayer > 0:
                            # get bias and variance
                            outProbit = probitFunc(self.averageInput.mean,self.averageInput.std)
                            self.feedforwardBiasVarGenerative(outProbit)         # for sigmoid activation function

                        # update bias variance
                        self.updateBiasVarianceGen(iHiddenLayer)

                        # growing
                        if generative:
                            self.growNodeIdentificationGen(iHiddenLayer)
                            if self.growNode and self.hnGrowing:
                                # pdb.set_trace()
                                # print('grow node generative')

                                # initiate the new weight
                                with torch.no_grad():
                                    self.feedforwardTrainGenerative(minibatch_xTrain)
                                    x_hat = self.scoresGenerativeTrain

                                WN = -(minibatch_x_noNoise - x_hat)
                                # AA = (torch.log(1/(minibatch_x_noNoise + 0.001) + 1))
                                # AA = (torch.log(1/(minibatch_x_noNoise + 0.001) - 1))
                                # BB = torch.matmul(self.HS, self.net[0].network.linear.weight.data)
                                # WN = -(1/(minibatch_xTrain + 0.001))*(AA + BB + self.net[0].network.biasDecoder.data.unsqueeze(0))
                                # pdb.set_trace()
                                # WN = (1/(AA + 0.001) * (1 - AA*BB - AA*self.net[0].network.biasDecoder.data.unsqueeze(0)))

                                # self.hiddenNodeGrowing(iHiddenLayer)
                                # pdb.set_trace()
                                self.hiddenNodeGrowingWithInit(init = WN, layerIdx = iHiddenLayer)
                                # if iHiddenLayer == 0:
                                #     self.averageFeature = meanStdCalculator()

                            # pruning
                            if not self.growNode and self.hnPruning:
                                self.pruneNodeIdentificationGen(iHiddenLayer)
                                if self.pruneNode:
                                    # pdb.set_trace()
                                    # print('prune node generative')
                                    self.hiddenNodePruning(iHiddenLayer)
                                    # if iHiddenLayer == 0:
                                    #     self.averageFeature = meanStdCalculator()

                    # declare parameters to be trained
                    if generative:
                        self.getTrainableParameters(mode=0, layerIdx = iHiddenLayer)

                        # specify optimizer
                        optimizer = torch.optim.SGD(self.netOptim, lr = self.lrGen)

                        # forward pass
                        self.feedforwardTrainGenerative(minibatch_xTrain)
                        loss = self.criterionGen(self.scoresGenerativeTrain, minibatch_x_noNoise)

                        # backward pass
                        optimizer.zero_grad()
                        loss.backward()

                        # apply gradient
                        optimizer.step()

            if self.nHiddenLayer > 1:
                # prepare training data for the next hidden layer
                with torch.no_grad():
                    currnet = self.net[iHiddenLayer].network
                    obj     = currnet.eval()
                    obj     = obj.to(device)
                    batchDataGen,_,_ = obj(batchDataGen)


    def discriminativeTraining(self,device = torch.device('cpu'),batchSize = 1,epoch = 1):
        self.growNode = False
        self.pruneNode = False

        # shuffle the data
        nData            = self.batchData.shape[0]
        
        # label for bias var calculation
        y_biasVar = F.one_hot(self.batchLabel, num_classes = self.net[-1].nOutputs).float()
        
        for iEpoch in range(0,epoch):

            shuffled_indices = torch.randperm(nData)

            for iData in range(0,nData,batchSize):
                # load data
                indices                 = shuffled_indices[iData:iData+batchSize]

                minibatch_xTrain         = self.batchData[indices]
                minibatch_xTrain         = minibatch_xTrain.to(device)
                minibatch_xTrain_biasVar = minibatch_xTrain

                minibatch_labelTrain     = self.batchLabel[indices]
                minibatch_labelTrain     = minibatch_labelTrain.to(device)
                minibatch_labelTrain     = minibatch_labelTrain.long()

                if iEpoch == 0:
                    minibatch_label_biasVar = y_biasVar[indices]
                    minibatch_label_biasVar = minibatch_label_biasVar.to(device)
                    
                    if batchSize > 1:
                        minibatch_xTrain_biasVar = torch.mean(minibatch_xTrain,dim=0).unsqueeze(dim=0)
                        minibatch_label_biasVar  = torch.mean(minibatch_label_biasVar,dim=0).unsqueeze(dim=0)

                    # calculate mean of input
                    # self.averageInput.updateMeanStd(torch.mean(minibatch_xTrain,dim=0).unsqueeze(dim=0))
                    # self.averageInput.updateMeanStd(minibatch_xTrain_biasVar)

                    # get bias and variance
                    # outProbit = probitFunc(self.averageInput.mean,self.averageInput.std)   # for Sigmoid activation function
                    # self.feedforwardBiasVarDiscriminative(outProbit,minibatch_label_biasVar)             # for Sigmoid activation function
                    self.feedforwardBiasVarDiscriminative(self.averageInput.mean,minibatch_label_biasVar)  # for ReLU activation function

                    # update bias variance
                    self.updateBiasVariance()

                    # growing
                    self.growNodeIdentification()
                    if self.growNode and self.hnGrowing: 
                        # print('grow node discriminative')
                        self.hiddenNodeGrowing(self.winLayerIdx)
                        # if self.winLayerIdx == 0:
                        #     self.averageFeature = meanStdCalculator()

                    # pruning
                    if not self.growNode and self.hnPruning:
                        self.pruneNodeIdentification(self.winLayerIdx)
                        if self.pruneNode:
                            # print('prune node discriminative')
                            self.hiddenNodePruning(self.winLayerIdx)
                            # if self.winLayerIdx == 0:
                            #     self.averageFeature = meanStdCalculator()

                # declare parameters to be trained
                self.getTrainableParameters()

                # specify optimizer
                optimizer = torch.optim.SGD(self.netOptim, lr = self.lr, momentum = 0.95, weight_decay = 0.00005)

                # forward pass
                self.feedforwardTrainDiscriminative(minibatch_xTrain)
                loss = self.criterion(self.scoresTrain,minibatch_labelTrain)

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # apply gradient
                optimizer.step()


    def trainingDataPreparation(self, batchData, batchLabel, activeLearning = False,
        advSamplesGenrator = False, minorityClass = None):

        if activeLearning:
            # sample selection
            # MCP: multiclass probability
            sortedMCP,_          = torch.sort(self.MultiClassProbability, descending=True)
            sortedMCP            = torch.transpose(sortedMCP, 1, 0)
            sampleConfidence     = sortedMCP[0]/torch.sum(sortedMCP[0:2], dim=0)
            indexSelectedSamples = sampleConfidence <= 0.75
            indexSelectedSamples = (indexSelectedSamples != 0).nonzero().squeeze()

            # selected samples
            batchData  = batchData[indexSelectedSamples]
            batchLabel = batchLabel[indexSelectedSamples]
            # print('selected sample size',batchData.shape[0])

        # training data preparation
        # there is no buffer data, no drift, no warning
        self.batchData  = batchData
        self.batchLabel = batchLabel
        # if self.driftStatus == 0 or self.driftStatus == 2:  # STABLE or DRIFT
        #     # check buffer
        #     if self.bufferData.shape[0] != 0:
        #         # add buffer to the current data batch
        #         self.batchData  = torch.cat((self.bufferData,batchData),0)
        #         self.batchLabel = torch.cat((self.bufferLabel,batchLabel),0)

        #         # clear buffer
        #         self.bufferData  = torch.Tensor().float()
        #         self.bufferLabel = torch.Tensor().long()
        #     else:
        #         # there is no buffer data
        #         self.batchData  = batchData
        #         self.batchLabel = batchLabel

        #     # provide data category for original samples
        #     # 0: original samples; 1: anomaly samples; 2: augmented samples
        #     nOriginalData       = self.batchData.shape[0]

        #     if self.driftStatus == 2 and self.anomalyDataNadine.anomalyData.shape[0] != 0:
        #         # check anomaly data if drift
        #         # add anomaly data to the current data batch
        #         nAnomalyData    = self.anomalyDataNadine.anomalyData.shape[0]
        #         self.batchData  = torch.cat((self.anomalyDataNadine.anomalyData, self.batchData),0)
        #         self.batchLabel = torch.cat((self.anomalyDataNadine.anomalyLabel,self.batchLabel),0)
        #         # print('$$$ Anomaly data is added to the training set. Number of data: ',self.batchData.shape[0],'$$$')
        #         self.anomalyDataNadine.reset()

        #         # provide data category for anomaly data
        #         # 0: original samples; 1: anomaly samples; 2: augmented samples

        # if self.driftStatus == 1:  # WARNING
        #     # store data to buffer
        #     # print('Store data to buffer')
        #     self.bufferData  = batchData.clone().detach()
        #     self.bufferLabel = batchLabel.clone().detach()

        # generate adversarial samples for minority class
        # if advSamplesGenrator and (self.driftStatus == 0 or self.driftStatus == 2):
        if advSamplesGenrator:
            # prepare data
            if minorityClass is not None:
                # select the minority class data
                adversarialBatchData  = self.batchData [self.batchLabel == minorityClass]
                adversarialBatchLabel = self.batchLabel[self.batchLabel == minorityClass]

                nMinorityClass = adversarialBatchData.shape[0]
                nMajorityClass = self.batchData.shape[0] - nMinorityClass
            else:
                # select all data
                adversarialBatchData  = self.batchData.clone().detach()
                adversarialBatchLabel = self.batchLabel.clone().detach()

            # forward pass
            adversarialBatchData.requires_grad_()
            self.feedforwardTrain(adversarialBatchData)
            lossAdversarial = self.criterion(self.scoresTrain,adversarialBatchLabel)

            # backward pass
            lossAdversarial.backward()

            # get adversarial samples
            adversarialBatchData = adversarialBatchData.clone().detach() + 0.007*torch.sign(adversarialBatchData.grad)

            self.batchData  = torch.cat((self.batchData,adversarialBatchData),0)
            self.batchLabel = torch.cat((self.batchLabel,adversarialBatchLabel),0)

            # provide data category for augmented data
            # 0: original samples; 1: anomaly samples; 2: augmented samples
            nAdversarialSamples = adversarialBatchData.shape[0]
            # print('selected sample size',self.batchData.shape[0])

    def getTrainableParameters(self, mode=1, layerIdx = 0):
        # mode: 1 discriminative, 0: generative
        if mode == 0:
            netOptim  = []
            netOptim  = netOptim + list(self.net[layerIdx].network.parameters())

        if mode == 1:
            netOptim  = []
            for iLayer in range(len(self.net)):
                netOptim  = netOptim + list(self.net[iLayer].network.parameters())
                
        self.netOptim = netOptim

    # ============================= Testing ==============================
    def testing(self,x,label,device = torch.device('cpu')):
        # load data
        x     = x.to(device)
        label = label.to(device)
        label = label.long()
        
        # testing
        start_test          = time.time()
        self.feedforwardTest(x)
        end_test            = time.time()
        self.testingTime    = end_test - start_test
        
        loss                = self.criterion(self.scoresTest,label)
        self.testingLoss    = loss.detach().item()
        correct             = (self.predictedLabel == label).sum().item()
        self.accuracy       = 100*correct/(self.predictedLabel == label).shape[0]  # 1: correct, 0: wrong
        self.trueClassLabel = label