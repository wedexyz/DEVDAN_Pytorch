import numpy as np
import time
import pdb
from utilsDEVDAN import meanStdCalculator, plotPerformance, labeledIdx
from DEVDANbasic import DEVDAN
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import progressbar
import pdb

def DEVDANmain(DEVDANnet,dataStreams,trainingBatchSize = 1, noOfEpoch = 1, labeled = True, nLabeled = 1, generative = True):

    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    Accuracy     = []
    testingLoss  = []
    testingTime  = []
    trainingTime = []
    Y_pred       = []
    Y_true       = []
    Iter         = []
    
    accuracyHistory      = []
    lossHistory          = []
    hiddenNodeHistory    = []
    hiddenNodeGenHistory = []
    hiddenLayerHistory   = []
    winningLayerHistory  = []

    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode    = []
    nHiddenNodeGen = []

    # batch loop
    bar = progressbar.ProgressBar(maxval=dataStreams.nBatch)
    for iBatch in range(0,dataStreams.nBatch):
        # print('Batch: ', iBatch)
        # load data
        batchIdx   = iBatch + 1
        batchData  = dataStreams.data[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.label[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        nBatchData = batchData.shape[0]

        # testing
        DEVDANnet.testing(batchData,batchLabel)
        if iBatch > 0:
            Y_pred = Y_pred + DEVDANnet.predictedLabel.tolist()
            Y_true = Y_true + DEVDANnet.trueClassLabel.tolist()

        start_train = time.time()

        # training data preparation generative
        DEVDANnet.trainingDataPreparation(batchData,batchLabel)

        # generative training
        DEVDANnet.generativeTraining(batchSize = trainingBatchSize, epoch = noOfEpoch, generative = generative)

        # calculate network evolution generative
        hiddenNodeGenHistory.append(DEVDANnet.nHiddenNode)

        # training data preparation discriminative
        if nLabeled < 1:
            lblIdx = labeledIdx(nBatchData, nLabeled)
            DEVDANnet.trainingDataPreparation(batchData[lblIdx],batchLabel[lblIdx])

        # discriminative training
        DEVDANnet.discriminativeTraining(batchSize = trainingBatchSize, epoch = noOfEpoch)

        # calculate network evolution discriminative
        nHiddenNode.append(DEVDANnet.nHiddenNode)
        
        end_train = time.time()
        training_time = end_train - start_train

        # store performance and evolution history
        accuracyHistory.append(DEVDANnet.accuracy)
        lossHistory.append(DEVDANnet.testingLoss)
        hiddenNodeHistory.append(DEVDANnet.nHiddenNode)
        hiddenLayerHistory.append(DEVDANnet.nHiddenLayer)
        winningLayerHistory.append(DEVDANnet.winLayerIdx+1)
        Iter.append(iBatch)

        if iBatch > 0:
            # calculate performance
            Accuracy.append(DEVDANnet.accuracy)
            testingLoss.append(DEVDANnet.testingLoss)
            testingTime.append(DEVDANnet.testingTime)
            trainingTime.append(training_time)

        bar.update(iBatch+1)
    
    
    print('\n')
    print('=== Performance result ===')
    print('Accuracy: ', np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime),'(+/-)',np.std(trainingTime))

    print('\n')
    print('=== Average network evolution ===')
    print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.std(nHiddenNode))

    print('\n')
    print('=== Final network structure ===')
    DEVDANnet.getNetProperties()
    
    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))

    allPerformance = [np.mean(Accuracy),
                        f1_score(Y_true, Y_pred, average='weighted'),precision_score(Y_true, Y_pred, average='weighted'),
                        recall_score(Y_true, Y_pred, average='weighted'),
                        (np.mean(trainingTime)),np.mean(testingTime),
                        DEVDANnet.nHiddenLayer,DEVDANnet.nHiddenNode]

    performanceHistory = [Iter,accuracyHistory,lossHistory,hiddenNodeHistory,hiddenNodeGenHistory]

    return DEVDANnet, performanceHistory, allPerformance


def DEVDANmainID(DEVDANnet,dataStreams,trainingBatchSize = 1, noOfEpoch = 1, labeled = True, nLabeled = 1, nInitLabel = 1000):

    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    Accuracy     = []
    testingLoss  = []
    testingTime  = []
    trainingTime = []
    Y_pred       = []
    Y_true       = []
    Iter         = []
    
    accuracyHistory      = []
    lossHistory          = []
    hiddenNodeHistory    = []
    hiddenNodeGenHistory = []
    hiddenLayerHistory   = []
    winningLayerHistory  = []

    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode    = []
    nHiddenNodeGen = []

    nInit = 0

    # batch loop
    bar = progressbar.ProgressBar(maxval=dataStreams.nBatch).start()
    for iBatch in range(0,dataStreams.nBatch):
        # load data
        batchIdx   = iBatch + 1
        batchData  = dataStreams.data[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.label[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        nBatchData = batchData.shape[0]

        nInit     += nBatchData

        # testing
        DEVDANnet.testing(batchData,batchLabel)
        
        if nInit > nInitLabel:
            Y_pred = Y_pred + DEVDANnet.predictedLabel.tolist()
            Y_true = Y_true + DEVDANnet.trueClassLabel.tolist()
            Accuracy.append(DEVDANnet.accuracy)
            testingLoss.append(DEVDANnet.testingLoss)

        start_train = time.time()

        # training data preparation generative
        DEVDANnet.trainingDataPreparation(batchData,batchLabel)

        # generative training
        DEVDANnet.generativeTraining(batchSize = trainingBatchSize, epoch = noOfEpoch)

        # calculate network evolution generative
        hiddenNodeGenHistory.append(DEVDANnet.nHiddenNode)

        # training data preparation discriminative
        if nLabeled < 1:
            lblIdx = labeledIdx(nBatchData, nLabeled)
            DEVDANnet.trainingDataPreparation(batchData[lblIdx],batchLabel[lblIdx])

        # discriminative training
        if nInit <= nInitLabel:
            DEVDANnet.discriminativeTraining(batchSize = trainingBatchSize, epoch = noOfEpoch)

        # calculate network evolution discriminative
        nHiddenNode.append(DEVDANnet.nHiddenNode)
        
        end_train = time.time()
        training_time = end_train - start_train

        # store performance and evolution history
        accuracyHistory.append(DEVDANnet.accuracy)
        lossHistory.append(DEVDANnet.testingLoss)
        hiddenNodeHistory.append(DEVDANnet.nHiddenNode)
        hiddenLayerHistory.append(DEVDANnet.nHiddenLayer)
        winningLayerHistory.append(DEVDANnet.winLayerIdx+1)
        Iter.append(iBatch)

        if nInit > nInitLabel:
            # calculate performance
            testingTime.append(DEVDANnet.testingTime)
            trainingTime.append(training_time)

        bar.update(iBatch+1)
    
    
    print('\n')
    print('=== Performance result ===')
    print('Accuracy: ', np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime),'(+/-)',np.std(trainingTime))

    print('\n')
    print('=== Average network evolution ===')
    print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.mean(nHiddenNode))

    print('\n')
    print('=== Final network structure ===')
    DEVDANnet.getNetProperties()

    allPerformance = [np.mean(Accuracy),
                        f1_score(Y_true, Y_pred, average='weighted'),precision_score(Y_true, Y_pred, average='weighted'),
                        recall_score(Y_true, Y_pred, average='weighted'),
                        (np.mean(trainingTime)),np.mean(testingTime),
                        DEVDANnet.nHiddenLayer,DEVDANnet.nHiddenNode]
    
    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))

    performanceHistory = [Iter,accuracyHistory,lossHistory,hiddenNodeHistory,hiddenNodeGenHistory]

    return DEVDANnet, performanceHistory, allPerformance