import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def generateCharSet(data, maxLength):
    charSet = set([' '])
    for smi in tqdm(list(data)):
        charSet = charSet.union(set(smi.ljust(maxLength)))
        
    charSet = sorted(list(charSet))
    charToInt = dict((c, i) for i, c in enumerate(charSet))
    intToChar = dict((i, c) for i, c in enumerate(charSet))
    return charSet, charToInt, intToChar

def idTo1Hot(id, labelsLength):
    oneHot = [0]*labelsLength
    oneHot[id] = 1
    hotTensor = np.array(oneHot) #one hot probability tensor
    return hotTensor

def toEmbedding(dataIn, charToInt, maxLength):
    dataIn = dataIn.ljust(maxLength)
    dataIn = list(dataIn)
    embedding = [charToInt[c] for c in dataIn]
    # embedding = [idTo1Hot(embed, 58) for embed in embedding]
    return np.array(embedding)

def embeddingAccuracy(x, xHat):
    correct = 0
    for i in range(len(x)):
        if x[i] == xHat[i]:
            correct += 1  
    return (correct/len(x))*100

# def idxToEmbed(id):
#     embedding = bin(id).replace('0b','')
#     embedding = [int(iEmbed) for iEmbed in embedding]
#     embedding = [0]*(11-len(embedding)) + embedding
#     embedding = np.array(embedding)
#     return embedding

def loadTrain(folderPath):
    print('loading training set')
    trainTensors = np.load('{}train.npy'.format(folderPath), allow_pickle=True)
    return trainTensors

def loadTest(folderPath):
    print('loading testing set')
    testTensors = np.load('{}test.npy'.format(folderPath), allow_pickle=True)
    return testTensors

def loadValid(folderPath):
    print('loading validation set')
    validTensors = np.load('{}valid.npy'.format(folderPath), allow_pickle=True)
    return validTensors

def loadDataset(folderPath):
    trainTensors = np.load('{}train.npy'.format(folderPath), allow_pickle=True)
    testTensors = np.load('{}test.npy'.format(folderPath), allow_pickle=True)
    validTensors = np.load('{}valid.npy'.format(folderPath), allow_pickle=True)
    return trainTensors, testTensors, validTensors

def stochasticToSmiles(matrix, vocab):
    tokenizedEmbedd = []
    for proBvect in matrix:
        proBvect = proBvect.tolist()
        token = np.random.choice(len(vocab), 1, proBvect)
        tokenizedEmbedd.append(token)
        
    return tokenizedEmbedd