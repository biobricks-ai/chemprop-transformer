import numpy as np
import torch
from tqdm import tqdm
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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
    return np.array(embedding)

def embeddingAccuracy(x, xHat):
    correct = 0
    for i in range(len(x)):
        if x[i] == xHat[i]:
            correct += 1  
    return (correct/len(x))*100

def loadTrain(folderPath):
    print('loading training set...')
    trainTensors = np.load('{}train.npy'.format(folderPath), allow_pickle=True)
    return trainTensors

def loadTest(folderPath):
    print('loading testing set...')
    testTensors = np.load('{}test.npy'.format(folderPath), allow_pickle=True)
    return testTensors

def loadValid(folderPath):
    print('loading validation set...')
    validTensors = np.load('{}valid.npy'.format(folderPath), allow_pickle=True)
    return validTensors

def loadTrainData(folderPath):
    trainTensors = loadTrain(folderPath)
    validTensors = loadValid(folderPath)
    print('done!')
    
    return trainTensors, validTensors

def prepareInputs(data, vocabSize, device):
    embedding = np.array([idTo1Hot(i, vocabSize) for i in list(data[0])])
    embedding = torch.Tensor(embedding)
    embedding = embedding.to(device)

    assay = data[1]
    assay = torch.Tensor(assay)
    assay = assay.to(device)
    
    value = data[2]
    value = torch.Tensor(value)
    value = value.to(device)
    
    labels = torch.cat((assay, value), dim=0)
    
    return embedding, labels


def plotTSNE(embeddings, activity, values, latentSpaceSize=512, plotsOutPath='/', colorCol='Activity'):
    labels = ['z{}'.format(i) for i in range(latentSpaceSize)]
    
    df = pd.DataFrame(embeddings, columns=labels)
    scaled = StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=labels)
    
    tsne = TSNE(init='pca', learning_rate='auto')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne_features = tsne.fit_transform(scaled)
    
    tsne_df = pd.DataFrame(
        data=tsne_features, 
        columns=['TSNE1', 'TSNE2'])
    
    tsne_df['Activity'] = activity
    tsne_df['Value'] = values
    
    sns.lmplot(
        x='TSNE1', 
        y='TSNE2', 
        data=tsne_df, 
        hue=colorCol, 
        fit_reg=False, 
        legend=True
        )
        
    plt.savefig('{}TSNE.png'.format(plotsOutPath))