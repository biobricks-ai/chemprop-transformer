import torch
from CVAE.model import CVAE
from utils import loadTest, plotTSNE, prepareInputs
from tqdm import tqdm
import numpy as np
from rdkit import Chem, DataStructs

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import json
import dvc.api
import os


def inference(model, processedData, vocab):
    vocabSize = model.vocabSize
    
    model = model.encoder
    model.eval()
    
    testTensors = loadTest(processedData)
    testTensors = testTensors[:1000]
    
    #tsne
    encodedList = []
    assayList = []
    valueList = []
    
    for data in tqdm(testTensors):
        assay = int(np.argmax(data[1]))
        assay =  vocab['assayMap'][assay]
        value = data[2].tolist()[0]
        
        embedding, labels = prepareInputs(data, vocabSize, device)
        embedding = embedding.view(-1)
        
        zMean, zLogvar = model(embedding, labels)
        
        std = torch.exp(0.5*zLogvar)
        eps = torch.randn_like(std)
        
        z = zMean + eps*std
        
        encodedList.append(z.tolist())
        assayList.append(assay)
        valueList.append(value)
    
    plotTSNE(encodedList, assayList, valueList, 512, plotsOutPath='{}reconVisualization'.format(metricsOutPath), colorCol='Activity')
        

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    params = dvc.api.params_show()
    
    latentDim = params['latentDim']
    
    modelFolder = params['inference']['modelFolder']
    processedData = params['inference']['processedData']
    metricsOutPath = params['inference']['metricsOutPath']
    
    os.makedirs(metricsOutPath, exist_ok=True)
    
    with open('{}/modelInfo.json'.format(processedData)) as f:
        modelInfo = json.load(f)
        
    with open('{}/vocabs.json'.format(processedData)) as f:
        vocab = json.load(f)
    
    embeddingSize = modelInfo['embeddingSize']
    vocabSize = modelInfo['vocabSize']
    assaySize = modelInfo['assaySize']
    
    labelsSize = assaySize + 1
    
    model = CVAE(embeddingSize, vocabSize, labelsSize, latentDim)
    model = model.to(device)
    
    model.load_state_dict(torch.load('{}bestModel.pt'.format(modelFolder)))
    
    inference(model, processedData, vocab)