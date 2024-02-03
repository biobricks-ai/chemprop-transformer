import torch
from CVAE.model import CVAE
from utils import loadTest, idTo1Hot, prepareInputs
from tqdm import tqdm
import numpy as np
from rdkit import Chem, DataStructs

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import json
import dvc.api
import os


def inference(model, processedData):
    vocabSize = model.vocabSize
    
    model = model.encoder
    model.eval()
    
    testTensors = loadTest(processedData)
    testTensors = testTensors[:1000]
    
    for data in tqdm(testTensors):
        embedding, labels = prepareInputs(data, vocabSize, device)
        embedding = embedding.view(-1)
        
        zMean, zLogvar = model(embedding, labels)
        
        std = torch.exp(0.5*zLogvar)
        eps = torch.randn_like(std)
        
        z = zMean + eps*std
        
        plt.scatter(z.tolist(), z.tolist()[::-1])
        
    plt.savefig('{}reconVisualization.png'.format(metricsOutPath))
        

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
    
    embeddingSize = modelInfo['embeddingSize']
    vocabSize = modelInfo['vocabSize']
    assaySize = modelInfo['assaySize']
    
    labelsSize = assaySize + 1
    
    model = CVAE(embeddingSize, vocabSize, labelsSize, latentDim)
    model = model.to(device)
    
    model.load_state_dict(torch.load('{}bestModel.pt'.format(modelFolder)))
    
    inference(model, processedData)