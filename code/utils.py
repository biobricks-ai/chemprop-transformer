import dvc.api, numpy as np, torch, os
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

def rootdir(p=Path(os.getcwd())):
    isroot = (p / 'dvc.yaml').exists()
    return p if(isroot) else rootdir(p.parent) if p.parent != p else None

def loadparams():
    cwd = os.getcwd()
    os.chdir(rootdir())
    res = dvc.api.params_show()
    os.chdir(cwd)
    return res

def res(path):
    "Returns the path relative to the root directory"
    return rootdir() / path

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
    dataIn = dataIn.ljust(maxLength)[0:maxLength]
    dataIn = list(dataIn)
    embedding = [charToInt[c] for c in dataIn]
    return np.array(embedding)

def embeddingAccuracy(x, xHat):
    correct = 0
    for i in range(len(x)):
        if x[i] == xHat[i]:
            correct += 1  
    return (correct/len(x))*100