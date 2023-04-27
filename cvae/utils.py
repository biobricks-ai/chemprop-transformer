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

def accuracy(pred, inval):
    pred = torch.sigmoid(pred)
    y_pred = (pred > 0.5).float()
    
    correct = (y_pred == inval).sum().item()
    total = inval.size(0)
    
    # Calculate accuracy
    acc = correct / total
    return acc

def balanced_accuracy(pred, inval):
    pred = torch.sigmoid(pred)
    y_pred = (pred > 0.5).float()

    true_positives = torch.zeros(2)
    false_negatives = torch.zeros(2)
    false_positives = torch.zeros(2)

    for i in range(2):
        true_positives[i] = ((y_pred == i) & (inval == i)).sum().item()
        false_negatives[i] = ((y_pred != i) & (inval == i)).sum().item()
        false_positives[i] = ((y_pred == i) & (inval != i)).sum().item()

    recall_per_class = true_positives / (true_positives + false_negatives)
    balanced_acc = torch.mean(recall_per_class).item()

    return balanced_acc