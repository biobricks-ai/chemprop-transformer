import h5py
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
tqdm.pandas()

from utils import *

def generateCharSet(data):
    charSet = set()
    for smi in data.unique().tolist():
        charSet = charSet.union(set(smi.ljust(120)))
    charSet = sorted(list(charSet))
    charToInt = dict((c, i) for i, c in tqdm(enumerate(charSet)))
    
    return charSet, charToInt

def smiToEmbedding(smi, charToInt, maxEmbeddingSize = 120):
    smi = smi.ljust(maxEmbeddingSize)
    smi = list(smi)
    embedding = [charToInt[c] for c in smi]
    return np.array(embedding)

def embedToHot(embed, charSetSize):
    oneHotMatrix = []
    for tk in embed:
        oneHot = [0]*charSetSize
        oneHot[tk] = 1
        oneHotMatrix.append(oneHot)
    return np.array(oneHotMatrix)

def assayToHot(assay, assaySize):
    oneHot = [0]*assaySize
    oneHot[assay] = 1
    return np.array(oneHot)

# def hotToEmbed(matrix):
#     return np.array([np.argmax(oneHot) for oneHot in matrix])

# def embedToSmi(embed, charSet):
#     return ''.join(charSet[tk] for tk in embed)

# def loadDataset(filename):
#     with h5py.File(filename, 'r') as h5f:
#         dataTrain = h5f['trainSet'][:]
#         dataTest = h5f['testSet'][:]
#         charset =  h5f['charset'][:]
#     return (dataTrain, dataTest, charset)

def preProcess(path, outPath):
    data = pd.read_csv(path)[['smiles', 'assay', 'value']]
    
    print('Generating charset')
    charSet, charToInt = generateCharSet(data['smiles'])
    
    
    data = data[data['smiles'].str.len() <= 120]
    data['Occur'] = data.groupby('assay')['value'].transform('size')
    data = data[data['Occur'] >= 500]
    
    uniqueAssays = data['assay'].unique().tolist()
    assaysDict = {k:i for i, k in enumerate(uniqueAssays)}
    
    print(len(uniqueAssays))
    
    # preTrainData = pd.DataFrame(data['smiles'].unique(), columns = ['smiles'])
    
    # a1 = data[data['assay'] == '8bd16db6-124f-447c-98c1-d2a86849b333']
    # a2 = data[data['assay'] == 'e597a518-81c2-41bf-b876-0e09f961ceb6']

    # a2 = a2.sample(n=10000)

    # data = pd.concat([a1, a2])
    
    preTrainData = pd.DataFrame(data['smiles'].unique(), columns = ['smiles'])
    print('Indexing embeddings for pretraining') 
    preTrainData['embedding'] = preTrainData['smiles'].progress_apply(lambda smiles: smiToEmbedding(smiles, charToInt))
    print('Enconding to 1hot embeddings for pretraining')
    preTrainData['embedding'] = preTrainData['embedding'].progress_apply(lambda smiles: embedToHot(smiles, len(charSet)))
    
    preTrainSet = preTrainData.sample(frac=.7)
    preTrainTestSet = preTrainData.drop(preTrainSet.index)
    
    PretrainEmbeddings = np.array(list(preTrainSet['embedding']))
    PretrainTestEmbeddings  = np.array(list(preTrainTestSet['embedding']))
       
    data = data.sample(n=1000000)
    print('Indexing embeddings for training') 
    data['embedding'] = data['smiles'].progress_apply(lambda smiles: smiToEmbedding(smiles, charToInt))
    print('Enconding to 1hot embeddings for training')
    data['embedding'] = data['embedding'].progress_apply(lambda smiles: embedToHot(smiles, len(charSet)))
    print('Enconding to 1hot activities for training')
    data['activity'] = data['assay'].progress_apply(lambda assay: assayToHot(assaysDict[assay], len(uniqueAssays)))
    print('Parsing value to tensors for training')
    data['value'] = data['value'].progress_apply(lambda value: np.array([value]))
    
    trainSet = data.sample(frac=.7)
    data = data.drop(trainSet.index)
    testSet = data.sample(frac=.65)
    validSet = data.drop(testSet.index)
    
    trainEmbeddings = np.array(list(trainSet['embedding']))
    testEmbeddings  = np.array(list(testSet['embedding']))
    validEmbeddings  = np.array(list(validSet['embedding']))
    
    trainActivities= np.array(list(trainSet['activity']))
    testActivities = np.array(list(testSet['activity']))
    validActivities = np.array(list(validSet['activity']))
    
    trainValues= np.array(list(trainSet['value']))
    testValues = np.array(list(testSet['value']))
    validValues = np.array(list(validSet['value']))
    
    with h5py.File(outPath, 'w') as f:
        f.create_dataset('charset', data = charSet)
        f.create_dataset('uniqueAssays', data = uniqueAssays)
        
        f.create_dataset('data_pretrain', shape = PretrainEmbeddings.shape, data = PretrainEmbeddings)
        f.create_dataset('data_pretrain_test', shape = PretrainTestEmbeddings.shape, data = PretrainTestEmbeddings)
        
        f.create_dataset('data_train', shape = trainEmbeddings.shape, data = trainEmbeddings)
        f.create_dataset('data_test', shape = testEmbeddings.shape, data = testEmbeddings)
        f.create_dataset('data_valid', shape = validEmbeddings.shape, data = validEmbeddings)
        
        f.create_dataset('data_train_activities', shape = trainActivities.shape, data = trainActivities)
        f.create_dataset('data_test_activities', shape = testActivities.shape, data = testActivities)
        f.create_dataset('data_valid_activities', shape = validActivities.shape, data = validActivities)
        
        f.create_dataset('data_train_values', shape = trainValues.shape, data = trainValues)
        f.create_dataset('data_test_values', shape = testValues.shape, data = testValues)
        f.create_dataset('data_valid_values', shape = validValues.shape, data = validValues)
        
if __name__ == '__main__':
    path = 'data/raw/RawChemHarmony.csv'
    outPath = 'data/processed/ProcessedChemHarmony.h5'
    os.makedirs('data/processed', exist_ok=True)
    preProcess(path, outPath) 
        
