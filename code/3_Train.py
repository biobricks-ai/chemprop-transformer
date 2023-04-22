from __future__ import print_function

import json
import os
import signal

import numpy as np
import dvc.api
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from CVAE.model import CVAE
from utils import loadTrainData, prepareInputs, plotTSNE, generateMask
from torch.optim.lr_scheduler import ExponentialLR

torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)

signal_received = False

def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
    
signal.signal(signal.SIGINT, handle_interrupt)

def cvaeLoss(x, xHat, mu, logvar, mask=None, beta=1):
    if mask is not None:
        RECON = F.cross_entropy(xHat, x, reduction='none')
        RECON = RECON * mask
        RECON = torch.sum(RECON) / torch.sum(mask)
    else:
        RECON = F.cross_entropy(xHat, x)
        
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta
    return RECON + KLD, RECON.item(), KLD.item()

def evaluate(model, validTensors):
    
    endToken = model.vocabSize-1
    
    model.eval()
    evalLoss = []
    reconLoss = []
    kldLoss = []
    for data in tqdm(validTensors):
        
        mask = generateMask(data[0], endToken)
        mask = mask.to(device)
        
        embedding, labels = prepareInputs(data, model.vocabSize, device)
        
        with torch.no_grad():
            xHat, z_mean, z_logvar, _ = model(embedding, labels)
            loss, recon, kld = cvaeLoss(embedding, xHat, z_mean, z_logvar, mask)
            evalLoss.append(loss.item())
            reconLoss.append(recon)
            kldLoss.append(kld)
            
        if signal_received:
            print('Stopping actual validation step...')
            break
    
    evalLoss = sum(evalLoss) / len(evalLoss)
    reconLoss = sum(reconLoss) / len(reconLoss)
    kldLoss = sum(kldLoss) / len(kldLoss)    
    
    return evalLoss, reconLoss, kldLoss
    

def train(model, optimizer, scheduler, folderPath, otuputFolder, vocab, epochs=5):
    trainTensors, validTensors = loadTrainData(folderPath)
    
    checkpointId = 0
    bestTrainLoss = np.inf
    bestEvalLoss = np.inf
    
    for epoch in range(epochs):
        model.train()
        print('{}/{}'.format(epoch+1, epochs))
        epochLoss = []
        reconLoss = []
        kldLoss = []
        
        #tsne
        encodedList = []
        assayList = []
        valueList = []
        
        for data in tqdm(trainTensors):
        
            mask = generateMask(data[0], model.vocabSize-1)
            mask = mask.to(device)
            
            assay = int(np.argmax(data[1]))
            assay =  vocab['assayMap'][assay]
            value = data[2].tolist()[0]
            
            embedding, labels = prepareInputs(data, model.vocabSize, device)
            
            xHat, z_mean, z_logvar, encodedEmbedding = model(embedding, labels)
            loss, recon, kld = cvaeLoss(embedding, xHat, z_mean, z_logvar, mask)
            epochLoss.append(loss.item())
            reconLoss.append(recon)
            kldLoss.append(kld)
            
            loss.backward()
            optimizer.step()
            
            encodedList.append(encodedEmbedding)
            assayList.append(assay)
            valueList.append(value)
            
            if signal_received:
                print('Stopping actual train step...')
                break
            
        scheduler.step()
        
        epochLoss = sum(epochLoss)/len(epochLoss)
        reconLoss = sum(reconLoss)/len(reconLoss)
        kldLoss = sum(kldLoss)/len(kldLoss)
        
        evalLoss, evalReconLoss, evalKldLoss = evaluate(model, validTensors)
        
        with open('{}loss.tsv'.format(metricsOutPath), 'a+') as f:
            f.write('{}\t{}\t{}\n'.format(epoch, epochLoss, evalLoss))
        
        with open('{}recon.tsv'.format(metricsOutPath), 'a+') as f:
            f.write('{}\t{}\t{}\n'.format(epoch, reconLoss, evalReconLoss))
            
        with open('{}kld.tsv'.format(metricsOutPath), 'a+') as f:
            f.write('{}\t{}\t{}\n'.format(epoch, kldLoss, evalKldLoss))
            
        if epochLoss < bestTrainLoss and evalLoss < bestEvalLoss:
            bestTrainLoss = epochLoss
            bestEvalLoss = evalLoss
            
            torch.save(model.state_dict(),'{}checkpoint{}epoch{}.pt'.format(otuputFolder, checkpointId, epoch))
            torch.save(model.state_dict(),'{}bestModel.pt'.format(otuputFolder))
            
            plotTSNE(encodedList, assayList, valueList, 512, plotsOutPath='{}tsne/checkpoint{}epoch{}'.format(metricsOutPath, checkpointId, epoch), colorCol='Activity')
            
            checkpointId += 1
            
        if signal_received:
            print('Stopping training...')
            break
    
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    params = dvc.api.params_show()
    
    latentDim = params['latentDim']
    
    metricsOutPath = params['training']['metricsOutPath']
    processedDataPath = params['training']['processedData']
    outModelFolder = params['training']['outModelFolder']
    epochs = params['training']['epochs']
    
    os.makedirs(outModelFolder, exist_ok=True)
    os.makedirs(metricsOutPath, exist_ok=True)
    os.makedirs('{}tsne/'.format(metricsOutPath), exist_ok=True)
    
    with open('{}loss.tsv'.format(metricsOutPath), 'w') as f:
        f.write('step\ttLoss\teLoss\n')
        
    with open('{}recon.tsv'.format(metricsOutPath), 'w') as f:
        f.write('step\ttLoss\teLoss\n')
        
    with open('{}kld.tsv'.format(metricsOutPath), 'w') as f:
        f.write('step\ttLoss\teLoss\n')
    
    with open('{}/modelInfo.json'.format(processedDataPath)) as f:
        modelInfo = json.load(f)
        
    with open('{}/vocabs.json'.format(processedDataPath)) as f:
        vocab = json.load(f)
        
    embeddingSize = modelInfo['embeddingSize']
    vocabSize = modelInfo['vocabSize']
    assaySize = modelInfo['assaySize']
    
    labelsSize = assaySize + 1
    
    model = CVAE(embeddingSize, vocabSize, labelsSize, latentDim)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    scheduler = ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
    
    train(model, optimizer, scheduler, processedDataPath, outModelFolder, vocab, epochs)