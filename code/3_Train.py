from __future__ import print_function
from tqdm import tqdm

import json, os, signal, dvc.api, numpy as np, pathlib, time
import torch, torch.nn.functional as F, torch.optim as optim
import torch.nn as nn

import torch.utils.data
from torch.utils.data import Dataset, Subset

import cvae.models
import cvae.utils as utils

device = torch.device(f'cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)


tensors = torch.load(utils.res("data/processed/train.pt")).tensors
tensors = [tensors[i] for i in [1,2,4]]
trnds = torch.utils.data.TensorDataset(*tensors)

dl = torch.utils.data.DataLoader(
    trnds, batch_size=32, shuffle=True, 
    pin_memory=True, num_workers=4,
    persistent_workers=True)

test = torch.load(utils.res("data/processed/validation.pt")).tensors
test = [test[i] for i in [1,2,4]]
testinlbl, testinval, testinvae = [x.to(device) for x in test]

signal_received = False

def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
    
signal.signal(signal.SIGINT, handle_interrupt)

model = cvae.models.biosim.BioSim(292, 1000).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
epochs = 100   

for epoch in range(epochs):
    print('{}/{}'.format(epoch+1, epochs))
    epochLoss = []
    pbar = tqdm(dl, ncols=100, unit="batch")
    model.train()
    lossfn = nn.BCEWithLogitsLoss().to(device)
    
    for _, tensors in enumerate(pbar):
        optimizer.zero_grad()
        
        # inputs for this batch
        inlbl, inval, invae = [x.to(device) for x in tensors]

        # outputs and loss
        pred = model(invae, inlbl).to(device)
        loss = lossfn(pred,inval)
        
        l1_coeff = 0.001
        l1_reg = torch.norm(model.lbl_embed[0].weight, p=1)
        loss += l1_coeff * l1_reg

        
        testpred = model(testinvae, testinlbl)
        acc = balanced_accuracy(testpred, testinval)

        # # record losses
        # epochLoss.append(loss.item())
        
        # update model
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f}, acc: {acc:.4f}")
        
        if signal_received:
            print('Stopping actual validation step...')
            break
        
    scheduler.step()
    
    epochLoss = sum(epochLoss)/len(epochLoss)
    
    if signal_received:
        print('Stopping actual validation step...')
        signal_received = False
        break
    
    if epochLoss < best_trn_loss and evalLoss < best_val_loss:
            best_trn_loss = epochLoss
            best_val_loss = evalLoss
            print("saving model...")
            torch.save(model.state_dict(),'{}/checkpoint_{}_epoch_{}.pt'.format(outdir, checkpt_id, epoch))
            torch.save(model.state_dict(),'{}/bestModel.pt'.format(outdir))
            
            checkpt_id += 1