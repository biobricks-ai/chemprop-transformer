from __future__ import print_function
from tqdm import tqdm

import json, os, signal, dvc.api, numpy as np, pathlib, time
import torch, torch.nn.functional as F, torch.optim as optim
import torch.nn as nn

import torch.utils.data
from torch.utils.data import Dataset, Subset

import cvae.models, cvae.models.biosim as bs
import cvae.utils as utils

device = torch.device(f'cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)


tensors = torch.load(utils.res("data/processed/train.pt")).tensors
tensors = [tensors[i] for i in [1,2,4]]
trnds = torch.utils.data.TensorDataset(*tensors)

dl = torch.utils.data.DataLoader(
    trnds, batch_size=100000, shuffle=True, 
    pin_memory=True, num_workers=4,
    persistent_workers=True)

test = torch.load(utils.res("data/processed/validation.pt")).tensors
test = [test[i] for i in [1,2,4]]
testinlbl, testinval, testinvae = [x.to(device).type(torch.float32) for x in test]
testinlbl = testinlbl.type(torch.int)
signal_received = False

def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
    
signal.signal(signal.SIGINT, handle_interrupt)

import importlib
importlib.reload(bs)
assays = len(tensors[0].unique(return_counts=True)[0])
model = bs.BioSim(292//2, assays).to(device)
optimizer = optim.Adam(model.parameters(),lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
epochs = 100   
best_trn_loss, best_tst_loss = np.inf, np.inf

model.train()

inlbl, inval, invae = [x.to(device) for x in tensors]
inval = inval.type(torch.float32)
inlbl = inlbl.type(torch.int)
torch.clamp(model.lbl_embed(inlbl),min=0,max=1)

# 38OCH 38 VLOSS: 0.73 VACC: 0.565 TLOSS: 0.56 
for epoch in range(100):
    epochLoss = []
    pbar = tqdm(range(5), unit="batch")
    lossfn = model.loss().to(device)
    
    testpred, emb = model(testinvae, testinlbl)
    evalloss = lossfn(testpred, testinval).item()
    evalbacc = bs.BioSim.balanced_accuracy(testpred, testinval)

    for _ in pbar:
    # for _ in range(5):
        optimizer.zero_grad()
        # model.cdense(invae[:,:292//2])
        
        # inputs for this batch
        # inlbl, inval, invae = [x.to(device) for x in tensors]
        # inval = inval.type(torch.float32)
        
        # outputs and loss
        pred, emb = model(invae, inlbl)
        emb
        pred = pred.to(device)
        loss = lossfn(pred, inval)
        acc = bs.BioSim.balanced_accuracy(pred, inval)
        # record losses
        epochLoss.append(loss.item())
        
        # update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pbar.set_description(f"EPOCH {epoch} VLOSS: {evalloss:.2f} VACC: {evalbacc:.3f} LOSS: {loss.item():.2f} ACC: {acc: .2f}")
        
        if signal_received:
            print('Stopping actual validation step...')
            break
        
    epochLoss = sum(epochLoss)/len(epochLoss)
    # scheduler.step(epochLoss)
    
    # writepath = pathlib.Path("metrics/loss.tsv")
    # with open(writepath, "a") as f:
    #     f.write(f"{epochLoss}\t{evalloss}\n")
        
    if signal_received:
        print('Stopping actual validation step...')
        signal_received = False
        break
    
    if epochLoss < best_trn_loss and evalloss < best_tst_loss:
            best_trn_loss = epochLoss
            best_tst_loss = evalloss
            path = pathlib.Path("brick/biosim.pt")
            torch.save(model.state_dict(), path)