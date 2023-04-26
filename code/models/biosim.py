from __future__ import print_function
from tqdm import tqdm

import json, os, signal, dvc.api, numpy as np, pathlib, time
os.chdir("code")
import torch, torch.nn.functional as F, torch.optim as optim
import torch.nn as nn

import utils
import torch.utils.data
from torch.utils.data import Dataset, Subset

import torch

device = torch.device(f'cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)


tensors = torch.load(utils.res("data/processed/train.pt")).tensors
tensors = [tensors[1],tensors[2],tensors[4]]
trnds = torch.utils.data.TensorDataset(*tensors)

dl = torch.utils.data.DataLoader(trnds, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)

test = torch.load(utils.res("data/processed/validation.pt")).tensors
test = [test[1],test[2],test[4]]
testinlbl, testinval, testinvae = [x.to(device) for x in test]

class Biosim(nn.Module):

    def __init__(self, vaedim, numlbl):
        super(Biosim, self).__init__()
        
        self.lbl_embed = nn.Sequential(nn.Embedding(numlbl, vaedim),nn.ReLU())
        self.chemlbl = nn.Sequential(nn.Linear(vaedim*2, vaedim),nn.ReLU(), nn.Linear(vaedim, vaedim),nn.Relu())
        
    def forward(self, invae, inlbl):
        eml = self.lbl_embed(inlbl)
        cat = torch.cat([invae, eml], dim=1)
        chemlbl = self.chemlbl(cat)
        dot = torch.sum(invae*chemlbl,dim=1)
        return dot


signal_received = False

def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
    
signal.signal(signal.SIGINT, handle_interrupt)

model = Biosim(292, 1000).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
epochs = 100   

# enumdl = enumerate(dl)
# _, tensors = next(enumdl)
inlbl, inval, invae = [x.to(device) for x in tensors]
pred = model(invae, inlbl)

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

for epoch in range(epochs):
    print('{}/{}'.format(epoch+1, epochs))
    epochLoss = []
    pbar = tqdm(range(10), ncols=100, unit="batch")
    model.train()
    lossfn = nn.BCEWithLogitsLoss().to(device)
    
    for _, tensors in enumerate(pbar):
        optimizer.zero_grad()
        
        # inputs for this batch
        # inlbl, inval, invae = [x.to(device) for x in tensors]

        # outputs and loss
        pred = model(invae, inlbl).to(device)
        loss = lossfn(pred,inval)
        
        l1_coeff = 0.001
        l1_reg = torch.norm(model.lbl_embed.weight, p=1)
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
    
    if signal_received:
        print('Stopping actual validation step...')
        signal_received = False
        break