from tqdm import tqdm

import json, os, signal, dvc.api, numpy as np, pathlib, time
import torch, torch.nn.functional as F, torch.optim as optim
import torch.nn as nn

import torch.utils.data
from torch.utils.data import Dataset, Subset
import gc

import pandas as pd
import cvae.models.tokenizer

device = torch.device(f'cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)

charset = cvae.models.tokenizer.charset
data = pd.read_csv('data/raw/RawChemHarmony.csv')[['smiles','assay','value']]
smiles = [x for x in data['smiles'].values if len(x) < 120 and set(x).issubset(charset)]

tst = torch.load("data/processed/validation.pt").tensors
hld = torch.load("data/processed/holdout.pt").tensors
tensors = torch.load("data/processed/train.pt").tensors
tensors = [torch.vstack([tensors[i], tst[i], hld[i]]) for i in range(len(tensors))]

ismi, ilbl, ivalue = (0,1,2)
tstds = torch.utils.data.TensorDataset(*[tst[i] for i in [ismi, ilbl, ivalue]])
trnds = torch.utils.data.TensorDataset(*[tensors[i] for i in [ismi, ilbl, ivalue]])

dl = torch.utils.data.DataLoader(trnds, batch_size=320, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
tdl = torch.utils.data.DataLoader(tstds, batch_size=320, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

signal_received = False
def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
    
signal.signal(signal.SIGINT, handle_interrupt)

import importlib, cvae.models.predvae, cvae.models.tokenizer
importlib.reload(cvae.models.predvae)
vocabsize, nlabels = 58, 152
model = cvae.models.predvae.PredVAE(vocabsize, nlabels).to(device)
# model = cvae.models.predvae.PredVAE.load().to(device)
self = model
self.eval()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, verbose=True)
epochs, best_trn_loss, best_tst_loss = 100, np.inf, np.inf

def example():
    _ = model.eval()
    randsmi = smiles[np.random.randint(0,len(smiles))]
    egsmi = torch.Tensor(cvae.models.tokenizer.smiles_one_hot(randsmi)).to(device)
    egsmi = egsmi.unsqueeze(0)
    print(randsmi)
    decsmi = model(egsmi)
    argmax = torch.argmax(decsmi[0],dim=1)
    print(cvae.models.tokenizer.decode_smiles_from_indexes(argmax[0]))
    decsmi = model(egsmi)
    argmax = torch.argmax(decsmi[0],dim=1)
    print(cvae.models.tokenizer.decode_smiles_from_indexes(argmax[0]))

def evaluate():
    _ = model.eval()
    losses = (0, 0, 0, 0)
    
    # testtensors = next(iter(tdl))
    for i, testtensors in enumerate(tdl):
        insmi, inlbl, inval = [x.to(device).squeeze(0) for x in testtensors]
        decsmi, zmean, zlogvar = model(insmi)
        losstuple = model.loss(decsmi, insmi, zmean, zlogvar)
        
        losses = [x + y.item() for x,y in zip(losses,losstuple)]
        if signal_received:
            break
        
    return [ x/len(dl) for x in losses ] 

writepath = pathlib.Path("metrics/predvaeloss.tsv")
writepath.write_text("epoch\tloss\tpredloss\trecloss\tklloss\tl1loss\ttrainloss\n")


for p in model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))

mean_epochloss = np.inf
best_trn_loss, best_tst_loss = np.inf, np.inf
elosses = evaluate()

# Commit 'working pvae'
# 4 ELOSS: 16495.1 LOSS: 15319.1 REC: 10864.5 KL: 2250.1 VL: 2204.6:  28%|█▍   | 1092/3879 [01:11<03:01, 15.36batch/s]
# Cc1ccccc1N=C(O)C1=Cc2ccccc2C(=NNc2ccc([N+](=O)[O-])cc2C)C1=O
# Cc1ccccc1N=C(O)CN(Cc2ccccc2C(=O\c2ccc([N+](=O)[O-])c2)CCCCC1
# Cc1ccccc1N=C(O)C1(Cc2ccccc2C(=O)Nc2cc([N+](=O)[O-])cc2)CCC1=
for epoch in range(1000):
        
    pbar = tqdm(enumerate(dl), total=len(dl), unit="batch", ncols=100)

    _ = model.train()
    # i,tensors = next(enumerate(dl))
    epochloss = 0
    
    losses = []
    for i,tensors in pbar:
        optimizer.zero_grad()
        
        # inputs for this batch
        insmi, inlbl, inval = [x.to(device) for x in tensors]
        inval = inval.type(torch.float)
        
        decsmi, zmean, zlogvar = model(insmi)
        loss, recloss, klloss = model.loss(decsmi, insmi, zmean, zlogvar)
        losses = losses + [loss.item()]
        
        loss.backward()
        _= torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epochloss = epochloss + loss.item()
        
        # if i % 5 == 0:
        msg = f"{epoch} ELOSS: {elosses[0]:.1f} LOSS: {loss.item():.1f}"
        msg = f"{msg} REC: {recloss.item():.1f} KL: {klloss.item():.1f}"
        pbar.set_description(msg)
            
        if signal_received:
            break
    
    example()
    elosses = evaluate()
    mean_epochloss = epochloss / len(dl)
    scheduler.step(elosses[0])
    
    if signal_received:
        signal_received = False
        break
    
    if epochloss < best_trn_loss and elosses[0] < best_tst_loss:
        print("saving!")
        best_trn_loss = epochloss
        best_tst_loss = elosses[0]
        path = pathlib.Path("brick/pvae.pt")
        torch.save(model.state_dict(), path)

# HEATMAP ====================================================================
import importlib, seaborn as sns, matplotlib.pyplot as plt, pandas as pd, numpy as np
importlib.reload(cvae.models.predvae)

charset = cvae.models.tokenizer.charset
data = pd.read_csv('data/raw/RawChemHarmony.csv')[['smiles','assay','value']]
smiles = [x for x in data['smiles'].values if len(x) < 120 and set(x).issubset(charset)]


pvae = cvae.models.predvae.PredVAE.load().to(device)
pvae.eval()

# select 1000 random smiles
ransmi = np.random.choice(smiles, 10000)
embsmi = [cvae.models.tokenizer.smiles_one_hot(x) for x in ransmi]
embsmi = np.stack(embsmi)
embsmi = torch.tensor(embsmi).to(device)


# get latent vectors
zmean, _ = pvae.encode(embsmi)
smidist = torch.cdist(zmean, zmean, p=2).detach().cpu().numpy()

# build heatmap
plt.figure(figsize=(10, 8))  # Optional: Adjust the figure size
g = sns.clustermap(smidist, cmap='viridis')  # Choose the color map that you prefer
plt.show()


# ADDITIVE CVAE===============================================================
class CVAE(nn.Module):
    
    def __init__(self, vae):
        super(CVAE, self).__init__()
        
        ldim = vae.ldim
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
            
        self.lblembed = nn.Sequential(nn.Embedding(nlabels,ldim - 1))
        self.propencoder = nn.Sequential(nn.Embedding(nlabels,ldim))
        self.vvalencoder = nn.Sequential(nn.Linear(1,ldim))
        
    def encode(self, insmi, inlbl, inval):
        _, _, zmean, zlogvar = self.vae(insmi, inlbl)
        
        propembedding = self.propencoder(inlbl)
    
    def decode(self, z):
        return self.vae.decode(z)
        
    def forward(self, insmi, inlbl, inval):
        self.encode(insmi, inlbl, inval)
        
    def loss(self, insmi, decsmi, zmean, zlogvar):
        self.vae.loss(insmi, decsmi, zmean, zlogvar)
        pass

## MOLECULAR WEIGHT EXAMPLE====================================================
mwcvae = CVAE(model)

from rdkit import Chem
from rdkit.Chem import Descriptors

mw = [Descriptors.MolWt(Chem.MolFromSmiles(smi)) for smi in smiles]
tmw = torch.tensor(mw).unsqueeze(1)

smihot = cvae.models.tokenizer.smiles_one_hot
tsmi = np.stack(data['smiles'].progress_apply(smihot).to_numpy())
tsmi = torch.tensor(tsmi)

tds = torch.utils.data.TensorDataset(tsmi,tmw)
tdl = torch.utils.data.DataLoader(tds, batch_size=320, shuffle=True)

# List of SMILES strings
smiles_list = ['CCO', 'CCN', 'CCC']  # replace with your actual SMILES strings

# Calculate and print molecular weights
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mw = Descriptors.MolWt(mol)
        print(f'The molecular weight of {smiles} is {mw}')
    else:
        print(f'Invalid SMILES string: {smiles}')

## TRAIN ON LABEL VALUES AGAIN=================================================