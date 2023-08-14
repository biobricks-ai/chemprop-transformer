from __future__ import print_function
from tqdm import tqdm

import json, os, signal, dvc.api, numpy as np, pathlib, time
import torch, torch.nn.functional as F, torch.optim as optim
import torch.nn as nn

import torch.utils.data
from torch.utils.data import Dataset, Subset
import gc

import cvae.models.predvae
import cvae.utils as utils
import cvae.models.tokenizer
import cvae.models, cvae.models.tokenizer as tokenizer
import pandas as pd

import shutil

tqdm.pandas()
device = torch.device(f'cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)

params = utils.loadparams()['preprocessing']
data = pd.read_csv(utils.res(params["rawDataPath"]))[['smiles']]

chunksize = 320
data_iterator = pd.read_csv(utils.res(params["rawDataPath"]), usecols=['smiles'], chunksize=chunksize)

# tmpdir = pathlib.Path("staging/smiles")
# if tmpdir.exists():
#     shutil.rmtree(tmpdir)    

# tmpdir.mkdir(parents=True, exist_ok=True)

# i,chunk = next(enumerate(data_iterator))
# for i,chunk in tqdm(enumerate(data_iterator), total=len(data) // chunksize):
#     valid = lambda x: len(x) < 120 and set(x).issubset(tokenizer.charset)
#     valid_smiles = chunk['smiles'].apply(valid)
#     chunk = chunk[valid_smiles]
    
#     tokens = chunk['smiles'].apply(tokenizer.smiles_one_hot)
#     tensor = torch.stack([torch.from_numpy(arr) for arr in tokens])
#     path = tmpdir / f"chunk{i}.pt"
#     torch.save(tensor, path)

class ChunkedTensorDataset(Dataset):
    
    def __init__(self, folder='staging/smiles', num_chunks=None):
        self.folder = folder
        self.chunk_files = sorted([f for f in os.listdir(folder) if f.endswith('.pt')])

    def __len__(self):
        return len(self.chunk_files)
    
    def __getitem__(self, idx):
        chunkfile = self.chunk_files[idx]
        chunkpath = os.path.join(self.folder, chunkfile)
        return torch.load(chunkpath)

# Instantiate the custom Dataset
ds = ChunkedTensorDataset()

tdl = torch.utils.data.DataLoader(
    ds, batch_size=1, shuffle=True, 
    pin_memory=True, num_workers=30,
    prefetch_factor=100,
    persistent_workers=True)

signal_received = False
def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
signal.signal(signal.SIGINT, handle_interrupt)


import importlib, cvae.models.predvae, cvae.models.tokenizer
importlib.reload(cvae.models.predvae)
vocabsize, nlabels = 58, 152
model = cvae.models.predvae.PredVAE(vocabsize, nlabels).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=21, factor=0.1, verbose=True)
epochs, best_trn_loss, best_tst_loss = 100, np.inf, np.inf

def example():
    _ = model.eval()
    # vae = cvae.models.vae.VAE.load(utils.res("brick/vae.pt")).to(device)
    smiles = 'C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O'
    test = torch.tensor(cvae.models.tokenizer.smiles_one_hot(smiles)).to(device).unsqueeze(0)
    test = test.to(device)
    print(smiles)
    decsmi = model(test)
    argmax = torch.argmax(decsmi[0],dim=1)
    print(cvae.models.tokenizer.decode_smiles_from_indexes(argmax[0]))
    decsmi = model(test)
    argmax = torch.argmax(decsmi[0],dim=1)
    print(cvae.models.tokenizer.decode_smiles_from_indexes(argmax[0]))
    
writepath = pathlib.Path("metrics/vaeloss.tsv")
_ = writepath.write_text("epoch\tloss\trecloss\tkloss\n")

for p in model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
    
# 5.4 totloss: 28782 REC: 28459 KL: 322: GRU
# 1.4 totloss: 22199 REC: 21900 KL: 317: LSTM SINGLE DIR
epochs, best_trn_loss, best_tst_loss = 100, np.inf, np.inf
for epoch in range(0,1000):
    
    # eloss, erec_loss, ekl_loss = evaluate()
    
    pbar = tqdm(enumerate(tdl), total=len(tdl), unit="batch")
    _ = model.train()
    epochloss, recloss, kloss, l1loss = 0, 0, 0, 0
    
    chunks = len(tdl)
    
    schedulerloss = []
    # chunk = next(iter(tdl))
    for i,chunk in pbar:
        optimizer.zero_grad()
        
        # inputs for this batch
        insmi = chunk.squeeze(0).to(device)
        
        # outputs and loss
        decoded, z_mean, z_logvar = model(insmi)
        loss, rec_loss, kl_loss = model.loss(decoded, insmi, z_mean, z_logvar)
        
        # update model
        loss.backward()
        optimizer.step()
        
        epochloss = epochloss + loss.item()
        recloss = recloss + rec_loss.item()
        kloss = kloss + kl_loss.item()
        schedulerloss.append(loss.item())
        
        el, rl, kl = epochloss / (i+1), recloss / (i+1), kloss / (i+1)
        msg = f"{epoch} totloss: {el:.2f} "
        msg = msg + f"REC: {rl:.2f} KL: {kl:.2f}"
        pbar.set_description(msg)
        
        if (i+1) % 1000 == 0:
            
            mschedulerloss = np.mean(schedulerloss)
            scheduler.step(mschedulerloss)
            schedulerloss = []
            
            example()
            _ = model.train()
            with open(writepath, "a") as f:
                _ = f.write(f"{epoch}\t{mschedulerloss}\t{rl}\t{kl}\n")
        
        if signal_received:
            break
            
    if signal_received:
        signal_received = False
        break
    
    _ = model.eval()
    example()

    if epochloss < best_trn_loss:
        print('saving!')
        best_trn_loss = epochloss
        # best_tst_loss = eloss
        path = pathlib.Path("brick/pvae.pt")
        torch.save(model.state_dict(), path) 