from __future__ import print_function
from tqdm import tqdm

import json, os, signal, dvc.api, numpy as np, pathlib, time
import torch, torch.nn.functional as F, torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, Subset

import utils, CVAE.model


torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)

signal_received = False

def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
    
signal.signal(signal.SIGINT, handle_interrupt)

def cvaeLoss(x, xhat, z_mean, z_logvar, beta=0.7):
    x1, xpred = x.view(-1), xhat.view(-1, xhat.shape[-1])
    RECON = F.cross_entropy(xpred, x1)
    KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()) * beta
    KLD = KLD / x.shape[0]
    return RECON + KLD, RECON.item(), KLD.item()

def evaluate(model, valds):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    evalLoss, reconLoss, kldLoss = [], [], []
    eval = torch.utils.data.DataLoader(valds, batch_size=2048)
    pbar = tqdm(eval, ncols=100, unit="batch")
    
    for _, tensors in enumerate(pbar):
        insmi, inlbl, inval  = [x.to(device) for x in tensors]
        
        with torch.no_grad():
            xhat, z_mean, z_logvar = model(insmi, inlbl, inval)
            loss, recon, kld = cvaeLoss(insmi, xhat, z_mean, z_logvar)
            evalLoss.append(loss.item())
            reconLoss.append(recon)
            kldLoss.append(kld)
            pbar.set_description('Eval Loss: {:.4f}'.format(loss.item()))
            
        if signal_received:
            print('Stopping actual validation step...')
            break
    
    evalLoss = sum(evalLoss) / len(evalLoss)
    reconLoss = sum(reconLoss) / len(reconLoss)
    kldLoss = sum(kldLoss) / len(kldLoss)    
    
    return evalLoss, reconLoss, kldLoss
    

def train(model, optimizer, scheduler, outdir, epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trnds = torch.load(utils.res("data/processed/train.pt"))
    valds = torch.load(utils.res("data/processed/validation.pt"))
    
    metricsdir = pathlib.Path(utils.res("metrics/train"))
    checkpt_id, best_trn_loss, best_val_loss = 0, np.inf, np.inf
    
    nbatch = 2048
    trnds = Subset(trnds,range(500000))
    trn = torch.utils.data.DataLoader(
        trnds, batch_size=nbatch, shuffle=True, num_workers=24,
        pin_memory=True, persistent_workers=True)
    
    for epoch in range(epochs):
        
        print('{}/{}'.format(epoch+1, epochs))
        epochLoss, reconLoss, kldLoss = [], [], []
        pbar = tqdm(trn, ncols=100, unit="batch")
        model.train()
            
        for _, tensors in enumerate(pbar):
                
            # inputs for this batch
            insmi, inlbl, inval  = [x.to(device) for x in tensors]

            # outputs and loss
            xhat, z_mean, z_logvar = model(insmi, inlbl, inval)
            loss, recon, kld = cvaeLoss(insmi, xhat, z_mean, z_logvar)
            
            # record losses
            epochLoss.append(loss.item())
            reconLoss.append(recon)
            kldLoss.append(kld)
            
            # update model
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            if signal_received:
                print('Stopping actual train step...')
                break
            
        scheduler.step()
        if signal_received:
            print('Stopping training...')
            signal_received = False
            break
        
        epochLoss = sum(epochLoss)/len(epochLoss)
        reconLoss = sum(reconLoss)/len(reconLoss)
        kldLoss = sum(kldLoss)/len(kldLoss)
        
        evalLoss, evalReconLoss, evalKldLoss = evaluate(model, valds)
        
        with open('{}/loss.tsv'.format(metricsdir), 'a+') as f:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                epoch, epochLoss, evalLoss, kldLoss, evalKldLoss, reconLoss, evalReconLoss))
            
        if epochLoss < best_trn_loss and evalLoss < best_val_loss:
            best_trn_loss = epochLoss
            best_val_loss = evalLoss
            print("saving model...")
            torch.save(model.state_dict(),'{}/checkpoint_{}_epoch_{}.pt'.format(outdir, checkpt_id, epoch))
            torch.save(model.state_dict(),'{}/bestModel.pt'.format(outdir))
            
            checkpt_id += 1

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    params = utils.loadparams()
    
    metricsdir = utils.res(params['training']['metricsOutPath'])
    processdir = utils.res(params['training']['processedData'])
    outdir = utils.res(params['training']['outModelFolder'])
    epochs = params['training']['epochs']
    
    metricsdir.mkdir(parents=True, exist_ok=True)
    processdir.mkdir(parents=True, exist_ok=True)
    
    (metricsdir / 'loss.tsv').write_text('epoch\ttLoss\teLoss\tkldtloss\tkldeloss\trecontloss\treconeloss\n')
    
    model_info = json.loads((processdir / 'modelInfo.json').read_text())
        
    smiles_padlength = model_info['smiles_padlength']
    smiles_vocabsize = model_info['smiles_vocabsize']
    assays_vocabsize = model_info['assays_vocabsize'] + 1
    latentdim = params['latentDim']
    model = CVAE.model.CVAE(smiles_padlength, smiles_vocabsize, assays_vocabsize, latentdim)
    model = model.to(device)
    
    # smilesinput, labelsinput, valueinput = trn[1:5][0], trn[1:5][1], trn[1:5][2]
    # smilesinput,labelsinput,valueinput=smilesinput.to(device),labelsinput.to(device),valueinput.to(device)
    # res = model.forward(smilesinput, labelsinput, valueinput)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
    
    train(model, optimizer, scheduler, outdir, epochs)
    