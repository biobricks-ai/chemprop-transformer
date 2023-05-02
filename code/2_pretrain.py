import os
import signal
import torch
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from model import MolecularVAE
from modelUtils import vae_loss
from utils import *

torch.manual_seed(42)

signal_received = False

def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
    
signal.signal(signal.SIGINT, handle_interrupt)

def preValidate(model, valid_loader, labels_size):    
    valid_loss = 0
    valid_recons_loss = 0
    valid_kl_loss = 0
    
    for batch_idx, data in tqdm(enumerate(valid_loader)):
        data = data[0].to(device)
        
        labels = torch.full((data.size(0), labels_size), 0.5).to(device)
        
        with torch.no_grad():
            output, mean, logvar = model(data, labels)
            loss, recons_loss, kl_loss = vae_loss(output, data, mean, logvar)
            
            valid_loss += loss.item()
            valid_recons_loss += recons_loss.item()
            valid_kl_loss += kl_loss.item()
            
    valid_loss /= len(valid_loader.dataset)
    valid_recons_loss /= len(valid_loader.dataset)
    valid_kl_loss /= len(valid_loader.dataset)
    
    return valid_loss, valid_recons_loss, valid_kl_loss

def pretrainLoop(model, optimizer, scheduler, train_loader, valid_loader, labels_size, epochs=100):
    best_pre_train_loss = np.inf
    best_valid_loss = np.inf
    checkpoint_count = 0
    
    pre_train_loss_epoch = [] 
    pre_train_recons_loss_epoch = []
    pre_train_kl_loss_epoch = []
    
    pre_valid_loss_epoch = [] 
    pre_valid_recons_loss_epoch = []
    pre_valid_kl_loss_epoch = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        pre_train_loss = 0
        pre_train_recons_loss = 0
        pre_train_kl_loss = 0
        
        for batch_idx, data in tqdm(enumerate(train_loader)):
            data = data[0].to(device)
            
            # labels for pretrain
            labels = torch.full((data.size(0), labels_size), 0.5).to(device)
            
            optimizer.zero_grad()
            output, mean, logvar = model(data, labels)
            
            if batch_idx==0:
                inp = data.cpu().numpy()
                outp = output.cpu().detach().numpy()
                
                model_input = decode_smiles_from_indexes(map(from_one_hot_array, inp[0]), charset)
                
                sampled = outp[0].reshape(1, 120, charset_size).argmax(axis=2)[0]
                model_output = decode_smiles_from_indexes(sampled, charset)
                
                tqdm.write(f"Input:  {model_input}")
                tqdm.write(f"Output: {model_output}")
                
                with open('metrics/pretrain/reconstructions.txt', 'a') as recon_file:
                    recon_file.write(f"Epoch:  {epoch}\n")
                    recon_file.write(f"\tInput:  {model_input}\n")
                    recon_file.write(f"\tOutput:  {model_output}\n")
                    
            loss, recons_loss, kl_loss = vae_loss(output, data, mean, logvar)
            loss.backward()
            pre_train_loss += loss.item()
            pre_train_recons_loss += recons_loss.item()
            pre_train_kl_loss += kl_loss.item()
            optimizer.step()
            
        scheduler.step()
        
        pre_train_loss /= len(train_loader.dataset)
        pre_train_recons_loss /= len(train_loader.dataset)
        pre_train_kl_loss /= len(train_loader.dataset)
        
        valid_loss, valid_recons_loss, valid_kl_loss = preValidate(model, valid_loader, labels_size)
        
        if pre_train_loss < best_pre_train_loss and valid_loss < best_valid_loss:
            best_pre_train_loss = pre_train_loss
            torch.save(model.state_dict(), f'model/pretrain/checkpoint{checkpoint_count}epoch{epoch}pretrained_model.pt')
            checkpoint_count += 1
            
        
        with open('metrics/pretrain/loss.csv', 'a') as loss_file:
            loss_file.write(f'{epoch}, {pre_train_loss}, {pre_train_recons_loss}, {pre_train_kl_loss}, {valid_loss}, {valid_recons_loss}, {valid_kl_loss}\n')

        pre_train_loss_epoch.append(pre_train_loss)
        pre_train_recons_loss_epoch.append(pre_train_recons_loss)
        pre_train_kl_loss_epoch.append(pre_train_kl_loss)
        
        pre_valid_loss_epoch.append(valid_loss)
        pre_valid_recons_loss_epoch.append(valid_recons_loss)
        pre_valid_kl_loss_epoch.append(valid_kl_loss)
        
        
        # plot_losses(pre_train_recons_loss_epoch, pre_train_kl_loss_epoch, 'metrics/pretrain')
        plot_metrics(pre_train_loss_epoch, pre_valid_loss_epoch, 'metrics/pretrain')
        plot_metrics(pre_train_recons_loss_epoch, pre_valid_recons_loss_epoch, 'metrics/pretrain','recon_loss')
        plot_metrics(pre_train_kl_loss_epoch, pre_valid_kl_loss_epoch, 'metrics/pretrain','kl_loss')
        
        if signal_received:
            print('Stopping pretraining...')
            break
    
    
if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataPath='data/ProcessedChemHarmony.h5'
    
    data_train, data_valid, charset, uniqueAssays = load_pretrain_dataset(dataPath)
    data_train = torch.from_numpy(data_train).float()
    data_valid = torch.from_numpy(data_valid).float()
    
    num_epochs = 300
    batch_size=250
    
    charset_size = len(charset)
    labels_size = len(uniqueAssays) + 1
    
    data_train = torch.utils.data.TensorDataset(data_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    
    data_valid = torch.utils.data.TensorDataset(data_valid)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True)
    
    model = MolecularVAE(len_charset=charset_size, labels_size=labels_size, verbose=False).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
    
    os.makedirs('metrics/pretrain', exist_ok=True)
    os.makedirs('model/pretrain', exist_ok=True)
    
    with open('metrics/pretrain/loss.csv', 'w') as loss_file:
        loss_file.write('epoch, loss, recon_loss, kl_loss, valid_loss, valid_recon_loss, valid_kl_loss\n')
        
    with open('metrics/pretrain/reconstructions.txt', 'w') as recon_file:
        recon_file.write("Reconstructions\n")
        
    pretrainLoop(model, optimizer, scheduler, train_loader, valid_loader, labels_size, epochs=300)
        
    
    