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

def trainLoop(model, optimizer, scheduler, train_loader, labels_size, epochs=100, valid_loader = None):
    bestLoss = np.inf
    checkpointCount = 0
    
    pre_train_loss_epoch = [] 
    pre_train_recons_loss_epoch = []
    pre_train_kl_loss_epoch = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        pre_train_loss = 0
        pre_train_recons_loss = 0
        pre_train_kl_loss = 0
        
        for batch_idx, data in tqdm(enumerate(train_loader)):
            embedding = data[0].to(device)
            activity = data[1].to(device)
            value = data[2].to(device)
            
            data = embedding
            
            # print(activity.shape)
            # print(value.shape)
            
            labels = torch.cat([activity, value], dim = 1)
            # print(labels.shape)
            
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
                
                with open('metrics/train/reconstructions.txt', 'a') as recon_file:
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
        
        pre_train_loss /= len_dataset
        pre_train_recons_loss /= len_dataset
        pre_train_kl_loss /= len_dataset
        if bestLoss > pre_train_loss:
            bestLoss = pre_train_loss
            torch.save(model.state_dict(), f'model/train/checkpoint{checkpointCount}epoch{epoch}model.pt')
            checkpointCount += 1
            
        
        with open('metrics/train/loss.csv', 'a') as loss_file:
            loss_file.write(f'{epoch}, {pre_train_loss}, {pre_train_recons_loss}, {pre_train_kl_loss}\n')

        pre_train_loss_epoch.append(pre_train_loss)
        pre_train_recons_loss_epoch.append(pre_train_recons_loss)
        pre_train_kl_loss_epoch.append(pre_train_kl_loss)
        
        plot_losses(pre_train_recons_loss_epoch, pre_train_kl_loss_epoch, 'metrics/train')
        
        if signal_received:
            print('Stopping training...')
            break


if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataPath = 'data/ProcessedChemHarmony.h5'
    pretrainedModelPath = 'model/pretrain/checkpoint99epoch150pretrained_model.pt'
    
    train_elements, test_elements, charset, uniqueAssays = load_train_dataset(dataPath)
    
    data_train, activities_train, values_train = train_elements
    
    data_train = torch.from_numpy(data_train).float()
    activities_train = torch.from_numpy(activities_train).float()
    values_train = torch.from_numpy(values_train).float()
    
    num_epochs = 300
    
    batch_size=250
    charset_size = len(charset)
    labels_size = len(uniqueAssays) + 1
    print(labels_size)
    
     
    
    
    data_train = torch.utils.data.TensorDataset(data_train, activities_train, values_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    
    len_dataset = len(train_loader.dataset)
    
    model = MolecularVAE(len_charset=charset_size, labels_size=labels_size, verbose=False).to(device)
    
    model.load_state_dict(torch.load(pretrainedModelPath))
    print(f'model loaded: {pretrainedModelPath}')
    
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
    
    os.makedirs('metrics/train', exist_ok=True)
    os.makedirs('model/train', exist_ok=True)
    
    with open('metrics/train/loss.csv', 'w') as loss_file:
        loss_file.write('epoch, loss, recon_loss, kl_loss\n')
        
    with open('metrics/train/reconstructions.txt', 'w') as recon_file:
        recon_file.write("Reconstructions\n")
        
    trainLoop(model, optimizer, scheduler, train_loader, labels_size, epochs=300)