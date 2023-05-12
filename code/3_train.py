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

def validate(model, valid_loader):    
    valid_loss = 0
    valid_recons_loss = 0
    valid_kl_loss = 0
    
    for batch_idx, data in tqdm(enumerate(valid_loader)):
        data, activity, value = data
        data = data.to(device)
        activity = activity.to(device)
        value = value.to(device)
        
        labels = torch.cat([activity, value], dim = 1)
        
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
    

def trainLoop(model, optimizer, scheduler, train_loader, valid_loader, epochs=100):
    best_train_loss = np.inf
    best_valid_loss = np.inf
    checkpoint_count = 0
    
    train_loss_epoch = [] 
    train_recons_loss_epoch = []
    train_kl_loss_epoch = []
    
    valid_loss_epoch = [] 
    valid_recons_loss_epoch = []
    valid_kl_loss_epoch = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_recons_loss = 0
        train_kl_loss = 0
        
        for batch_idx, data in tqdm(enumerate(train_loader)):
            data, activity, value = data
            
            data = data.to(device)
            activity = activity.to(device)
            value = value.to(device)
            
            # embedding = data[0].to(device)
            # activity = data[1].to(device)
            # value = data[2].to(device)
            
            # data = embedding
            
            labels = torch.cat([activity, value], dim = 1)
            
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
            train_loss += loss.item()
            train_recons_loss += recons_loss.item()
            train_kl_loss += kl_loss.item()
            optimizer.step()
            
        scheduler.step()
        
        train_loss /= len_dataset
        train_recons_loss /= len_dataset
        train_kl_loss /= len_dataset
        
        valid_loss, valid_recons_loss, valid_kl_loss = validate(model, valid_loader)
        
        # if epochLoss < bestTrainLoss and evalLoss < bestEvalLoss:
        
        if train_loss < best_train_loss and valid_loss < best_valid_loss:
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'models/train/checkpoint{checkpoint_count}epoch{epoch}model.pt')
            torch.save(model.state_dict(), f'models/train/LastModel.pt')
            checkpoint_count += 1
            
        
        with open('metrics/train/loss.csv', 'a') as loss_file:
            loss_file.write(f'{epoch}, {train_loss}, {train_recons_loss}, {train_kl_loss}, {valid_loss}, {valid_recons_loss}, {valid_kl_loss}\n')

        train_loss_epoch.append(train_loss)
        train_recons_loss_epoch.append(train_recons_loss)
        train_kl_loss_epoch.append(train_kl_loss)
        
        valid_loss_epoch.append(valid_loss)
        valid_recons_loss_epoch.append(valid_recons_loss)
        valid_kl_loss_epoch.append(valid_kl_loss)
        
        # plot_metrics(train_recons_loss_epoch, train_kl_loss_epoch, 'metrics/train')
        plot_metrics(train_loss_epoch, valid_loss_epoch, 'metrics/train')
        plot_metrics(train_recons_loss_epoch, valid_recons_loss_epoch, 'metrics/train','recon_loss')
        plot_metrics(train_kl_loss_epoch, valid_kl_loss_epoch, 'metrics/train','kl_loss')
        
        
        if signal_received:
            print('Stopping training...')
            break


if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataPath = 'data/processed/ProcessedChemHarmony.h5'
    pretrainedModelPath = 'models/pretrain/LastPretrainedModel.pt'
    num_epochs = 100
    batch_size=250
    
    train_elements, valid_elements, charset, uniqueAssays = load_train_dataset(dataPath)
    
    data_train, activities_train, values_train = train_elements
    
    data_train = torch.from_numpy(data_train).float()
    activities_train = torch.from_numpy(activities_train).float()
    values_train = torch.from_numpy(values_train).float()
    
    
    data_valid, activities_valid, values_valid = valid_elements
    
    data_valid = torch.from_numpy(data_valid).float()
    activities_valid = torch.from_numpy(activities_valid).float()
    values_valid = torch.from_numpy(values_valid).float()
    
    charset_size = len(charset)
    labels_size = len(uniqueAssays) + 1
    
    data_train = torch.utils.data.TensorDataset(data_train, activities_train, values_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    
    data_valid = torch.utils.data.TensorDataset(data_valid, activities_valid, values_valid)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True)
    
    len_dataset = len(train_loader.dataset)
    
    model = MolecularVAE(len_charset=charset_size, labels_size=labels_size, verbose=False).to(device)
    
    model.load_state_dict(torch.load(pretrainedModelPath))
    print(f'model loaded: {pretrainedModelPath}')
    
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
    
    os.makedirs('metrics/train', exist_ok=True)
    os.makedirs('models/train', exist_ok=True)
    
    with open('metrics/train/loss.csv', 'w') as loss_file:
        loss_file.write('epoch, loss, recon_loss, kl_loss, valid_loss, recon_valid_loss, kl_valid_loss,\n')
        
    with open('metrics/train/reconstructions.txt', 'w') as recon_file:
        recon_file.write("Reconstructions\n")
        
    trainLoop(model, optimizer, scheduler, train_loader, valid_loader, epochs=100)