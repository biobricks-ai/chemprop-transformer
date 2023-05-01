import matplotlib.pyplot as plt

import numpy as np
import h5py

def one_hot_array(i, n):
    return map(int, [ix == i for ix in xrange(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x].decode("utf-8"), vec)).strip()


def load_pretrain_dataset(filename):
    with h5py.File(filename, 'r') as h5f:
        data_train = h5f['data_pretrain'][:]
        data_test = h5f['data_pretrain_test'][:]
        charset =  h5f['charset'][:]
        uniqueAssays =  h5f['uniqueAssays'][:]
        
    return data_train, data_test, charset, uniqueAssays

def load_train_dataset(filename):
    with h5py.File(filename, 'r') as h5f:
        data_train = h5f['data_train'][:]
        data_train_activities = h5f['data_train_activities'][:]
        data_train_values = h5f['data_train_values'][:]
        
        data_test = h5f['data_test'][:]
        data_test_activities = h5f['data_test_activities'][:]
        data_test_values = h5f['data_test_values'][:]
        
        charset =  h5f['charset'][:]
        uniqueAssays =  h5f['uniqueAssays'][:]
        
    return (data_train, data_train_activities, data_train_values), (data_test, data_test_activities, data_test_values), charset, uniqueAssays

def load_valid_dataset(filename):
    with h5py.File(filename, 'r') as h5f:
        data_valid = h5f['data_valid'][:]
        data_valid_activities = h5f['data_valid_activities'][:]
        data_valid_values = h5f['data_valid_values'][:]
        
        charset =  h5f['charset'][:]
        uniqueAssays =  h5f['uniqueAssays'][:]
        
    return (data_valid, data_valid_activities, data_valid_values),  charset, uniqueAssays
    

def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    uniqueAssays =  h5f['uniqueAssays'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset, uniqueAssays)
    else:
        return (data_test, charset, uniqueAssays)
    
    
def plot_losses(recon_losses, kl_losses, out_path):
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Reconstruction', color='tab:red')
    ax1.plot(recon_losses, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('KL', color='tab:blue')
    ax2.plot(kl_losses, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    plt.savefig(f'{out_path}/loss.png')