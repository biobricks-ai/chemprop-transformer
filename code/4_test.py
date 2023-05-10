import os
import torch
import torch.utils.data
import numpy as np
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from tqdm import tqdm

from model import MolecularVAE
from modelUtils import vae_loss
from utils import *

import json

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

RDLogger.DisableLog('rdApp.*')

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.patches as mpatches

plt.rc('font', size=14)  # controls default text sizes
sns.set(font_scale=1.5)  # Increase the font size scale in Seaborn

def plot_tsne_histogram(reduced_data, activities, save_path):
    tsne_values_per_activity = {activity: {'tsne1': [], 'tsne2': []} for activity in list(sorted(set(activities)))[2:]}
    # tsne_values_per_activity = {activity: {'tsne1': [], 'tsne2': []} for activity in set(activities)}
    

    for (tsne1, tsne2), act in zip(reduced_data, activities):
        if act in tsne_values_per_activity.keys():
            tsne_values_per_activity[act]['tsne1'].append(tsne1)
            tsne_values_per_activity[act]['tsne2'].append(tsne2)

    for k, v in tsne_values_per_activity.items():
        print(k, len(v['tsne1']), len(v['tsne2']))
        
    fig = plt.figure(figsize=(15, 15))  # Adjust the figure size here
    gs = plt.GridSpec(4, 4)

    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    for activity in tsne_values_per_activity:
        sns.kdeplot(x=tsne_values_per_activity[activity]['tsne1'], y=tsne_values_per_activity[activity]['tsne2'], ax=ax_joint, label=f"Activity {activity}")
        sns.kdeplot(x=tsne_values_per_activity[activity]['tsne1'], ax=ax_marg_x)
        sns.kdeplot(y=tsne_values_per_activity[activity]['tsne2'], ax=ax_marg_y)
        
    
    legend_elements = []
    for activity, color in zip(tsne_values_per_activity, sns.color_palette()):
        legend_elements.append(mpatches.Patch(color=color, label=activity))

    ax_joint.legend(handles=legend_elements, loc='upper right')

    ax_joint.set_xlabel('t-SNE1', fontsize=25)
    ax_joint.set_ylabel('t-SNE2', fontsize=25)
    
    ax_joint.tick_params(axis='both', labelsize=20)
    ax_marg_x.tick_params(axis='both', labelsize=20)
    ax_marg_y.tick_params(axis='both', labelsize=20)
    
    ax_marg_x.label_outer()
    ax_marg_y.label_outer()

    plt.legend()
    plt.savefig(save_path)

def plot_tsne(latent_vectors, activities, save_path):
    tsne = TSNE(n_components=2, random_state=42, init='random')
    reduced_data = tsne.fit_transform(latent_vectors)
    
    plot_tsne_histogram(reduced_data, activities, save_path)

def extract_latent_vectors(model, data_loader, device):
    latent_vectors = []
    activities = []
    a = 0
    for batch_idx, data in enumerate(data_loader):
        data, activity, value = data
        data = data.to(device)
        activity = activity.to(device)
        value = value.to(device)

        labels = torch.cat([activity, value], dim=1)

        with torch.no_grad():
            _, mean, logvar = model(data, labels)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            latent_vectors.append(z.cpu().numpy()[batch_idx])
            
            
            for act, val in zip(activity.argmax(axis=1).cpu().tolist(), value.cpu().tolist()):
                label = f'act:{act} val:{int(val[0])}'
                activities.append(label)

    return np.vstack(latent_vectors), activities


def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return None

    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    
    tanimoto_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return tanimoto_sim

def test_model(model, test_loader, device='cpu'):
    model = model.to(device)
    model.eval()

    total_generations = 0
    reconstructions = []
    
    test_loss = 0
    test_recons_loss = 0
    test_kl_loss = 0

    for batch_idx, data in tqdm(enumerate(test_loader)):
        data, activity, value = data
        data = data.to(device)
        activity = activity.to(device)
        value = value.to(device)

        labels = torch.cat([activity, value], dim=1)

        with torch.no_grad():
            output, mean, logvar = model(data, labels)
            loss, recons_loss, kl_loss = vae_loss(output, data, mean, logvar)

            test_loss += loss.item()
            test_recons_loss += recons_loss.item()
            test_kl_loss += kl_loss.item()

            for idx in range(data.size(0)):
                original = data[idx].cpu().numpy()
                recon = output[idx].cpu().numpy()

                ori_smiles = decode_smiles_from_indexes(map(from_one_hot_array, original), charset)
                recon_smiles = decode_smiles_from_indexes(recon.argmax(axis=1), charset)

                tanimoto_sim = tanimoto_similarity(ori_smiles, recon_smiles)
                
                total_generations += 1
                if tanimoto_sim is not None:
                    reconstructions.append({
                        'ori_smiles': ori_smiles,
                        'recon_smiles': recon_smiles,
                        'tanimoto_sim': tanimoto_sim
                    })

    test_loss /= len(test_loader.dataset)
    test_recons_loss /= len(test_loader.dataset)
    test_kl_loss /= len(test_loader.dataset)

    return total_generations, reconstructions, test_loss, test_recons_loss, test_kl_loss


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrainedModelPath = 'models/train/checkpoint23epoch77model.pt'
    dataPath = 'data/processed/ProcessedChemHarmony.h5'
    batch_size = 250

    test_data, charset, uniqueAssays = load_test_dataset(dataPath)
    charset_size = len(charset)
    labels_size = len(uniqueAssays) + 1

    model = MolecularVAE(len_charset=charset_size, labels_size=labels_size, verbose=False).to(device)
    model.load_state_dict(torch.load(pretrainedModelPath))

    test_data, activities_test, values_test = test_data

    test_data = torch.from_numpy(test_data).float()
    activities_test = torch.from_numpy(activities_test).float()
    values_test = torch.from_numpy(values_test).float()

    data_test = torch.utils.data.TensorDataset(test_data, activities_test, values_test)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)
    
    total_generations, reconstructions, test_loss, test_recons_loss, test_kl_loss = test_model(model, test_loader, device)
    
    valid_generations = len(reconstructions)
    avg_tanimoto = sum([rec['tanimoto_sim'] for rec in reconstructions])/len(reconstructions)
    
    os.makedirs('metrics/test', exist_ok=True)
    with open('metrics/test/tanimoto_similarities.json', 'w') as f:
        json.dump({
            'total generations': total_generations,
            'valid generations': valid_generations,
            'avg similarity': avg_tanimoto,
            'avg loss for test': test_loss,
            'avg recon loss for test': test_recons_loss,
            'avg kl loss for test': test_kl_loss,
            'reconstructions': reconstructions
            }, f, indent=2)
    
    print(f"Number of valid generations: {valid_generations}")
    print(f"Test loss: {test_loss:.4f}, Test reconstruction loss: {test_recons_loss:.4f}, Test KL loss: {test_kl_loss:.4f}")
    
    latent_vectors, activities = extract_latent_vectors(model, test_loader, device)
    plot_tsne(latent_vectors, activities, save_path='metrics/test/tsne_plot.png')
