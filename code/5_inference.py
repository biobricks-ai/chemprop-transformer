import torch
import os
import numpy as np
from model import MolecularVAE
from utils import *
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

def decode_random_latent_space(model, labels_size, charset, num_samples=1, num_choices=1, device='cpu', stochastic=True):
    generated_smiles_list = []

    model = model.to(device)
    model.eval()

    for _ in tqdm(range(num_samples)):
        z = torch.randn(1, 292).to(device)
        labels = torch.full((1, labels_size), 0.5).to(device)

        with torch.no_grad():
            decoded_output = model.decode(z, labels)

        output_np = decoded_output.cpu().detach().numpy()
        sample_choices = []

        for _ in range(num_choices):
            if stochastic:
                sampled_indexes = [np.random.choice(charset_size, p=output_np[0, i]) for i in range(output_np.shape[1])]
            else:
                sampled_indexes = np.argmax(output_np, axis=2)[0]

            output_smiles = decode_smiles_from_indexes(sampled_indexes, charset)
            sample_choices.append(output_smiles)

        generated_smiles_list.append(sample_choices)

    return generated_smiles_list

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrainedModelPath = 'models/pretrain/LastPretrainedModel.pt'
    dataPath = 'data/processed/ProcessedChemHarmony.h5'
    num_samples = 10000
    num_choices = 100

    train_elements, valid_elements, charset, uniqueAssays = load_train_dataset(dataPath)
    charset_size = len(charset)
    labels_size = len(uniqueAssays) + 1

    model = MolecularVAE(len_charset=charset_size, labels_size=labels_size, verbose=False).to(device)
    model.load_state_dict(torch.load(pretrainedModelPath))

    generated_smiles_list = decode_random_latent_space(model, labels_size, charset, num_samples, num_choices, device)
    
    validated_smiles = validate_smiles(generated_smiles_list)
    total_generated_smiles = num_samples * num_choices
    print(f"Valid SMILES: {len(validated_smiles)}/{total_generated_smiles}")

    os.makedirs('metrics/inference', exist_ok=True)
    generated_smiles_path = 'metrics/inference/generations.txt'
    validated_smiles_path = 'metrics/inference/validated_generations.txt'
    
    write_smiles_to_file(generated_smiles_list, generated_smiles_path)
    
    write_smiles_to_file(validated_smiles, validated_smiles_path)
