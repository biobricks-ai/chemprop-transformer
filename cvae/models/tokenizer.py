from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch, numpy as np

charset = [" ", "(", ".", "0", "2", "4", "6", "8", "@", "B", "F", "H", "L", 
           "N", "P", "R", "T", "V", "X", "Z", "\\", "b", "d", "l", "n", "p", 
           "r", "t", "#", "%", ")", "+", "-", "/", "1", "3", "5", "7", "9", "=", 
           "A", "C", "G", "I", "K", "M", "O", "S", "[", "]", "a", "c", "e", "g", 
           "i", "o", "s", "u"]

nchar = len(charset)
char_to_index = {char: i for i, char in enumerate(charset)}

def smiles_one_hot(smiles):
    cropped = list(smiles.ljust(120))
    indices = np.array([char_to_index[c] for c in cropped])
    one_hot = np.eye(nchar, dtype=np.float32)[indices].T
    if one_hot.shape != (nchar,120):
        raise ValueError(f"Expected output shape {(nchar,120)}, but got {one_hot.shape}")
    return one_hot

def valid_smiles(smiles):
    is_mol = Chem.MolFromSmiles(smiles) is not None
    valid_char = set(smiles).issubset(set(charset))
    valid_length = len(smiles) <= 120
    return is_mol and valid_char and valid_length

def smiles_to_adjacency(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    # Create an empty atom feature vector
    atom_vector = np.zeros((num_atoms,))

    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        
        # This is a simple example, replace with your actual atom features
        atom_vector[i] = atom.GetAtomicNum()

      
def smiles_to_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    # Generate the Morgan fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    fingerprint_np = np.zeros((nBits,), dtype=np.uint8)
    for bit in fingerprint.GetOnBits():
        fingerprint_np[bit] = 1
        
    return fingerprint_np

def decode_smiles_from_indexes(vec):
    return "".join(map(lambda x: charset[x], vec)).strip()