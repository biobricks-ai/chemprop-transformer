import pandas as pd, sqlite3, seaborn as sns, matplotlib.pyplot as plt, os, pathlib, numpy as np
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me
from cvae.tokenizer import SelfiesPropertyValTokenizer
import torch
from tqdm import tqdm
tqdm.pandas()

outdir = pathlib.Path("data/property_benchmarks")
outdir.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(f'cuda:0')
model : me.MoE = me.MoE.load("brick/moe").to(DEVICE)
model = torch.nn.DataParallel(model)
tokenizer : SelfiesPropertyValTokenizer = model.module.tokenizer

#%% GET PROPERTIES OF INTEREST ===========================================================
conn = sqlite3.connect('brick/cvae.sqlite')

# get all the property_tokens for tox21 properties
prop_src = pd.read_sql("SELECT property_token,title,source FROM property p INNER JOIN source s on p.source_id = s.source_id", conn)
prop_src = prop_src.groupby('property_token').first().reset_index()

def tensors_to_df(tensor_dir = pathlib.Path("data/tensordataset/multitask_tensors/hld")):
    tensors = [torch.load(file) for file in tqdm(list(tensor_dir.iterdir()))]
    df = pd.DataFrame([
        {'selfies': selfie.tolist(), 'assay_vals': assay_val.tolist()}
        for tensor in tqdm(tensors)
        for selfie, assay_val in zip(tensor['selfies'], tensor['assay_vals'])
    ])

    # check that all the selfies are unique, handle the fact that selfies are lists
    df['selfies_str'] = df['selfies'].progress_apply(lambda x: ' '.join(map(str, x)))
    assert df['selfies_str'].nunique() == len(df), "Selfies are not unique"

    # Process the assay_vals column
    def process_assay_vals(assay_vals):
        # Ignore the first and last values, and remove all zeros
        processed = [val for val in assay_vals[1:] if val != 0]
        processed = processed[:-1]
        
        # Group into pairs of assay and value
        return [(processed[i], processed[i+1]) for i in range(0, len(processed), 2)]

    # Apply the processing function and explode the result
    exploded_df = df.progress_apply(lambda row: pd.Series({
        'selfies_str': row['selfies_str'],
        'selfies': row['selfies'],
        'assay_value_pairs': process_assay_vals(row['assay_vals'])
    }), axis=1).explode('assay_value_pairs')

    # Split the pairs into separate columns
    exploded_df[['assay', 'value']] = pd.DataFrame(exploded_df['assay_value_pairs'].tolist(), index=exploded_df.index)
    exploded_df = exploded_df.drop('assay_value_pairs', axis=1).reset_index(drop=True)

    return exploded_df

trn_df = tensors_to_df(pathlib.Path("data/tensordataset/multitask_tensors/trn"))
hld_df = tensors_to_df(pathlib.Path("data/tensordataset/multitask_tensors/hld"))

trn_df.to_parquet(outdir / "trn_df.parquet")
hld_df.to_parquet(outdir / "hld_df.parquet")

# assert that there is no overlap in selfies_str between trn_df and hld_df
assert not trn_df['selfies_str'].isin(hld_df['selfies_str']).any(), "There is overlap in selfies_str between trn_df and hld_df"

#%% EVALUATE PROPERTIES ===========================================================
trn_df = pd.read_parquet(outdir / "trn_df.parquet")
hld_df = pd.read_parquet(outdir / "hld_df.parquet")

# get some tox21 properties
tox21_props = prop_src[prop_src['source'] == 'tox21']['property_token'].tolist()

# this finds informative, non-redundant features for a given property-token
def find_features_for_properties(property : int):
    # 1. Filter trndf to the given property
    property_df = trn_df[trn_df['assay'] == property]
    
    # 2. Join it with trndf on selfies_str
    joined_df = property_df.merge(trn_df[['selfies_str', 'assay','value']], on='selfies_str', suffixes=('_x', '_y'))
    counts_df = joined_df.groupby(['assay_y']).size().reset_index(name='counts')
    counts_df = counts_df[counts_df['counts'] > 30]
    joined_counts = joined_df.merge(counts_df, on=['assay_y'], how='inner')

    # check that nunique(assay_y) == counts_df.shape[0]
    assert joined_counts['assay_y'].nunique() == counts_df.shape[0], "Assay_y is not unique"

    # 3. Evaluate the entropy of the results by comparing the value_x and value_y
    def calculate_entropy(group):
        probs = group['value_y'].value_counts(normalize=True)
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy

    entropy_df = joined_counts.groupby(['assay_y','counts']).progress_apply(calculate_entropy).reset_index()
    entropy_df.columns = ['assay_y', 'counts','entropy']
    entropy_df = entropy_df.sort_values('entropy', ascending=True)

    non_redundant_features = entropy_df[entropy_df['entropy'] > 0.1]['assay_y']
    
    # 5. Remove features that provide no information
    max_entropy = np.log2(2)  # Entropy of a binary random variable (completely non-informative)
    informative_features = entropy_df[entropy_df['entropy'] <= 0.9 * max_entropy]['assay_y']
    
    # Combine the filters
    useful_features = set(non_redundant_features) & set(informative_features)
    
    return list(useful_features)

def build_selfies_assay_vals_tensors(selfies_str, property_token, features_df, nprops=5, ntensors=1):
    # Filter features_df to the input selfies_str
    filtered_df = features_df[features_df['selfies_str'] == selfies_str]
    
    # Initialize list to store our tensors
    X = []
    
    for _ in range(ntensors):
        # Randomly select a subset of features
        selected_features = filtered_df.sample(n=min(nprops, len(filtered_df)), replace=False)
        
        # Create a tensor for the selected features
        feature_tensor = torch.zeros(nprops*2+4, dtype=torch.long)
        feature_tensor[0] = tokenizer.SEP_IDX
        idx = 1
        for _, row in selected_features.iterrows():
            feature_tensor[idx] = row['assay_y']
            feature_tensor[idx + 1] = row['value_y']
            idx += 2
        
        # Add property_token
        feature_tensor[idx] = property_token
        idx += 1
        
        # Add padding value
        feature_tensor[idx] = tokenizer.PAD_IDX
        idx += 1
        
        feature_tensor[-1] = tokenizer.END_IDX
        
        # Pad the tensor if necessary
        if idx < nprops*2+3:
            feature_tensor[idx:-1] = tokenizer.PAD_IDX
        
        X.append(feature_tensor)
    
    # Convert list to tensor
    X = torch.stack(X)
    
    return X

def evaluate_selfies_predictions(selfies_str, sf_tensor, property_token, features_df, nprops=4, ntensors=1):
    
    av_tensors = build_selfies_assay_vals_tensors(selfies_str, property_token, features_df, nprops, ntensors=10).to(DEVICE)
    rep_sf_tensor = sf_tensor.unsqueeze(0).repeat(av_tensors.shape[0], 1).to(DEVICE)
    
    with torch.no_grad():
        preds = model(rep_sf_tensor, av_tensors)
    
    # get the values of val_index_0 and val_index_1
    val0index, val1index = tokenizer.value_indexes()[0], tokenizer.value_indexes()[1]
    val_logits = preds[:, 11, [val0index,val1index]] 
    val_probs = torch.nn.functional.softmax(val_logits, dim=1)
    mean_pred = val_probs[:,1].mean()
    return mean_pred.item()

# this evaluates a property-token by using the trained model to evaluate it with randomly selected features
def evaluate_property(property : int):
    df = hld_df[hld_df['assay'] == property]
    features = find_features_for_properties(property)
    features_df = hld_df[hld_df['assay'].isin(features)]
    features_df = df.merge(features_df, on='selfies_str')
    
    sf_preds = []
    for index in tqdm(range(len(df))):
        selfies_str = df['selfies_str'].iloc[index]
        sf_tensor = torch.tensor(df['selfies'].iloc[index], dtype=torch.long)
        value = df['value'].iloc[index]
        value = 0 if value == tokenizer.value_indexes()[0] else 1
        pred = evaluate_selfies_predictions(selfies_str, sf_tensor, property, features_df)
        sf_preds.append({'value': value, 'pred': pred})
    
    pdf = pd.DataFrame(sf_preds)[['value','pred']]
    pdf['binary_pred'] = (pdf['pred'] > 0.5).astype(int)
    accuracy = (pdf['value'] == pdf['binary_pred']).mean()
    
    # group by selfies_str, count, and sort descending
    tst = df.groupby('selfies_str').size().reset_index(name='counts')
    tst = tst.sort_values('counts', ascending=False)
    
    return hld_df_property['value'].mean()

tmp = prop_src.merge(shared_props_df, left_on='property_token', right_on='assay_y').sort_values('entropy', ascending=True)
tmp.to_csv('temp.csv', index=False)
print(f"Data has been written to temp.csv")
