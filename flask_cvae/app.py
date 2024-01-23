from flask import Flask, request, jsonify
import pandas as pd
import cvae.models.multitask_transformer as mtt
import cvae.tokenizer.selfies_tokenizer as st
import cvae.tokenizer.selfies_property_val_tokenizer as spt
import cvae.spark_helpers as H
import numpy as np
import torch
import selfies as sf
import sqlite3
import types
import tqdm
import pandas
from itertools import chain

DEVICE = torch.device(f'cuda:0')
class Predictor():
    
    def __init__(self):
        self.dburl = sqlite3.connect('brick/cvae.sqlite')
        self.model = mtt.MultitaskDecoderTransformer.load("brick/mtransform2").to(DEVICE)
        self.tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
        
        all_property_tokens = [row['property_pytorch_index'] for row in self.db.execute("SELECT DISTINCT property_pytorch_index FROM property")]
        self.all_property_tokens = [self.tokenizer.assay_id_to_token_idx(p) for p in all_property_tokens]
        
    # return dict with 
    # source, inchi, property_token, data, category, reason, strength, binary_value
    def _get_known_properties(self, inchi, category = None) -> list[dict]:
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        query = f"""
        SELECT source, inchi, prop.property_pytorch_index property_token, prop.data, cat.category, prop_cat.reason, prop_cat.strength, act.binary_value FROM activity act 
        INNER JOIN source src ON act.source_id = src.source_id 
        INNER JOIN property prop ON act.property_id = prop.property_id
        INNER JOIN property_category prop_cat ON prop.property_id = prop_cat.property_id
        INNER JOIN category cat ON prop_cat.category_id = cat.category_id
        WHERE inchi = ?"""
        
        params = [inchi]
        if category is not None:
            query += " AND cat.category = ?"
            params.append(category)
            
        cursor.execute(query, params)

        res = [dict(row) for row in cursor.fetchall()]
        for r in res:
            r['property_token'] = self.tokenizer.assay_id_to_token_idx(r['property_token'])
            r['value_token'] = int(self.tokenizer.value_id_to_token_idx(r['binary_value']))

        conn.close()
        return res
        
    # returns all predicted properties along with their categories
    # inchi = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
    def predict(self, inchi) -> list[dict]:
        smiles = H.inchi_to_smiles_safe(inchi)
        selfies = H.smiles_to_selfies_safe(smiles)
        
        known_props = pd.DataFrame(self._get_known_properties(inchi))
        unknown_property_tokens = [p for p in self.all_property_tokens if p not in known_props['property_token'].values]
        
        property_value_pairs = list(zip(known_props['property_token'], known_props['value_token']))

        results = []
        selfies_tokens = torch.LongTensor(self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies))
        av_flat = torch.LongTensor(list(chain.from_iterable(property_value_pairs)))
        av_reshaped = av_flat.reshape(av_flat.size(0) // 2, 2)
        
        rand_tensors = []
        for _ in range(100):
            av_shuffled = av_reshaped[torch.randperm(av_reshaped.size(0)),:].reshape(av_flat.size(0))
            av_truncate = av_shuffled[0:18]
            
            av_sos_trunc = torch.cat([torch.LongTensor([self.tokenizer.SEP_IDX]), av_truncate])
            selfies_av = torch.hstack([selfies_tokens, av_sos_trunc])
            rand_tensors.append(selfies_av)
        
        rand_tensors = torch.stack(rand_tensors).to(DEVICE)
        value_indexes = list(self.tokenizer.value_indexes().values())
        one_index = value_indexes.index(self.tokenizer.value_id_to_token_idx(1))
        for property_token in tqdm.tqdm(unknown_property_tokens):
            property_token_tensor = torch.LongTensor([property_token, self.tokenizer.PAD_IDX, self.tokenizer.END_IDX]).view(1,-1).to(DEVICE)
            av_add_prop = torch.cat([rand_tensors, property_token_tensor.expand(100, -1)], dim=1)
            teach_force = torch.clone(av_add_prop)
            predictions = torch.softmax(self.model(av_add_prop, teach_force)[:, -3, value_indexes], dim=1).detach().cpu().numpy()
            mean_pred = np.mean(predictions[:,one_index], axis=0)
            results.append({'property_token': property_token, 'probability_of_one': mean_pred})
        
        results_df = pd.DataFrame(results)
        return known_props.merge(results_df, how='inner', on='property_token')


app = Flask(__name__)
predictor = Predictor()


@app.route('/predict', methods=['GET'])
def predict():
    inchi = request.args.get('inchi')
    if inchi is None: return jsonify({'error': 'inchi parameter is required'})
    
    df = predictor.predict(inchi)
    return jsonify(df.to_json(orient="records"))

