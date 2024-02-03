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
import os
import threading

DEVICE = torch.device(f'cuda:0')
predict_lock = threading.Lock()

class Predictor():
    
    def __init__(self):
        self.dburl = 'brick/cvae.sqlite'
        self.model = mtt.MultitaskDecoderTransformer.load("brick/working_mtt").to(DEVICE)
        self.tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
        
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row 
        self.all_props = self._get_all_properties()
        self.all_property_tokens = [r['property_token'] for r in conn.execute("SELECT DISTINCT property_token FROM property")]
        conn.close()
    
    def _get_all_properties(self):
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        query = f"""
        SELECT source, prop.property_token, prop.data, cat.category, prop_cat.reason, prop_cat.strength
        FROM property prop
        INNER JOIN source src ON prop.source_id = src.source_id 
        INNER JOIN property_category prop_cat ON prop.property_id = prop_cat.property_id
        INNER JOIN category cat ON prop_cat.category_id = cat.category_id
        """
        
        cursor.execute(query)
        res = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return pd.DataFrame(res)
        
    # return dict with 
    # source, inchi, property_token, data, category, reason, strength, binary_value
    def _get_known_properties(self, inchi, category = None) -> list[dict]:
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        query = f"""
        SELECT source, inchi, prop.property_token, prop.data, cat.category, prop_cat.reason, prop_cat.strength, act.value_token, act.value FROM activity act 
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

        conn.close()
        return res
    
    # returns predictions for one property_token
    # self = Predictor()
    # inchi = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
    # property_token = 3808
    def predict_property(self, inchi, property_token) -> dict:
        smiles = H.inchi_to_smiles_safe(inchi)
        selfies = H.smiles_to_selfies_safe(smiles)
        
        known_props = pd.DataFrame(self._get_known_properties(inchi))
        property_value_pairs = list(zip(known_props['property_token'], known_props['value_token']))

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
        
        property_token_tensor = torch.LongTensor([property_token, self.tokenizer.PAD_IDX, self.tokenizer.END_IDX]).view(1,-1).to(DEVICE)
        av_add_prop = torch.cat([rand_tensors, property_token_tensor.expand(100, -1)], dim=1)
        teach_force = torch.clone(av_add_prop)
        predictions = torch.softmax(self.model(av_add_prop, teach_force)[:, -3, value_indexes], dim=1).detach().cpu().numpy()
        mean_pred = np.mean(predictions[:,one_index], axis=0)
                
        return mean_pred
    
    # returns all predicted properties along with their categories
    # self = Predictor()
    # inchi = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
    # def predict(self, inchi) -> list[dict]:
    #     smiles = H.inchi_to_smiles_safe(inchi)
    #     selfies = H.smiles_to_selfies_safe(smiles)
        
    #     known_props = pd.DataFrame(self._get_known_properties(inchi))
    #     property_value_pairs = list(zip(known_props['property_token'], known_props['value_token']))

    #     results = []
    #     selfies_tokens = torch.LongTensor(self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies))
    #     av_flat = torch.LongTensor(list(chain.from_iterable(property_value_pairs)))
    #     av_reshaped = av_flat.reshape(av_flat.size(0) // 2, 2)
        
    #     rand_tensors = []
    #     for _ in range(100):
    #         av_shuffled = av_reshaped[torch.randperm(av_reshaped.size(0)),:].reshape(av_flat.size(0))
    #         av_truncate = av_shuffled[0:18]
            
    #         av_sos_trunc = torch.cat([torch.LongTensor([self.tokenizer.SEP_IDX]), av_truncate])
    #         selfies_av = torch.hstack([selfies_tokens, av_sos_trunc])
    #         rand_tensors.append(selfies_av)
        
    #     rand_tensors = torch.stack(rand_tensors).to(DEVICE)
    #     value_indexes = list(self.tokenizer.value_indexes().values())
    #     one_index = value_indexes.index(self.tokenizer.value_id_to_token_idx(1))
        
    #     for property_token in tqdm.tqdm(self.all_property_tokens):
    #         property_token_tensor = torch.LongTensor([property_token, self.tokenizer.PAD_IDX, self.tokenizer.END_IDX]).view(1,-1).to(DEVICE)
    #         av_add_prop = torch.cat([rand_tensors, property_token_tensor.expand(100, -1)], dim=1)
    #         teach_force = torch.clone(av_add_prop)
    #         predictions = torch.softmax(self.model(av_add_prop, teach_force)[:, -3, value_indexes], dim=1).detach().cpu().numpy()
    #         mean_pred = np.mean(predictions[:,one_index], axis=0)
    #         results.append({'property_token': property_token, 'probability_of_one': mean_pred})
        
    #     # select propert
    #     results_df = pd.DataFrame(results)
    #     known_df = known_props[['property_token', 'value']]
    #     return_df = self.all_props.merge(results_df, how='left', on='property_token')
    #     return_df = return_df.merge(known_df, how='left', on='property_token')
        
    #     # rename some columns for clarity
    #     return_df = return_df.rename(columns={'value': 'known_value'})
    #     return_df = return_df.rename(columns={'probability_of_one': 'predicted_positive_probability'})
    #     return_df = return_df.rename(columns={'strength': 'strength_of_property_categorization'})
        
    #     return return_df


app = Flask(__name__)
predictor = Predictor()


@app.route('/predict', methods=['GET'])
def predict():
    inchi = request.args.get('inchi')
    if inchi is None:
        return jsonify({'error': 'inchi parameter is required'})
    
    with predict_lock:
        df = predictor.predict(inchi)

    return jsonify(df.to_json(orient="records"))
