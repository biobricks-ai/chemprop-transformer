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
import uuid

DEVICE = torch.device(f'cuda:0')
predict_lock = threading.Lock()
cvaesql = sqlite3.connect('brick/cvae.sqlite')
cvaesql.row_factory = sqlite3.Row  # This enables column access by name

psqlite = sqlite3.connect('flask_cvae/predictions.sqlite')
# create predictions table property_token, property_title, inchi, value
cmd = "CREATE TABLE IF NOT EXISTS prediction (inchi TEXT, property_token INTEGER, value float)"
psqlite.execute(cmd)
cmd = "CREATE INDEX IF NOT EXISTS idx_inchi_property_token ON prediction (inchi, property_token)"
psqlite.execute(cmd)

                    
class Prediction():
    
    def __init__(self, inchi, property_token, value):
        self.inchi = inchi
        self.value = value
        self.property_token = property_token
    
    @staticmethod
    def save(inchi, property_token, value):
        cmd = "INSERT INTO prediction (inchi, property_token, value) VALUES (?, ?, ?)"
        psqlite.execute(cmd, (inchi, property_token, value))
        psqlite.commit()
    
    @staticmethod
    def get(inchi, property_token):
        cmd = "SELECT value FROM prediction WHERE inchi = ? AND property_token = ?"
        res = psqlite.execute(cmd, (inchi, property_token)).fetchone()
        if res:
            return Prediction(inchi, property_token, res[0])  # Return the found prediction
        return None  # Return None if no prediction was found
        
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
    def predict_property(self, inchi, property_token, seed=137) -> dict:
        # Set the seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        smiles = H.inchi_to_smiles_safe(inchi)
        selfies = H.smiles_to_selfies_safe(smiles)
        
        known_props = pd.DataFrame(self._get_known_properties(inchi))
        property_value_pairs = list(zip(known_props['property_token'], known_props['value_token']))

        selfies_tokens = torch.LongTensor(self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies))
        av_flat = torch.LongTensor(list(chain.from_iterable(property_value_pairs)))
        av_reshaped = av_flat.reshape(av_flat.size(0) // 2, 2)
        
        num_rand_tensors = 1000
        rand_tensors = []
        for _ in range(num_rand_tensors):
            av_shuffled = av_reshaped[torch.randperm(av_reshaped.size(0)),:].reshape(av_flat.size(0))
            av_truncate = av_shuffled[0:18]
            
            av_sos_trunc = torch.cat([torch.LongTensor([self.tokenizer.SEP_IDX]), av_truncate])
            selfies_av = torch.hstack([selfies_tokens, av_sos_trunc])
            rand_tensors.append(selfies_av)
        
        rand_tensors = torch.stack(rand_tensors).to(DEVICE)
        value_indexes = list(self.tokenizer.value_indexes().values())
        one_index = value_indexes.index(self.tokenizer.value_id_to_token_idx(1))
        
        property_token_tensor = torch.LongTensor([property_token, self.tokenizer.PAD_IDX, self.tokenizer.END_IDX]).view(1,-1).to(DEVICE)
        av_add_prop = torch.cat([rand_tensors, property_token_tensor.expand(num_rand_tensors, -1)], dim=1)
        teach_force = torch.clone(av_add_prop)
        predictions = torch.softmax(self.model(av_add_prop, teach_force)[:, -3, value_indexes], dim=1).detach().cpu().numpy()
        mean_pred = np.mean(predictions[:,one_index], axis=0)
                
        return mean_pred
    
    def cached_predict_property(self, inchi, property_token):
        prediction = Prediction.get(inchi, property_token)
        if prediction is not None: 
            return prediction.value
        
        prediction = float(self.predict_property(inchi, property_token))
        Prediction.save(inchi, property_token, prediction)
        return prediction

app = Flask(__name__)
predictor = Predictor()

@app.route('/predict', methods=['GET'])
def predict():
    inchi = request.args.get('inchi')
    property_token = request.args.get('property_token', None)
    if inchi is None or property_token is None:
        return jsonify({'error': 'inchi and property token parameters are required'})
    
    with predict_lock:
        mean_value = float(predictor.predict_property(inchi, int(property_token)))

    return jsonify({"inchi":inchi, "property_token":property_token, "positive_prediction":mean_value})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    