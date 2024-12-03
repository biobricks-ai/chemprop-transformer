import pandas as pd
import numpy as np
import cvae.models.mixture_experts as moe
import cvae.spark_helpers as H
import torch
import torch.nn
import sqlite3
import itertools
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])

DEVICE = torch.device(f'cuda:0')

class Prediction:
    
    def __init__(self, inchi, property_token, value):
        self.inchi = inchi
        self.value = value
        self.property_token = property_token

class Predictor:
    
    def __init__(self):
        self.dburl = 'brick/cvae.sqlite'
        self.model = moe.MoE.load("brick/moe").to(DEVICE)
        self.tokenizer = self.model.tokenizer
        self.model = torch.nn.DataParallel(self.model)  
        
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row 
        self.all_property_tokens = [r['property_token'] for r in conn.execute("SELECT DISTINCT property_token FROM property")]
        conn.close()
         
    def _get_known_properties(self, inchi, category=None):
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        query = """
        SELECT source, inchi, prop.property_token, prop.data, cat.category, prop_cat.reason, prop_cat.strength, act.value_token, act.value 
        FROM activity act 
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
        
        return pd.DataFrame(res) if len(res) > 0 else pd.DataFrame(columns=['property_token'])
    
    def predict_property_with_randomized_tensors(self, inchi, property_token, seed, num_rand_tensors=1000):
        if property_token not in self.all_property_tokens:
            logging.error(f"Property token {property_token} is not valid")
            return np.nan
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        smiles = H.inchi_to_smiles_safe(inchi)
        selfies = H.smiles_to_selfies_safe(smiles)
        input = torch.LongTensor(self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies))
        input = input.view(1, -1).to(DEVICE)
        
        known_props = self._get_known_properties(inchi)
        known_props = known_props[known_props['property_token'] != property_token]
        teach_force = torch.LongTensor([1, self.tokenizer.SEP_IDX, property_token]).view(1, -1).to(DEVICE)
        value_indexes = list(self.tokenizer.value_indexes().values())
        
        if known_props.empty:
            result_logit = self.model(input, teach_force)[:, -1, value_indexes]
            return torch.softmax(result_logit, dim=1).detach().cpu().numpy()

        property_value_pairs = list(zip(known_props['property_token'], known_props['value_token']))        
        av_flat = torch.LongTensor(list(itertools.chain.from_iterable(property_value_pairs)))
        av_reshaped = av_flat.reshape(av_flat.size(0) // 2, 2)
        
        rand_tensors = []
        for i in range(num_rand_tensors):
            av_shuffled = av_reshaped[torch.randperm(av_reshaped.size(0)), :].reshape(av_flat.size(0))
            av_truncate = av_shuffled[0:8]
            av_sos_trunc = torch.cat([torch.LongTensor([1, self.tokenizer.SEP_IDX]), av_truncate])
            rand_tensor = torch.cat([av_sos_trunc, torch.LongTensor([property_token])])
            rand_tensors.append(rand_tensor)
        
        rand_tensors = torch.stack(rand_tensors).to(DEVICE)
        input = input.repeat(num_rand_tensors, 1)
        print(f"Stacked Random Tensors size: {rand_tensors.size()}")
        
        result_logit = self.model(input, rand_tensors)[:, -1, value_indexes]
        return torch.softmax(result_logit, dim=1).detach().cpu().numpy()
    
    def predict_property(self, inchi, property_token, seed=137):
        value_indexes = list(self.tokenizer.value_indexes().values())
        one_index = value_indexes.index(self.tokenizer.value_id_to_token_idx(1))
        predictions = self.predict_property_with_randomized_tensors(inchi, property_token, seed)
        
        if predictions.size == 0:
            logging.info(f"No predictions generated for InChI: {inchi} and property token: {property_token}")
            return np.nan
        
        return np.mean(predictions[:, one_index], axis=0)
