import json
import threading
import pandas as pd
import numpy as np
import cvae.models.mixture_experts as moe
import cvae.spark_helpers as H
import torch
import torch.nn
import sqlite3
import itertools
import logging
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])

DEVICE = torch.device(f'cuda:0')

@dataclass
class Category:
    category: str
    reason: str
    strength: str

@dataclass
class Property:
    property_token: int
    source: str
    title: str
    metadata: dict
    categories: list[Category]

@dataclass
class Prediction:
    inchi: str
    property_token: int
    property: Property
    value: float

class Predictor:
    
    def __init__(self):
        self.dburl = 'brick/cvae.sqlite'
        self.dblock = threading.Lock()
        self.model = moe.MoE.load("brick/moe").to(DEVICE)
        self.tokenizer = self.model.tokenizer
        self.model = torch.nn.DataParallel(self.model)  
        
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row 
        self.all_property_tokens = [r['property_token'] for r in conn.execute("SELECT DISTINCT property_token FROM property")]
        self.property_map = self.build_property_map()
        conn.close()
    
    def build_property_map(self):
        with self.dblock:
            conn = sqlite3.connect(self.dburl)
            conn.row_factory = lambda cursor, row: dict((cursor.description[i][0], value) for i, value in enumerate(row))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.property_token, p.title, p.data as metadata, s.source, c.category, pc.reason, pc.strength
                FROM property p
                INNER JOIN property_category pc ON p.property_id = pc.property_id 
                INNER JOIN category c ON pc.category_id = c.category_id
                INNER JOIN source s ON p.source_id = s.source_id
            """)
            res = cursor.fetchall()
            
            # Group results by property_token
            property_map = {}
            for property_token, group in itertools.groupby(res, key=lambda x: x['property_token']):
                group_list = list(group)
                categories = [Category(category=r['category'], reason=r['reason'], strength=r['strength']) 
                            for r in group_list]
                
                property = Property(property_token=property_token,
                                  title=group_list[0]['title'],
                                  metadata=json.loads(group_list[0]['metadata']),
                                  source=group_list[0]['source'],
                                  categories=categories)
                                  
                property_map[property_token] = property
                
            return property_map
    
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
    
    def predict_property(self, inchi, property_token, seed=137, num_rand_tensors=1000):
        value_indexes = list(self.tokenizer.value_indexes().values())
        one_index = value_indexes.index(self.tokenizer.value_id_to_token_idx(1))
        predictions = self.predict_property_with_randomized_tensors(inchi, property_token, seed, num_rand_tensors=num_rand_tensors)
        
        if predictions.size == 0:
            logging.info(f"No predictions generated for InChI: {inchi} and property token: {property_token}")
            return np.nan
        
        token_property = self.property_map.get(property_token, None)
        meanpred = float(np.mean(predictions[:, one_index], axis=0))
        prediction = Prediction(inchi=inchi, property_token=property_token, property=token_property, value=meanpred)
        return prediction
    
    def _build_random_tensors(self, inchi, seed, num_rand_tensors):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        smiles = H.inchi_to_smiles_safe(inchi)
        selfies = H.smiles_to_selfies_safe(smiles)
        input = torch.LongTensor(self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies))
        input = input.view(1, -1)
        
        known_props = self._get_known_properties(inchi)
        teach_force = torch.LongTensor([1, self.tokenizer.SEP_IDX]).view(1, -1)
        
        if known_props.empty:
            return input, teach_force

        property_value_pairs = list(zip(known_props['property_token'], known_props['value_token']))        
        av_flat = torch.LongTensor(list(itertools.chain.from_iterable(property_value_pairs)))
        av_reshaped = av_flat.reshape(av_flat.size(0) // 2, 2)
        
        rand_tensors = []
        for _ in range(num_rand_tensors):
            av_shuffled = av_reshaped[torch.randperm(av_reshaped.size(0)), :].reshape(av_flat.size(0))
            av_truncate = av_shuffled[0:8]
            av_sos_trunc = torch.cat([torch.LongTensor([1, self.tokenizer.SEP_IDX]), av_truncate])
            rand_tensors.append(av_sos_trunc)
        
        rand_tensors = torch.stack(rand_tensors)
        print(f"Stacked Random Tensors size: {rand_tensors.size()}")
        
        return input, rand_tensors
    
    # test with inchi=InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H
    def predict_all_properties(self, inchi, seed=137, max_num_rand_tensors=100) -> list[Prediction]:
        input, rand_tensors_raw = self._build_random_tensors(inchi, seed, max_num_rand_tensors)
        
        # Remove duplicate rows from rand_tensors
        rand_tensors = torch.unique(rand_tensors_raw, dim=0)
        num_rand_tensors = rand_tensors.size(0)

        value_indexes = list(self.tokenizer.value_indexes().values())
        one_index = value_indexes.index(self.tokenizer.value_id_to_token_idx(1))
        
        # Pre-build tensors for all property tokens
        all_property_tokens_tensor = torch.LongTensor(self.all_property_tokens)
        repeated_property_tokens = all_property_tokens_tensor.repeat_interleave(num_rand_tensors).unsqueeze(1)
        
        # Repeat rand_tensors for all properties
        repeated_rand_tensors = rand_tensors.repeat(len(self.all_property_tokens), 1)
        
        # Concatenate rand_tensors with property tokens
        all_prop_tensors = torch.cat([repeated_rand_tensors, repeated_property_tokens], dim=1)

        # Create a TensorDataset and DataLoader for efficient batching
        simultaneous_properties = 100
        batch_size = simultaneous_properties * num_rand_tensors
        dataset = TensorDataset(all_prop_tensors)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        raw_preds = []
        # batch = next(iter(dataloader))
        for batch in tqdm(dataloader):
            prop_tensors_batch = batch[0].to(DEVICE)
            batch_input = input.repeat(prop_tensors_batch.size(0), 1).to(DEVICE)
            # Model inference
            with torch.no_grad():  # Disable gradients for inference
                result_logit = self.model(batch_input, prop_tensors_batch)[:, -1, value_indexes]
                batch_preds = torch.softmax(result_logit, dim=1)[:, one_index]
                # Calculate mean predictions
                batch_preds_mean = batch_preds.view(-1, num_rand_tensors).mean(dim=1)
                raw_preds.extend(batch_preds_mean.detach().cpu().numpy())
            
        raw_preds = [float(x) for x in raw_preds]
        property_tokens = [self.all_property_tokens[i] for i in range(len(raw_preds))]
        properties = [self.property_map.get(property_token, None) for property_token in property_tokens]
        preds = [Prediction(inchi=inchi, property_token=property_tokens[i], property=properties[i], value=raw_preds[i]) for i in range(len(raw_preds))]
        return preds
