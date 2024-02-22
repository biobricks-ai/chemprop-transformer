import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, torch.nn as nn, torch.nn.functional as F
import math
import pathlib
from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer
import cvae.utils

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, d_model]
        # Update positional encoding to match batch size and sequence length
        x = x + self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)
        return self.dropout(x)

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask

def generate_static_mask(selfies_sz: int, assayval_sz:int) -> torch.Tensor:
    sf_sf_mask = generate_square_subsequent_mask(selfies_sz)
    sf_av_mask = torch.zeros((selfies_sz, assayval_sz)) # no mask
    av_sf_mask = torch.zeros((assayval_sz, selfies_sz)) # no mask
    
    av_av_sz = assayval_sz // 2
    av_av_mask = torch.tril(torch.ones((av_av_sz,av_av_sz), dtype=torch.bool)).repeat_interleave(2, dim=1).repeat_interleave(2,dim=0).float()
    av_av_mask = av_av_mask.masked_fill(av_av_mask == 1., float(0.0)).masked_fill(av_av_mask == 0., float("-inf"))
    
    mask = torch.cat([torch.cat([sf_sf_mask, sf_av_mask], dim=1), torch.cat([av_sf_mask, av_av_mask], dim=1)], dim=0) 
    
    return mask
    
class MultitaskDecoderTransformer(nn.Module):
    
    def __init__(self, tokenizer, selfies_sz=120, hdim=512, nhead=16, num_layers=3, dim_feedforward=4096, dropout_rate=0.1):
        super().__init__()
        
        self.selfies_sz = selfies_sz
        self.vocab_size = tokenizer.vocab_size
        self.hdim = hdim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.tokenizer = tokenizer
        self.token_pad_idx = tokenizer.PAD_IDX
        
        self.embedding = nn.Embedding(self.vocab_size, self.hdim)
        self.positional_encoding = PositionalEncoding(self.hdim, dropout=dropout_rate)
        
        decode_args = {"d_model": self.hdim, "nhead": self.nhead, "dim_feedforward": self.dim_feedforward, "dropout": dropout_rate}
        decode_layer = nn.TransformerDecoderLayer(**decode_args, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(self.hdim)
        
        self.classification_layer = nn.Linear(self.hdim, self.vocab_size)


    def forward(self, input, teach_forcing):
        
        input = input[:,0:10]
        memory_mask = input == self.token_pad_idx
        
        input_embedding = self.positional_encoding(self.embedding(input))
        
        teach_forcing = self.positional_encoding(self.embedding(teach_forcing))
        tgt_mask = generate_square_subsequent_mask(teach_forcing.size(1)).to(input.device)
        decoded = self.decoder(teach_forcing, input_embedding, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        decoded = self.decoder_norm(decoded)
        
        logits = self.classification_layer(decoded)
        
        return logits
    
    @staticmethod
    def lossfn(ignore_index = None, weight_decay=1e-5):
        ce_lossfn = nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss(reduction='sum')
        def lossfn(parameters, logits, output):
            ce_loss = ce_lossfn(logits, output)
            # l2 = sum(p.pow(2.0).sum() for p in parameters if p.requires_grad)
            return ce_loss # + weight_decay * l2
        return lossfn    
    
    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        
        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        torch.save(self.state_dict(), path / "mtransformer.pt")
        return path
    
    @staticmethod
    def load(dirpath = pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        model = MultitaskDecoderTransformer(tokenizer)
        model.load_state_dict(torch.load(dirpath / 'mtransformer.pt'))
        model.eval()
        return model

class MultitaskTransformer(nn.Module):
    
    def __init__(self, tokenizer, selfies_sz=120, hdim=128, nhead=2, num_layers=4, dim_feedforward=256, dropout_rate=0.1):
        super().__init__()
        
        self.selfies_sz = selfies_sz
        self.vocab_size = tokenizer.vocab_size
        self.hdim = hdim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.tokenizer = tokenizer
        self.token_pad_idx = tokenizer.PAD_IDX
        
        self.embedding = nn.Embedding(self.vocab_size, self.hdim)
        self.positional_encoding = PositionalEncoding(self.hdim, dropout=dropout_rate)
        
        encode_layer = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=self.nhead, 
                                                  dim_feedforward=self.dim_feedforward, 
                                                  batch_first=True, dropout=dropout_rate, norm_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(self.hdim)
        
        self.output_embedding = nn.Embedding(self.vocab_size, self.hdim)
        decode_layer = nn.TransformerDecoderLayer(d_model=self.hdim, nhead=self.nhead, 
                                                  dim_feedforward=self.dim_feedforward, 
                                                  batch_first=True, dropout=dropout_rate, norm_first=True)
        self.decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(self.hdim)
        
        self.classification_layer = nn.Linear(self.hdim, self.vocab_size)


    def forward(self, input, teach_forcing):
        
        memory_mask = input == self.token_pad_idx
        
        input_embedding = self.positional_encoding(self.embedding(input))
        input_encoding = self.encoder(input_embedding, src_key_padding_mask=input == memory_mask)
        input_encoding = self.encoder_norm(input_encoding)
        
        tgt_mask = generate_static_mask(0, teach_forcing.size(1)).to(input.device)
        teach_forcing_encoding = self.positional_encoding(self.output_embedding(teach_forcing))
        decoded = self.decoder(teach_forcing_encoding, input_encoding, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        decoded = self.decoder_norm(decoded)
        
        logits = self.classification_layer(decoded)
        
        return logits
    
    @staticmethod
    def lossfn(ignore_index = None, weight_decay=1e-5):
        ce_lossfn = nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss(reduction='sum')
        def lossfn(parameters, logits, output):
            ce_loss = ce_lossfn(logits, output)
            # l2 = sum(p.pow(2.0).sum() for p in parameters if p.requires_grad)
            return ce_loss # + weight_decay * l2
        return lossfn    
    
    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        
        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        torch.save(self.state_dict(), path / "mtransformer.pt")
        return path
    
    @staticmethod
    def load(dirpath = pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        model = MultitaskTransformer(tokenizer)
        model.load_state_dict(torch.load(dirpath / 'mtransformer.pt'))
        model.eval()
        return model

import pathlib, tqdm
import torch, torch.utils.data, torch.nn.functional as F

class SequenceShiftDataset(torch.utils.data.Dataset):

    def __init__(self, path, pad_idx, sep_idx, end_idx):
        self.data = []
        self.cumulative_lengths = [0]
        cumulative_length = 0
        self.pad_idx, self.sep_idx, self.end_idx = pad_idx, sep_idx, end_idx

        # file_path = next(pathlib.Path(path).glob("*.pt"))
        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt")):
            file_data = torch.load(file_path)
            
            num_props = file_data['assay_vals'].size(1)
            assay_vals = file_data['assay_vals'][num_props > 9]
            self.data.extend([(file_data['selfies'], file_data['assay_vals'])])
            cumulative_length += file_data['selfies'].size(0)
            self.cumulative_lengths.append(cumulative_length)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        
        # Find which section this index falls into and update the index to be relative to that section
        file_idx = next(i for i, total_length in enumerate(self.cumulative_lengths) if total_length > idx) - 1
        idx -= self.cumulative_lengths[file_idx]
        
        idxdata = self.data[file_idx]
        selfies_raw, raw_assay_vals = idxdata[0][idx], idxdata[1][idx]
        
        # remove padding from selfies
        selfies = selfies_raw[selfies_raw != self.pad_idx]
        
        # assay_val munging - unpad, randomly permute, add sos/eos tokens
        assay_vals = raw_assay_vals[raw_assay_vals != self.pad_idx][1:-1]
        reshaped_av = assay_vals.reshape(assay_vals.size(0) // 2, 2)
        av_shuffled = reshaped_av[torch.randperm(reshaped_av.size(0)),:].reshape(assay_vals.size(0))
        
        # truncate to 10 random features
        av_truncate = av_shuffled[0:20]
        
        # add start and end tokends and pad to 120 length
        av_sos_eos = torch.cat([torch.LongTensor([self.sep_idx]), av_truncate, torch.LongTensor([self.end_idx])])
        
        # create sequence input by stacking selfies + assay_vals and 
        out_raw = torch.hstack([selfies, av_sos_eos])
        
        # add padding up to 150
        out_pad = F.pad(out_raw, (0, 150 - out_raw.size(0)), value=self.pad_idx)
        
        out_shift = torch.hstack([out_pad[1:], torch.tensor([self.pad_idx])])
        
        return selfies_raw, out_pad, out_shift

class PropertyValSequenceShiftDataset(torch.utils.data.Dataset):

    def __init__(self, path, pad_idx, sep_idx, end_idx):
        self.data = []
        self.cumulative_lengths = [0]
        cumulative_length = 0
        self.pad_idx, self.sep_idx, self.end_idx = pad_idx, sep_idx, end_idx

        # file_path = next(pathlib.Path(path).glob("*.pt"))
        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt")):
            file_data = torch.load(file_path)
            self.data.extend([(file_data['selfies'], file_data['assay_vals'])])
            cumulative_length += file_data['selfies'].size(0)
            self.cumulative_lengths.append(cumulative_length)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        
        # Find which section this index falls into and update the index to be relative to that section
        file_idx = next(i for i, total_length in enumerate(self.cumulative_lengths) if total_length > idx) - 1
        idx -= self.cumulative_lengths[file_idx]
        
        idxdata = self.data[file_idx]
        selfies, raw_assay_vals = idxdata[0][idx], idxdata[1][idx]
        
        # assay_val munging - unpad, randomly permute, add sos/eos tokens
        assay_vals = raw_assay_vals[raw_assay_vals != self.pad_idx][1:-1]
        reshaped_av = assay_vals.reshape(assay_vals.size(0) // 2, 2)
        av_shuffled = reshaped_av[torch.randperm(reshaped_av.size(0)),:].reshape(assay_vals.size(0))
        
        # truncate to 10 random features
        av_truncate = av_shuffled[0:10]
        
        # add start and end tokends and pad to 12 length
        padlength = 14
        av_sos_eos = torch.cat([torch.LongTensor([self.sep_idx]), av_truncate, torch.LongTensor([self.end_idx])])
        av_pad = F.pad(av_sos_eos, (0, padlength - av_sos_eos.size(0)), value=self.pad_idx)
        
        # create sequence input by stacking selfies + assay_vals and 
        inp = selfies
        
        # pad by allowing 120 selfies tokens and 60 assays
        out = torch.hstack([av_pad[1:], torch.tensor([self.pad_idx])])
        
        return inp, av_pad, out