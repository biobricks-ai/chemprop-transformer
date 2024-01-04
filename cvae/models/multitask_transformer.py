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
    
class MultitaskTransformer(nn.Module):
    
    def __init__(self, tokenizer, selfies_sz=120, hdim=256, nhead=32, num_layers=8, dim_feedforward=500, dropout_rate=0.1):
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
        
        decode_layer = nn.TransformerDecoderLayer(d_model=self.hdim, nhead=self.nhead, 
                                                  dim_feedforward=self.dim_feedforward, 
                                                  batch_first=True, dropout=dropout_rate, norm_first=True)
        self.decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(self.hdim)
        
        self.classification_layer = nn.Linear(self.hdim, self.vocab_size)


    def forward(self, input):
        
        memory_mask = input == self.token_pad_idx
        
        input_embedding = self.positional_encoding(self.embedding(input))
        input_encoding = self.encoder(input_embedding, src_key_padding_mask=input == memory_mask)
        input_encoding = self.encoder_norm(input_encoding)
        
        teach_forcing_embed = input_embedding.clone()
        tgt_mask = generate_static_mask(self.selfies_sz, input.size(1) - self.selfies_sz).to(input.device)
        decoded = self.decoder(teach_forcing_embed, input_encoding, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        decoded = self.decoder_norm(decoded)
        
        logits = self.classification_layer(decoded)
        
        return logits
    
    @staticmethod
    def loss(decsmi, insmi, pad_idx):
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad_idx)
        return  criterion(decsmi, insmi)
    
    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        
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
