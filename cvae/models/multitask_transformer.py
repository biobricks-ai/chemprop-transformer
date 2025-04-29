import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data
import json
from torch import Tensor
from typing import Tuple
from torch.utils.data import Dataset

import math
import pathlib
import tqdm
import bisect

from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer
import cvae.utils

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable embeddings for the positions
        self.positional_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # Create a tensor of position indices [0, 1, 2, ..., sequence_length-1]
        position_indices = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))
        
        # Retrieve the positional embeddings for the position indices
        position_embeddings = self.positional_embeddings(position_indices)
        
        # Add the positional embeddings to the input embeddings
        x = x + position_embeddings
        
        return self.dropout(x)

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

    # def forward(self, x):
    #     # x shape: [batch_size, sequence_length, d_model]
    #     # Update positional encoding to match batch size and sequence length
    #     x = x + self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)
    #     return self.dropout(x)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
#     """ Generate the attention mask for causal decoding """
#     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
#     mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
#     return mask

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a standard causal mask (upper triangular with -inf above diag)"""
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

# def generate_custom_subsequent_mask(sz: int) -> torch.Tensor:
#     """ Generate a custom attention mask for causal decoding with specific unmasked positions """
#     mask = generate_square_subsequent_mask(sz)
    
#     for i in range(1, sz-1, 2):
#         mask[i,i+1] = 0.0
    
#     return mask

@torch.no_grad()
def generate_custom_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    mask = torch.full((sz, sz), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    for i in range(1, sz - 1, 2):
        mask[i, i + 1] = 0.0  # allow peeking
    return mask


class MultitaskTransformer(nn.Module):
    
    def __init__(self, tokenizer, hdim=256, nhead=2, num_layers=4, dim_feedforward=256, dropout_rate=0.1, output_size=None):
        
        super().__init__()
        
        self.output_size = tokenizer.vocab_size if output_size is None else output_size
        self.hdim = hdim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.tokenizer = tokenizer
        self.token_pad_idx = tokenizer.PAD_IDX
        
        self.embedding = nn.Embedding(tokenizer.vocab_size, self.hdim)
        self.outemb = nn.Embedding(tokenizer.vocab_size, self.hdim)        
        
        self.positional_encoding_inp = PositionalEncoding(self.hdim)
        # self.positional_encoding_inp = LearnedPositionalEncoding(self.hdim)

        self.positional_encoding_out = PositionalEncoding(self.hdim)
        # self.positional_encoding_out = LearnedPositionalEncoding(self.hdim)
        
        self.embedding_norm = nn.LayerNorm(self.hdim)
        
        encode_args = {"d_model": self.hdim, "nhead": self.nhead, "dim_feedforward": self.dim_feedforward, "dropout": dropout_rate}
        encode_layer = nn.TransformerEncoderLayer(**encode_args, batch_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_layers=num_layers)
        
        decode_args = {"d_model": self.hdim, "nhead": self.nhead, "dim_feedforward": self.dim_feedforward, "dropout": dropout_rate}
        decode_layer = nn.TransformerDecoderLayer(**decode_args, batch_first=True)
        self.decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(self.hdim)
        
        self.classification_layers = nn.Sequential(
            nn.LayerNorm(self.hdim),
            nn.Linear(self.hdim, self.hdim),
            nn.ReLU(),
            nn.Linear(self.hdim, self.output_size)
        )


    def forward(self, input, teach_forcing):

        memory_mask = input == self.token_pad_idx
        
        input_embedding = self.positional_encoding_inp(self.embedding_norm(self.embedding(input)))
        input_encoding = self.encoder(input_embedding, src_key_padding_mask=memory_mask)
        
        teach_forcing = self.positional_encoding_out(self.outemb(teach_forcing))
        tgt_mask = generate_custom_subsequent_mask(teach_forcing.size(1), device=input.device)
        
        decoded = self.decoder(teach_forcing, input_encoding, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        decoded = self.decoder_norm(decoded)
        
        logits = self.classification_layers(decoded)
        
        return logits


    @staticmethod
    def lossfn(ignore_index=-100, weight_decay=1e-5):
        ce_lossfn = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index, label_smoothing=0.01)

        def lossfn(parameters, logits, output):
            loss = ce_lossfn(logits, output)
            return loss

        return lossfn
    
    @staticmethod
    def build_eval_lossfn(value_indexes, device):
        value_token_ids = torch.tensor(list(value_indexes), dtype=torch.long).to(device)
        ce_lossfn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100, label_smoothing=0.01)

        def lossfn(parameters, logits, output):
            """
            logits: [B, V, T]
            output: [B, T]
            """
            B, V, T = logits.shape
            logits = logits.permute(0, 2, 1).contiguous()  # → [B, T, V]

            logits_flat = logits.reshape(-1, V)  # [B*T, V]
            output_flat = output.reshape(-1)     # [B*T]

            mask = torch.isin(output_flat, value_token_ids)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=output.device)

            logits_sel = logits_flat[mask]       # [N, V]
            output_sel = output_flat[mask]       # [N]

            return ce_lossfn(logits_sel, output_sel)

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


@torch.jit.script
def process_assay_vals(raw_assay_vals: Tensor, pad_idx: int, sep_idx: int, end_idx: int, nprops: int) -> Tuple[Tensor, Tensor]:
    mask = raw_assay_vals != pad_idx
    assay_vals = raw_assay_vals[mask][1:-1]
    reshaped = assay_vals.view(-1, 2).contiguous()

    # Safety check
    assert reshaped.numel() > 0, "No assay values found."

    perm = torch.randperm(reshaped.size(0))
    shuffled = reshaped[perm].flatten()
    av_truncate = shuffled[: nprops * 2]

    device = raw_assay_vals.device
    av_sos_eos = torch.cat([
        torch.tensor([sep_idx], device=device),
        av_truncate,
        torch.tensor([end_idx], device=device)
    ])
    pad_value = float(pad_idx)
    out = F.pad(av_sos_eos, (0, nprops * 2 + 2 - av_sos_eos.size(0)), value=pad_value)
    tch = torch.cat([torch.tensor([1]), out[:-1]])
    
    return tch, out


class SequenceShiftDataset(Dataset):
    def __init__(self, path, tokenizer: SelfiesPropertyValTokenizer, nprops=5, assay_filter=[]):
        self.nprops = nprops
        self.assay_filter = assay_filter
        self.data = []
        self.cumulative_lengths = [0]
        cumulative_length = 0
        self.tokenizer = tokenizer
        self.pad_idx, self.sep_idx, self.end_idx = tokenizer.PAD_IDX, tokenizer.SEP_IDX, tokenizer.END_IDX

        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt")):
            file_data = torch.load(file_path, map_location="cpu")
            self.data.append((file_data["selfies"], file_data["assay_vals"]))
            cumulative_length += file_data["selfies"].size(0)
            self.cumulative_lengths.append(cumulative_length)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        local_idx = idx - self.cumulative_lengths[file_idx]

        selfies_raw = self.data[file_idx][0][local_idx]
        raw_assay_vals = self.data[file_idx][1][local_idx]

        tch, out = process_assay_vals(
            raw_assay_vals,
            self.pad_idx,
            self.sep_idx,
            self.end_idx,
            self.nprops
        )

        return selfies_raw, tch, out

class FastPackedSequenceShiftDataset(Dataset):

    def __init__(self, path_prefix):
        meta = json.load(open(f"{path_prefix}_meta.json"))
        def load_tensor(name):
            shape = tuple(meta[name]["shape"])
            dtype = getattr(torch, meta[name]["dtype"].split('.')[-1])  # e.g. "torch.int64" → torch.int64
            numel = int(np.prod(shape))
            return torch.from_file(
                filename=f"{path_prefix}_{name}.dat",
                shared=False,
                size=numel,
                dtype=dtype
            ).view(shape)


        self.selfies = load_tensor("selfies")
        self.tch = load_tensor("tch")
        self.out = load_tensor("out")

    def __len__(self):
        return self.selfies.size(0)

    def __getitem__(self, idx):
        return self.selfies[idx], self.tch[idx], self.out[idx]

class LabelSmoothingCrossEntropySequence(nn.Module):
    def __init__(self, epsilon_ls=0.05, ignore_index=None):
        super(LabelSmoothingCrossEntropySequence, self).__init__()
        self.epsilon_ls = epsilon_ls
        self.ignore_index = ignore_index

    def forward(self, out, tgt):
        num_classes = out.size(-1)
        dev = out.device
        
        if tgt.dim() == 1:
            # Masked case: (num_tokens,)
            fill = self.epsilon_ls / (num_classes - 1)
            with torch.no_grad():
                smooth_label = torch.full(size=(tgt.size(0), num_classes), fill_value=fill, device=dev)
                tgt = tgt.unsqueeze(-1)
                smooth_label.scatter_(-1, tgt, 1.0 - self.epsilon_ls)

                if self.ignore_index is not None:
                    ignore_mask = tgt.eq(self.ignore_index)
                    smooth_label.masked_fill_(ignore_mask, 0.0)

            loss = -smooth_label * F.log_softmax(out, dim=-1)

            if self.ignore_index is not None:
                loss.masked_fill_(ignore_mask, 0.0)

            loss = loss.sum(dim=-1)
            loss = loss.mean()

        else:
            # Regular case: (batch_size, seq_len)
            batch_size, seq_len = tgt.size()
            fill = self.epsilon_ls / (num_classes - 1)
            with torch.no_grad():
                smooth_label = torch.full(size=(batch_size, seq_len, num_classes), fill_value=fill, device=dev)
                tgt = tgt.unsqueeze(-1)
                smooth_label.scatter_(-1, tgt, 1.0 - self.epsilon_ls)

                if self.ignore_index is not None:
                    ignore_mask = tgt.eq(self.ignore_index)
                    smooth_label.masked_fill_(ignore_mask, 0.0)

            loss = -smooth_label * F.log_softmax(out, dim=-1)

            if self.ignore_index is not None:
                loss.masked_fill_(ignore_mask, 0.0)

            loss = loss.sum(dim=-1)
            loss = loss.masked_select(~ignore_mask.squeeze(-1)).mean()

        return loss

class NoamLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, model_size, warmup_steps, last_epoch=-1, multiplier=1.0, max_lr=5e-5, min_lr=1e-6):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.multiplier = multiplier
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        scale = self.model_size ** (-0.5) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        lrs = [self.max_lr * self.multiplier * scale for base_lr in self.base_lrs]
        lrs = [min(lr, self.max_lr) for lr in lrs]
        lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs
