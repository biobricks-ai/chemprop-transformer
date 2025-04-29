#!/usr/bin/env python
# -*- coding: utf-8 -*-
# torchrun --standalone --nproc-per-node=8 --master-port=29500 code/3_1_train_multitask_transformer_parallel.py 2> cache/train_multitask_transformer_parallel/logs/err.log

import os
import sys
import shutil
import pathlib
import logging
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

import cvae.tokenizer
import cvae.utils
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
from lion_pytorch import Lion
from helper.trainer import Trainer

# Environment variables setup
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_IB_DISABLE'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'

# Enable TF32 for A100s
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
mp.set_start_method("fork", force=True)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def init_logging(rank):
    if rank == 0:
        logdir = pathlib.Path("cache/train_multitask_transformer_parallel/logs")
        logdir.mkdir(exist_ok=True, parents=True)
        cvae.utils.setup_logging(logdir / "log.txt", logging)
        logging.info("Logging initialized.")

def main(rank, world_size):
    setup(rank, world_size)
    init_logging(rank)

    outdir = pathlib.Path("cache/train_multitask_transformer_parallel")
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"

    logging.info(f"Rank {rank} starting setup.")
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    # Epoch: 0, Step: 6700, Train Loss (last cycle): 0.3774, Eval Loss: 0.3448, BAC: 0.7082, AUC: 0.8653, LR: 0.000020
    # model = me.MoE(tokenizer, num_experts=16, k=4, hdim=256, dim_feedforward=1024, nhead=4, balance_loss_weight=0.1, expert_layers=6)
    # batch_size = 100

    # 2025-04-27 19:51:22 - INFO - Epoch: 0, Step: 1700, Train Loss (last cycle): 2.1533, Eval Loss: 0.4625, BAC: 0.5000, AUC: 0.7146, LR: 0.000078
    # model = me.MoE(tokenizer, num_experts=16, k=4, hdim=512, dim_feedforward=2048, nhead=4, balance_loss_weight=0.1, expert_layers=6)
    # batch_size = 25

    # 2025-04-28 08:57:34 - INFO - Rank 0: Evaluation complete. Loss: 0.5168, AUC: 0.7997, BAC: 0.6644
    # model = me.MoE(tokenizer, num_experts=18, k=4, hdim=512, dim_feedforward=2048, nhead=4, balance_loss_weight=0.1, expert_layers=6)
    # batch_size = 25

    # 2025-04-28 20:12:00 - INFO - Epoch: 0, Step: 6000, Train Loss (last cycle): 0.4948, Eval Loss: 0.4219, BAC: 0.7163, AUC: 0.8700, LR: 0.000100
    # model = me.MoE(tokenizer, num_experts=24, k=4, hdim=512, dim_feedforward=2048, nhead=4, balance_loss_weight=0.1, diversity_loss_weight=1e-4, expert_layers=6)
    batch_size = 50
    model = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    cpus_per_rank = max(2, (os.cpu_count() or 8) // world_size)
    train_workers = max(2, min(4, cpus_per_rank))
    val_workers = max(1, min(2, cpus_per_rank))

    trnds = mt.FastPackedSequenceShiftDataset("cache/pack_multitask_tensors/packed_trn")
    valds = mt.FastPackedSequenceShiftDataset("cache/pack_multitask_tensors/packed_tst")

    trndl = torch.utils.data.DataLoader(
        trnds, batch_size=batch_size, shuffle=False, num_workers=train_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=20,
        sampler=torch.utils.data.distributed.DistributedSampler(trnds, num_replicas=world_size, rank=rank, drop_last=True)
    )

    valdl = torch.utils.data.DataLoader(
        valds, batch_size=batch_size, shuffle=False, num_workers=val_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=20,
        sampler=torch.utils.data.distributed.DistributedSampler(valds, num_replicas=world_size, rank=rank, drop_last=True)
    )

    trainer = Trainer(model, rank, tokenizer, trndl, batch_size=batch_size, max_epochs=10000)
    trainer.set_validation_dataloader(valdl)
    trainer.set_mask_percent(0.1)
    trainer.set_model_savepath(modeldir / "moe")
    trainer.set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)

    if rank == 0:
        trainer.log(f"{len(trndl)} train batches")
        trainer.log(f"{len(valdl)} validation batches")
        trainer.log(f"{num_params/1e6:.2f} million parameters")
        trainer.log(f"Gradient accumulation: {trainer.gradient_accumulation_steps}")

    trainer.start()
    cleanup()

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"RANK {rank} starting with WORLD_SIZE {world_size}")
    main(rank, world_size)
