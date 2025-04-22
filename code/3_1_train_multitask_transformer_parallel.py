#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Set MKL environment variables before any imports
import os
# Fix MKL threading layer issue - set only once
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '0'
os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = '1'

# Import numpy first to ensure it picks up the MKL settings
import numpy as np

# Now import everything else
import sys, shutil
import itertools
import pathlib
import tqdm
import torch
import torch.utils.data
import torch.optim as optim
import torch.multiprocessing as mp
import cvae.tokenizer
import cvae.utils as utils
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts

import logging
import time
import datetime

# Set NCCL environment variables
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
os.environ['NCCL_DEBUG'] = 'INFO'  # Set to INFO for debugging
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Use TORCH_ prefix
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_SOCKET_NTHREADS'] = '4'
os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
os.environ['NCCL_BUFFSIZE'] = '2097152'
os.environ['NCCL_SOCKET_IFNAME'] = '' 

# ENABLE IB AND P2P
os.environ['NCCL_IB_DISABLE'] = '0'     # 🔧 MODIFIED
os.environ['NCCL_P2P_DISABLE'] = '0'    # 🔧 MODIFIED

# Enable TF32 precision for A100 GPUs
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
mp.set_start_method("fork", force=True) 

# Set up logging
logdir = pathlib.Path("cache/train_multitask_transformer_parallel/logs")
logdir.mkdir(exist_ok=True, parents=True)
cvae.utils.setup_logging(logdir / "log.txt", logging)

# paths
outdir = pathlib.Path("cache/train_multitask_transformer_parallel")
outdir.mkdir(exist_ok=True)

modeldir = outdir / "models"
modeldir.mkdir(exist_ok=True)

metricsdir = outdir / "metrics"
metricsdir.mkdir(exist_ok=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()



class Trainer():

    def safe_loss_fn(self, parameters, logits, output):
        vocab_size = logits.size(-2)
        output = output.clamp(min=0, max=vocab_size - 1)
        logits = logits.permute(0, 2, 1).contiguous()
        B, T, V = logits.shape
        output_flat = output.reshape(-1)
        logits_flat = logits.reshape(-1, V)
        return (logits_flat[output_flat != self.tokenizer.pad_idx].sum() * 0.0).mean()

    def __init__(self, model, rank, tokenizer, trn_iterator, batch_size, max_epochs=100):
        self.rank = rank
        self.global_step = 0
        self.trn_iterator = trn_iterator

        # init model
        firstbatch = next(iter(trn_iterator))
        model = model.to(rank)
        
        with torch.inference_mode():
            dummy_inp = torch.full((1, 128), fill_value=tokenizer.PAD_IDX, dtype=torch.long, device=rank)
            dummy_teach = torch.full((1, 42), fill_value=tokenizer.PAD_IDX, dtype=torch.long, device=rank)
            res = model(firstbatch[0].to(rank), firstbatch[1].to(rank))  # warm-up forward pass
            
        model = torch.compile(model)
        self.model = DDP(model, device_ids=[rank])
        self.max_epochs = max_epochs

        # Add gradient accumulation for effectively larger batch sizes
        self.gradient_accumulation_steps = 8
        effective_batch_size = batch_size * torch.cuda.device_count() * self.gradient_accumulation_steps  # 16 * 8 * 8
        base_lr = 2e-4
        warmup_steps = 200  # Can be tuned

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.01
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps),
                CosineAnnealingWarmRestarts(self.optimizer, T_0=2000, T_mult=2)
            ],
            milestones=[warmup_steps]
        )

        self.lossfn = self.model.module.build_lossfn()
        # self.lossfn = self.safe_loss_fn
        self.metrics_path = None
        self.best_loss = np.inf
        self.tokenizer = tokenizer
        self.eval_loss = mt.MultitaskTransformer.build_eval_lossfn(value_indexes=self.tokenizer.value_indexes().values(), device=self.rank)
        
        
        # Update GradScaler initialization with device parameter
        self.scaler = GradScaler()
        
        # Reduce evaluation frequency
        self.eval_every = 2000
    
    def log(self, msg):
        if self.rank==0:
            logging.info(msg)
    
    def set_model_savepath(self, savepath):
        self.savepath = pathlib.Path(savepath)
        self.savepath.mkdir(exist_ok=True)
        return self

    def set_trn_iterator(self, iterator):
        self.trn_iterator = iterator
        return self

    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        return self

    def set_mask_percent(self, mask_percent):
        self.mask_percent = mask_percent
        return self

    def set_metrics_file(self, metrics_path, overwrite=False):
        if self.rank == 0:
            self.metrics_path = metrics_path
            if overwrite:
                with open(self.metrics_path, 'w') as f:
                    f.write("type\tbatch\tloss\tlr\n")
        return self

    def _evaluation_loss(self):
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        dist.barrier()
        for i, (inp, teach, out) in enumerate(self.valdl):
            inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):  # Use mixed precision for evaluation too
                    pred = self.model(inp, teach)
                    pred = pred.permute(0, 2, 1).contiguous()
                    loss = self.eval_loss(self.model.module.parameters(), pred, out)
                total_loss += loss.item() * inp.size(0)
                num_samples += inp.size(0)
        
        # Aggregate the total loss and number of samples across all ranks
        total_loss_tensor = torch.tensor(total_loss, device=self.rank)
        num_samples_tensor = torch.tensor(num_samples, device=self.rank)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)
        
        if num_samples_tensor.item() != 0:
            mean_loss = total_loss_tensor.item() / num_samples_tensor.item()
        else:
            mean_loss = 0.
        return mean_loss

    def _train_batch(self, inp, teach, out):
        # Only zero gradients at the beginning of accumulation cycle
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
            
        inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
        assert teach.max().item() < self.tokenizer.vocab_size, "teach has invalid index"
        assert out.max().item() < self.tokenizer.vocab_size, "out has invalid index"

        with autocast(device_type='cuda', dtype=torch.float16):
            pred = self.model(inp, teach) # [batch_size, seq_len, vocab_size]
            pred = pred.permute(0, 2, 1).contiguous()
            loss = self.lossfn(self.model.module.parameters(), pred, out)
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Scale gradients and accumulate
        self.scaler.scale(loss).backward()
        
        # Only update weights at the end of accumulation cycle
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        
        self.global_step += 1
        return loss.detach().item() * self.gradient_accumulation_steps
    
    def _evaluate_balanced_accuracy(self):
        self.model.eval()
        selected_preds = []
        selected_targets = []

        value_token_ids = set(self.tokenizer.value_indexes().values())

        dist.barrier()
        for i, (inp, teach, out) in enumerate(self.valdl):
            inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(inp, teach)
                pred = pred.argmax(dim=2)  # [batch_size, seq_len]

                # Create mask where out == any value token ID
                mask = torch.isin(out, torch.tensor(list(value_token_ids), device=out.device))

                # Use the mask to extract only relevant preds/targets
                selected_preds.append(pred[mask])
                selected_targets.append(out[mask])

        if len(selected_preds) == 0:
            return 0.0

        try:
            all_preds = torch.cat(selected_preds, dim=0)
            all_targets = torch.cat(selected_targets, dim=0)

            # 🛡️ Limit max tokens per rank (to avoid excessive payloads)
            max_tokens = 5000
            if all_preds.size(0) > max_tokens:
                all_preds = all_preds[:max_tokens]
                all_targets = all_targets[:max_tokens]

            # 🛡️ Gather from all ranks
            gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
            gathered_targets = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]

            dist.all_gather(gathered_preds, all_preds)
            dist.all_gather(gathered_targets, all_targets)

            if self.rank == 0:
                all_preds = torch.cat(gathered_preds, dim=0)
                all_targets = torch.cat(gathered_targets, dim=0)

                unique_classes = torch.unique(all_targets)
                class_accuracies = []

                for cls in unique_classes:
                    mask = all_targets == cls
                    if mask.sum() > 0:
                        acc = (all_preds[mask] == all_targets[mask]).float().mean()
                        class_accuracies.append(acc)

                balanced_accuracy = torch.stack(class_accuracies).mean().item()
                return balanced_accuracy

            return 0.0

        except Exception as e:
            if self.rank == 0:
                logging.warning(f"Balanced accuracy evaluation failed: {e}")
            return -1.0


    def start(self):
        
        # Initialize accumulated loss
        iter = 0
        for epoch in range(self.max_epochs):
            self.trn_iterator.sampler.set_epoch(epoch)
            
            self.log(f"Starting epoch {epoch}")
            
            self.model.train()  # Ensure model is in training mode

            for i, (inp, teach, out) in enumerate(self.trn_iterator):
                iter = iter + 1
                self.log(f"Batch {iter}")
                loss = self._train_batch(inp, teach, out)
                self.log(f"Loss: {loss}")
                
                # Evaluate less frequently to speed up training
                if iter % self.eval_every == 0:
                    # Ensure all processes are synced before evaluation
                    torch.cuda.synchronize()
                    dist.barrier()
                    
                    self.model.eval()  # Switch to eval mode
                    with torch.no_grad():  # Prevent gradient computation during eval
                        eval_loss = self._evaluation_loss()
                        bac = self._evaluate_balanced_accuracy()
                    
                    if self.rank == 0:
                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.model.module.save(self.savepath)
                        
                        # also just save periodically in case of crash
                        modelpath = modeldir / f"step_{self.global_step}"
                        modelpath.mkdir(exist_ok=True)

                        self.model.module.save(modelpath)

                        with open(self.metrics_path, 'a') as f:
                            f.write(f"eval\t{i}\t{eval_loss:.4f}\t{self.optimizer.param_groups[0]['lr']}\n")
                            logging.info(f"Epoch: {epoch}, Step: {i}, Train Loss: {loss:.4f}, "
                                f"Eval Loss: {eval_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']}, "
                                f"BAC: {bac:.4f}")
                    
                    self.model.train()  # Switch back to training mode
                    dist.barrier()  # Ensure all processes are synced before continuing

                # Log training metrics less frequently
                if iter % 10 == 0 and self.rank == 0:
                    with open(self.metrics_path, 'a') as f:
                        f.write(f"train\t{i}\t{loss:.4f}\t{self.optimizer.param_groups[0]['lr']}\n")
                        logging.info(f"Epoch: {epoch}, Step: {i}, Train Loss: {loss:.4f}")

def main(rank, world_size):

    setup(rank, world_size)
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    
    # Set sharing strategy for multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Load model
    
    # Main: 0.2348, Balance: 0.7670, Diversity: 0.0000, Total loss: 1.0018
    # v1 model = me.MoE(tokenizer, num_experts=32, hdim=32*8, dim_feedforward=32*8, nhead=4, balance_loss_weight=.1, expert_layers=6)
    
    # updated diversity loss
    # use top-1 expert hard gating
    # reduce num experts, increase expert size
    # v3 model = me.MoE(tokenizer, num_experts=8, hdim=512, dim_feedforward=2048, nhead=8, balance_loss_weight=.1, expert_layers=8)
    # model = me.MoE(tokenizer, num_experts=8, hdim=512, dim_feedforward=2048, nhead=8, balance_loss_weight=.1, expert_layers=8)
    
    # v4 model = me.MoE(tokenizer, num_experts=8, hdim=512, dim_feedforward=2048, nhead=8, balance_loss_weight=.1, expert_layers=8)
    model = me.MoE(tokenizer, num_experts=16, hdim=512, dim_feedforward=2048, nhead=8, balance_loss_weight=0.1, expert_layers=8)

    # model = me.MoE.load(outdir / "moe")
    # model.balance_loss_weight = 1.0

    # model = me.MoE.load("brick/moe")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Optimize batch size for A100s - increase to utilize GPU memory better
    batch_size = 16*4  # Increased from 64
    omp_threads = int(os.environ.get('OMP_NUM_THREADS', 10))
    world_size = torch.cuda.device_count()  # 8 on your machine
    
    cpus_per_rank = os.cpu_count() // world_size  # 240 // 8 = 30
    train_workers = max(2, cpus_per_rank)
    val_workers = max(1, cpus_per_rank)

    prefetch_factor = 100

    trnds = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/trn", tokenizer, nprops=20)
    trndl = torch.utils.data.DataLoader(
        trnds, batch_size=batch_size, shuffle=False, 
        num_workers=train_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            trnds, num_replicas=world_size, rank=rank, drop_last=True)
    )

    valds = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/tst", tokenizer, nprops=20)
    valdl = torch.utils.data.DataLoader(
        valds, batch_size=batch_size, shuffle=False, 
        num_workers=val_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            valds, num_replicas=world_size, rank=rank, drop_last=True)
    )

    trainer = Trainer(model, rank, tokenizer, trndl, batch_size=batch_size, max_epochs=10000)\
        .set_validation_dataloader(valdl)\
        .set_mask_percent(0.1)\
        .set_model_savepath(outdir / "moe")\
        .set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)
    
    if rank == 0:
        trainer.log(f"{len(trndl)} train batches")
        trainer.log(f"Process {rank} has {len(valdl)} batches to validate.")
        trainer.log(f"{num_params/1e6:.2f} million params")
        trainer.log(f"Gradient accumulation steps: {trainer.gradient_accumulation_steps}")
        trainer.log(f"Effective batch size: {batch_size * world_size * trainer.gradient_accumulation_steps}")
        
    trainer.start()

    cleanup()

    if rank == 0:
        if os.path.exists("brick/moe"):
            shutil.rmtree("brick/moe")
        shutil.copytree(outdir / "moe", "brick/moe")

if __name__ == "__main__":
    import sys
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"RANK {rank} starting with WORLD_SIZE {world_size}")
    main(rank, world_size)
    # world_size = torch.cuda.device_count()
    # torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)