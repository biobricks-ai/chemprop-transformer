import sys, os
sys.path.append(os.path.abspath('./'))

import numpy as np
import itertools
import pathlib
import tqdm
import torch
import torch.utils.data
import torch.optim as optim
import cvae.tokenizer
import cvae.utils as utils
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class Trainer():
    def __init__(self, model, rank, tokenizer, max_epochs=10):
        self.rank = rank
        self.model = DDP(model.to(rank), device_ids=[rank])
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
        self.lossfn = mt.MultitaskTransformer.lossfn(ignore_index=tokenizer.pad_idx)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        self.max_epochs = max_epochs
        self.metrics_path = None
        self.best_loss = np.inf
        self.tokenizer=tokenizer
    
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
                    f.write("epoch\tstep\ttype\tloss\n")
        return self

    def _evaluation_loss(self):
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        dist.barrier()
        for i, (inp, teach, out) in enumerate(self.valdl):
            inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
            with torch.no_grad():
                pred = self.model(inp, teach)
                loss = self.lossfn(self.model.module.parameters(), pred.permute(0, 2, 1), out)
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
        self.optimizer.zero_grad()
        inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
        pred = self.model(inp, teach)
        loss = self.lossfn(self.model.module.parameters(), pred.permute(0, 2, 1), out)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def start(self):
        total_batches = len(self.trn_iterator)
        self.validation_interval = total_batches // 4  # Evaluate four times per epoch
        for epoch in range(self.max_epochs):
            self.valdl.sampler.set_epoch(epoch)
            self.trn_iterator.sampler.set_epoch(epoch)  # Ensure randomness in distributed training
            for i, (inp, teach, out) in enumerate(self.trn_iterator):
                loss = self._train_batch(inp, teach, out)
                if self.rank == 0:
                    with open(self.metrics_path, 'a') as f:
                        f.write(f"{epoch}\t{i}\ttrain\t{loss:.4f}\n")
                
                if i % 100 == 0:
                    eval_loss = self._evaluation_loss()
                    self.scheduler.step(eval_loss)
                    
                    if self.rank == 0:
                        if eval_loss < self.best_loss:
                            self.model.module.save(self.savepath)
                            
                        with open(self.metrics_path, 'a') as f:
                            lr = self.optimizer.param_groups[0]['lr']
                            f.write(f"{epoch}\t{i}\teval\t{eval_loss:.4f}\t{lr}\n")
                            print(f"Epoch: {epoch}, Step: {i}, Train Loss: {loss:.4f}, Eval Loss: {eval_loss:.4f}, LR: {lr}")


def main(rank, world_size):
    setup(rank, world_size)
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    model = me.MoE(tokenizer)
    # model = me.MoE.load("brick/moe")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    trnds = mt.SequenceShiftDataset("data/tensordataset/multitask_tensors/trn", tokenizer, nprops=5)
    trndl = torch.utils.data.DataLoader(
        trnds, batch_size=16*8, shuffle=False, num_workers=4, pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(trnds, num_replicas=world_size, rank=rank)
    )
    
    valds = mt.SequenceShiftDataset("data/tensordataset/multitask_tensors/tst", tokenizer, nprops=5)
    valdl = torch.utils.data.DataLoader(valds, batch_size=16*8, shuffle=False, num_workers=0, pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(valds, num_replicas=world_size, rank=rank))
    
    trainer = Trainer(model, rank, tokenizer, max_epochs=10)\
        .set_trn_iterator(trndl)\
        .set_validation_dataloader(valdl)\
        .set_mask_percent(0.1)\
        .set_model_savepath('brick/moe')\
        .set_metrics_file(pathlib.Path("metrics/multitask_loss.tsv"), overwrite=True)
    
    if rank ==0:
        print(f"{len(trndl)} train batches")
        print(f"Process {rank} has {len(valdl)} batches to validate.")
        print(f"{num_params/1e6} million params")
        
    trainer.start()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
