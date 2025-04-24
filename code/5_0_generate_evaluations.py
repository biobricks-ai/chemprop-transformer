# torchrun --nproc-per-node=8 --master-port=29500 distributed_eval_torchrun.py 
import os, itertools, uuid, pathlib, shutil, logging
import pandas as pd, torch, numpy as np, tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import cvae.tokenizer
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
import cvae.utils
from cvae.tokenizer import SelfiesPropertyValTokenizer

import glob
import pandas as pd

# Setup output directories
outdir = pathlib.Path("cache/generate_evaluations")
outdir.mkdir(exist_ok=True, parents=True)

tmpdir = outdir / "temp"
tmpdir.mkdir(exist_ok=True, parents=True)
[f.unlink() for f in tmpdir.glob('*')]

metdir = outdir / "metrics"
metdir.mkdir(exist_ok=True, parents=True)
[f.unlink() for f in metdir.glob('*')]

logdir = outdir / "logs"
logdir.mkdir(exist_ok=True, parents=True)

# Set up logging
cvae.utils.setup_logging(logdir / "log.txt", logging)
logging.info("Starting evaluation generation")

def log(msg, rank):
    if rank == 0:
        logging.info(msg)

nprops = 5
batch_size = 5
perm_indices = list(itertools.permutations(range(nprops)))
perm_count = len(perm_indices)

def run_eval(i, raw_inp, raw_out, model, tokenizer, device):
    inp, raw_out = raw_inp.to(device), raw_out.to(device)

    x = torch.sum(torch.isin(raw_out, tokenizer.value_indexes_tensor), dim=1) >= nprops
    chemical_id = torch.where(x)[0] + (i * batch_size)
    inp, trunc_out = inp[x], raw_out[x, 1:(2 * nprops + 1)].reshape(-1, nprops, 2)
    if inp.size(0) == 0:
        return pd.DataFrame()
    perm_out = torch.cat([trunc_out[:, list(perm), :] for perm in perm_indices], dim=0).reshape(-1, nprops * 2)
    sep_tensor = torch.full((perm_out.size(0), 1), tokenizer.SEP_IDX, device=device)
    out = torch.cat([sep_tensor, perm_out, torch.zeros_like(sep_tensor)], dim=1)
    teach = torch.cat([torch.ones_like(sep_tensor), out[:, :-1]], dim=1)
    rep_inp = inp.repeat(perm_count, 1)
    with torch.no_grad():
        prob = torch.softmax(model(rep_inp, teach), dim=2)
    
    assays_mask = torch.isin(out, tokenizer.assay_indexes_tensor)
    assays = out[assays_mask]

    values_mask = torch.isin(out, tokenizer.value_indexes_tensor)
    values = out[values_mask]
    prob_vals = torch.argmax(prob, dim=2)[values_mask]
    rawprobs = prob[values_mask][:, tokenizer.value_indexes_tensor]
    probs = (rawprobs / rawprobs.sum(dim=1, keepdim=True))[:, 1]

    assays, values, assays, prob_vals, probs = map(lambda x: x.cpu().numpy(), [assays, values, assays, prob_vals, probs])
    position = np.tile(np.arange(nprops), chemical_id.size(0) * perm_count)
    chemical_id = torch.repeat_interleave(chemical_id, perm_count)
    chemical_id = torch.repeat_interleave(chemical_id, nprops).cpu().numpy()
    assays_reshaped = assays.reshape(-1, nprops).astype(str)
    prior_assays = [' + '.join(assays_reshaped[i, :j + 1]) for i in range(len(assays_reshaped)) for j in range(nprops)]

    values_reshaped = values.reshape(-1, nprops).astype(str)
    prior_values = [values_reshaped[i, :j + 1] for i in range(len(values_reshaped)) for j in range(nprops)]
    return pd.DataFrame({
        'batch': i,
        'chemical_id': chemical_id,
        'prior_assays': prior_assays,
        'prior_values': prior_values,
        'assay': assays,
        'value': values,
        'probs': probs,
        'nprops': position,
        'prob_assays': assays,
        'prob_vals': prob_vals
    })

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(rank, world_size)
    torch.cuda.set_device(local_rank)
    # device = torch.device("cuda") # for testing
    device = torch.device(f"cuda:{local_rank}")

    log(f"Loading model on rank {rank}", rank)
    model: me.MoE = me.MoE.load("brick/moe").to(device)
    model.eval()
    model = DDP(model, device_ids=[local_rank])

    # tokenizer = model.tokenizer # for testing
    tokenizer: SelfiesPropertyValTokenizer = model.module.tokenizer
    tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()), device=device)
    tokenizer.value_indexes_tensor = torch.tensor(list(tokenizer.value_indexes().values()), device=device)

    # dataset = mt.FastPackedSequenceShiftDataset("cache/pack_multitask_tensors/packed_hld")
    dataset = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/hld", tokenizer, nprops=nprops)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) # for testing
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=24, pin_memory=True, prefetch_factor=20)

    log(f"Starting evaluation loop with {len(dataloader)} batches", rank)

    seen_inputs = set()
    batch_accum = []

    testloader = enumerate(dataloader)
    i, (raw_inp, _, raw_out) = next(testloader)
    for i, (raw_inp, _, raw_out) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        log(f"Processing batch {i} out of {len(dataloader)}", 0)
        batch_tuples = tuple(map(lambda x, y: (tuple(x.tolist()), tuple(y.tolist())), raw_inp, raw_out))
        new_inputs_mask = [t not in seen_inputs for t in batch_tuples]
        seen_inputs.update(batch_tuples)
        if any(new_inputs_mask):
            new_raw_inp = raw_inp[new_inputs_mask]
            new_raw_out = raw_out[new_inputs_mask]
            # i, raw_inp, raw_out, model, tokenizer, device = 0, new_raw_inp, new_raw_out, model, tokenizer, device
            batch_df = run_eval(i, new_raw_inp, new_raw_out, model, tokenizer, device)
            if not batch_df.empty:
                batch_accum.append(batch_df)
        if len(batch_accum) > 0 and sum(len(df) for df in batch_accum) > 1_000_000:
            log(f"Saving batch accumulation at step {i}", rank)
            pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{uuid.uuid4()}.parquet", index=False)
            batch_accum = []

    if batch_accum:
        log("Saving final batch accumulation", rank)
        pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{uuid.uuid4()}.parquet", index=False)

    cleanup()


if __name__ == "__main__":
    main_worker()
    parquet_files = glob.glob(str(tmpdir / "*.parquet"))
    df = pd.concat([pd.read_parquet(file) for file in parquet_files])
    df.to_parquet(outdir / "multitask_predictions.parquet", index=False, engine="pyarrow", compression="zstd", compression_level=9)