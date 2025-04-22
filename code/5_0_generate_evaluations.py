import itertools, uuid, pathlib, os
import pandas as pd, torch, numpy as np, tqdm
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me
from cvae.tokenizer import SelfiesPropertyValTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

# Setup output directories
outdir = pathlib.Path("cache/generate_evaluations")
temp_dir = outdir / "temp"
metrics_dir = outdir / "metrics"
for d in [outdir, temp_dir, metrics_dir]:
    d.mkdir(exist_ok=True, parents=True)

# Clear temp_dir
for file in temp_dir.iterdir():
    file.unlink()

nprops = 5
batch_size = 20
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
    values_mask = torch.isin(out, tokenizer.value_indexes_tensor)

    assays = out[assays_mask]
    values = out[values_mask]
    prob_assays = torch.argmax(prob, dim=2)[assays_mask]
    prob_vals = torch.argmax(prob, dim=2)[values_mask]
    rawprobs = prob[values_mask][:, tokenizer.value_indexes_tensor]
    probs = (rawprobs / rawprobs.sum(dim=1, keepdim=True))[:, 1]

    assays, values, prob_assays, prob_vals, probs = map(lambda x: x.cpu().numpy(), [assays, values, prob_assays, prob_vals, probs])
    position = np.tile(np.arange(nprops), chemical_id.size(0) * perm_count)

    chemical_id = torch.repeat_interleave(chemical_id, perm_count)
    chemical_id = torch.repeat_interleave(chemical_id, nprops).cpu().numpy()

    assays_reshaped = assays.reshape(-1, nprops).astype(str)
    values_reshaped = values.reshape(-1, nprops).astype(str)
    prior_assays = [' + '.join(assays_reshaped[i, :j + 1]) for i in range(len(assays_reshaped)) for j in range(nprops)]
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
        'prob_assays': prob_assays,
        'prob_vals': prob_vals
    })

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    model: me.MoE = me.MoE.load("brick/moe").to(device)
    model.eval()
    model = DDP(model, device_ids=[rank])
    tokenizer: SelfiesPropertyValTokenizer = model.module.tokenizer
    tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()), device=device)
    tokenizer.value_indexes_tensor = torch.tensor(list(tokenizer.value_indexes().values()), device=device)

    dataset = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/hld", tokenizer, nprops=nprops)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    seen_inputs = set()
    batch_accum = []

    for i, (raw_inp, _, raw_out) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_tuples = tuple(map(lambda x, y: (tuple(x.tolist()), tuple(y.tolist())), raw_inp, raw_out))
        new_inputs_mask = [t not in seen_inputs for t in batch_tuples]
        seen_inputs.update(batch_tuples)

        if any(new_inputs_mask):
            new_raw_inp = raw_inp[new_inputs_mask]
            new_raw_out = raw_out[new_inputs_mask]
            batch_df = run_eval(i, new_raw_inp, new_raw_out, model, tokenizer, device)
            if not batch_df.empty:
                batch_accum.append(batch_df)

        if len(batch_accum) > 0 and sum(len(df) for df in batch_accum) > 1_000_000:
            pd.concat(batch_accum).to_parquet(temp_dir / f"multitask_predictions_{uuid.uuid4()}.parquet", index=False)
            batch_accum = []

    if batch_accum:
        pd.concat(batch_accum).to_parquet(temp_dir / f"multitask_predictions_{uuid.uuid4()}.parquet", index=False)

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()