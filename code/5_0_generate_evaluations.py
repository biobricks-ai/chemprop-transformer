import math
import os, itertools, uuid, pathlib, shutil, logging
import pandas as pd, torch, numpy as np, tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import cvae.tokenizer
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
import cvae.utils
from cvae.tokenizer import SelfiesPropertyValTokenizer

import glob
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import faulthandler
faulthandler.enable()

torch._nested_tensor_from_mask_left = None

class EvalContext:
    def __init__(self, rank, local_rank, model, tokenizer, device, perm_indices, perm_count, nprops, batch_size):
        self.rank = rank
        self.local_rank = local_rank
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.perm_indices = perm_indices
        self.perm_count = perm_count
        self.nprops = nprops
        self.batch_size = batch_size

def run_eval(i, raw_inp, raw_out, context: EvalContext):
    inp, raw_out = raw_inp.to(context.device), raw_out.to(context.device)
    
    x = torch.sum(torch.isin(raw_out, context.tokenizer.value_indexes_tensor), dim=1) >= context.nprops
    chemical_id = torch.where(x)[0] + (i * context.batch_size)
    inp, trunc_out = inp[x], raw_out[x, 1:(2 * context.nprops + 1)].reshape(-1, context.nprops, 2)

    perm_out = torch.cat([trunc_out[:, list(perm), :] for perm in context.perm_indices], dim=0).reshape(-1, context.nprops * 2)
    sep_tensor = torch.full((perm_out.size(0), 1), context.tokenizer.SEP_IDX, device=context.device)
    out = torch.cat([sep_tensor, perm_out, torch.zeros_like(sep_tensor)], dim=1)
    teach = torch.cat([torch.ones_like(sep_tensor), out[:, :-1]], dim=1)
    rep_inp = inp.repeat(context.perm_count, 1)

    with torch.no_grad():
        prob = torch.softmax(context.model(rep_inp, teach), dim=2)

    assays_mask = torch.isin(out, context.tokenizer.assay_indexes_tensor)
    assays = out[assays_mask]

    values_mask = torch.isin(out, context.tokenizer.value_indexes_tensor)
    values = out[values_mask]
    prob_vals = torch.argmax(prob, dim=2)[values_mask]
    rawprobs = prob[values_mask][:, context.tokenizer.value_indexes_tensor]
    probs = (rawprobs / rawprobs.sum(dim=1, keepdim=True))[:, 1]

    assays, values, assays, prob_vals, probs = map(lambda x: x.cpu().numpy(), [assays, values, assays, prob_vals, probs])
    position = np.tile(np.arange(context.nprops), chemical_id.size(0) * context.perm_count)
    chemical_id = torch.repeat_interleave(chemical_id, context.perm_count)
    chemical_id = torch.repeat_interleave(chemical_id, context.nprops).cpu().numpy()
    assays_reshaped = assays.reshape(-1, context.nprops).astype(str)
    prior_assays = [' + '.join(assays_reshaped[i, :j + 1]) for i in range(len(assays_reshaped)) for j in range(context.nprops)]

    values_reshaped = values.reshape(-1, context.nprops).astype(str)
    prior_values = [values_reshaped[i, :j + 1] for i in range(len(values_reshaped)) for j in range(context.nprops)]
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

def setup():
    outdir = pathlib.Path("cache/generate_evaluations")
    tmpdir = outdir / "temp"
    logdir = outdir / "logs"
    print(f"RANK: {os.environ.get('RANK')}")
    if os.environ.get("RANK") == "0":
        print(f"SETUP")
        outdir.mkdir(exist_ok=True, parents=True)
        tmpdir.mkdir(exist_ok=True, parents=True)
        [f.unlink() for f in tmpdir.glob('*')]
        print(f"files remaining in {tmpdir}: {list(tmpdir.glob('*'))}")
        logdir.mkdir(exist_ok=True, parents=True)
        print("Starting evaluation generation")

    
    logfile = (logdir / f"log_{os.environ.get('RANK')}.txt").as_posix()
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    dist.barrier()

    logging.info("finished setup")
    return outdir, tmpdir

def cleanup():
    dist.destroy_process_group()

def main_worker(context: EvalContext, repetitions, outdir, tmpdir):
    dataset = mt.FastPackedSequenceShiftDataset("cache/pack_multitask_tensors/packed_hld")
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=context.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=context.batch_size, sampler=sampler, 
        num_workers=4, pin_memory=True, prefetch_factor=5, persistent_workers=True)

    logging.info(f"Initialized model and tokenizer on rank {context.rank}")
    logging.info(f"Using batch size: {context.batch_size}")
    logging.info(f"Starting evaluation loop with {len(dataloader)} batches")
    seen_inputs = set()
    batch_accum = []

    for repeat in range(repetitions):
        logging.info(f"Starting repeat {repeat}")
        for i, (raw_inp, _, raw_out) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            logging.info(f"Processing batch {i} out of {len(dataloader)}")
            batch_tuples = tuple(map(lambda x, y: (tuple(x.tolist()), tuple(y.tolist())), raw_inp, raw_out))
            new_inputs_mask = [t not in seen_inputs for t in batch_tuples]
            seen_inputs.update(batch_tuples)
            
            if any(new_inputs_mask):
                new_raw_inp = raw_inp[new_inputs_mask]
                new_raw_out = raw_out[new_inputs_mask]
                batch_df = run_eval(i, new_raw_inp, new_raw_out, context)
                batch_accum.append(batch_df)

            if len(batch_accum) > 0 and sum(len(df) for df in batch_accum) > 1e6:
                logging.info(f"Saving batch accumulation at step {i} with {sum(len(df) for df in batch_accum)} rows")
                pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{os.environ.get('RANK')}_{uuid.uuid4()}.parquet", index=False)
                batch_accum = []

        if batch_accum:
            logging.info("Saving final batch accumulation")
            pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{os.environ.get('RANK')}_{uuid.uuid4()}.parquet", index=False)

    cleanup()

def finalize_output(outdir, tmpdir):
    if int(os.environ["RANK"]) == 0:
        logging.info("Concatenating parquet files")
        parquet_files = glob.glob(str(tmpdir / "*.parquet"))
        df = pd.concat([pd.read_parquet(file) for file in parquet_files])
        df.to_parquet(outdir / "multitask_predictions_single_files.parquet", index=False, engine="pyarrow", compression="zstd", compression_level=9)

        table = pq.read_table(outdir / "multitask_predictions_single_files.parquet")
        ds.write_dataset(
            data=table,
            base_dir=outdir / "multitask_predictions.parquet",
            format="parquet",
            file_options=ds.ParquetFileFormat().make_write_options(compression="zstd", compression_level=9),
            max_rows_per_file=25_000_000,
            existing_data_behavior="overwrite_or_ignore",
            basename_template="part-{i}.parquet",
        )

        import sklearn.metrics
        df = pd.read_parquet(outdir / "multitask_predictions.parquet")
        auc_by_nprop = df.groupby('nprops').apply(lambda x: sklearn.metrics.roc_auc_score(x['value'], x['probs']))
        logging.info(f"AUC by number of properties: {auc_by_nprop}")

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model: me.MoE = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe").to(device)
    
    # model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    tokenizer: SelfiesPropertyValTokenizer = model.module.tokenizer
    tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()), device=device)
    tokenizer.value_indexes_tensor = torch.tensor(list(tokenizer.value_indexes().values()), device=device)

    batch_size = min(4, 2 ** (torch.cuda.get_device_properties(device).total_memory // (2**30)))  # dynamic batch size cap
    nprops = 5

    context = EvalContext(
        rank=rank,
        local_rank=local_rank,
        model=model,
        tokenizer=tokenizer,
        device=device,
        perm_indices=list(itertools.permutations(range(nprops))),
        perm_count=math.factorial(nprops),
        nprops=nprops,
        batch_size=batch_size
    )

    outdir, tmpdir = setup()
    dist.barrier()
    print("setup done")
    logging.info(f"Selected dynamic batch size: {batch_size}")
    logging.info(f"Starting evaluation generation on rank {rank}")
    main_worker(context, repetitions=24, outdir=outdir, tmpdir=tmpdir)
    finalize_output(outdir, tmpdir)
