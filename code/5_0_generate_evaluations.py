import math
import os, itertools, uuid, pathlib, shutil, logging
import pandas as pd, torch, numpy as np, tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# Enable faulthandler early to catch segfaults better
import faulthandler
faulthandler.enable()

import cvae.tokenizer
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
import cvae.utils
from cvae.tokenizer import SelfiesPropertyValTokenizer

import glob
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# Suppress the specific nested tensor warning if it's too noisy (optional)
# import warnings
# warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage*", category=UserWarning)

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
    rank = context.rank # For logging convenience
    logging.info(f"Rank {rank} - Entering run_eval for batch {i}")
    try:
        logging.info(f"Rank {rank} - Batch {i} - Initial raw_inp shape: {raw_inp.shape}, raw_out shape: {raw_out.shape}, device: {raw_inp.device}")
        inp, raw_out = raw_inp.to(context.device), raw_out.to(context.device)
        logging.info(f"Rank {rank} - Batch {i} - Tensors moved to device: {context.device}")

        # Filter based on nprops
        x = torch.sum(torch.isin(raw_out, context.tokenizer.value_indexes_tensor), dim=1) >= context.nprops
        chemical_id = torch.where(x)[0] + (i * context.batch_size) # Note: i*batch_size might be imprecise if batches are uneven from DDP
        inp, trunc_out = inp[x], raw_out[x, 1:(2 * context.nprops + 1)].reshape(-1, context.nprops, 2)
        logging.info(f"Rank {rank} - Batch {i} - After filtering: inp shape {inp.shape}, trunc_out shape {trunc_out.shape}. Num valid chemicals: {len(chemical_id)}")

        if inp.shape[0] == 0:
             logging.warning(f"Rank {rank} - Batch {i} - No valid inputs after filtering, skipping model evaluation.")
             return pd.DataFrame() # Return empty DataFrame

        # Permutations and tensor construction
        perm_out = torch.cat([trunc_out[:, list(perm), :] for perm in context.perm_indices], dim=0).reshape(-1, context.nprops * 2)
        sep_tensor = torch.full((perm_out.size(0), 1), context.tokenizer.SEP_IDX, device=context.device)
        out = torch.cat([sep_tensor, perm_out, torch.zeros_like(sep_tensor)], dim=1)
        teach = torch.cat([torch.ones_like(sep_tensor), out[:, :-1]], dim=1)
        rep_inp = inp.repeat(context.perm_count, 1)
        logging.info(f"Rank {rank} - Batch {i} - After permutations: rep_inp shape {rep_inp.shape}, teach shape {teach.shape}")

        prob = None # Initialize prob
        with torch.no_grad():
            logging.info(f"Rank {rank} - Batch {i} - Entering torch.no_grad()")
            logging.info(f"Rank {rank} - Batch {i} - Memory Before Model Call: Allocated={torch.cuda.memory_allocated(context.device)/1e6:.2f}MB, Reserved={torch.cuda.memory_reserved(context.device)/1e6:.2f}MB")
            logging.info(f"Rank {rank} - Batch {i} - === Calling context.model(rep_inp, teach) ===")
            try:
                prob = context.model(rep_inp, teach)
                # Check for NaNs/Infs immediately after model call
                if torch.isnan(prob).any() or torch.isinf(prob).any():
                     logging.error(f"Rank {rank} - Batch {i} - NaNs or Infs detected in model output 'prob'!")
                     # Handle error appropriately, maybe return empty df or raise
                     return pd.DataFrame()
                prob = torch.softmax(prob, dim=2) # Apply softmax only if model output is valid

            except Exception as e:
                 logging.exception(f"Rank {rank} - Batch {i} - !!! Exception during context.model() call !!!")
                 # Even with faulthandler, maybe log this before potential segfault
                 raise # Re-raise the exception if it's not a segfault

            logging.info(f"Rank {rank} - Batch {i} - === Returned from context.model(...) ===")
            logging.info(f"Rank {rank} - Batch {i} - Output 'prob' shape: {prob.shape if prob is not None else 'Error/None'}")
            logging.info(f"Rank {rank} - Batch {i} - Memory After Model Call: Allocated={torch.cuda.memory_allocated(context.device)/1e6:.2f}MB, Reserved={torch.cuda.memory_reserved(context.device)/1e6:.2f}MB")

        if prob is None:
             logging.error(f"Rank {rank} - Batch {i} - 'prob' is None after model call block, cannot proceed.")
             return pd.DataFrame()

        # Post-processing
        logging.info(f"Rank {rank} - Batch {i} - Starting post-processing")
        assays_mask = torch.isin(out, context.tokenizer.assay_indexes_tensor)
        assays = out[assays_mask]

        values_mask = torch.isin(out, context.tokenizer.value_indexes_tensor)
        values = out[values_mask]
        prob_vals = torch.argmax(prob, dim=2)[values_mask]
        rawprobs = prob[values_mask][:, context.tokenizer.value_indexes_tensor]
        probs = (rawprobs / rawprobs.sum(dim=1, keepdim=True))[:, 1] # Potential division by zero if sum is 0
        logging.info(f"Rank {rank} - Batch {i} - Shapes before cpu(): assays={assays.shape}, values={values.shape}, prob_vals={prob_vals.shape}, probs={probs.shape}")

        # Move to CPU and convert to numpy
        assays_np, values_np, prob_vals_np, probs_np = map(lambda x: x.cpu().numpy(), [assays, values, prob_vals, probs])
        logging.info(f"Rank {rank} - Batch {i} - Data moved to CPU/Numpy")

        # Prepare DataFrame columns
        position = np.tile(np.arange(context.nprops), chemical_id.size(0) * context.perm_count)
        chemical_id_np = torch.repeat_interleave(chemical_id.cpu(), context.perm_count) # Ensure chemical_id is on CPU
        chemical_id_np = torch.repeat_interleave(chemical_id_np, context.nprops).numpy()

        assays_reshaped = assays_np.reshape(-1, context.nprops).astype(str)
        prior_assays = [' + '.join(assays_reshaped[k, :j + 1]) for k in range(len(assays_reshaped)) for j in range(context.nprops)]

        values_reshaped = values_np.reshape(-1, context.nprops).astype(str)
        prior_values = [' + '.join(values_reshaped[k, :j + 1]) for k in range(len(values_reshaped)) for j in range(context.nprops)] # Corrected prior_values join

        logging.info(f"Rank {rank} - Batch {i} - Creating DataFrame")
        df = pd.DataFrame({
            'batch': i,
            'chemical_id': chemical_id_np,
            'prior_assays': prior_assays,
            'prior_values': prior_values,
            'assay': assays_np,
            'value': values_np,
            'probs': probs_np,
            'nprops': position,
            'prob_assays': assays_np, # Redundant? Check if needed
            'prob_vals': prob_vals_np
        })
        logging.info(f"Rank {rank} - Batch {i} - DataFrame created with shape {df.shape}")
        logging.info(f"Rank {rank} - Successfully completed run_eval for batch {i}")
        return df

    except Exception as e:
        logging.exception(f"Rank {rank} - !!! Unhandled Exception in run_eval for batch {i} !!!")
        # Return empty DataFrame or re-raise depending on desired behavior on error
        return pd.DataFrame()


def setup(rank): # Pass rank for logging before dist init maybe
    # Basic logging setup first
    outdir = pathlib.Path("cache/generate_evaluations")
    tmpdir = outdir / "temp"
    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = (logdir / f"log_{rank}.txt").as_posix() # Log using rank passed initially
    log_format = '%(asctime)s - RANK {} - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'.format(rank)
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S', force=True) # force=True helps if called multiple times?

    logging.info("--- Starting Setup ---")
    logging.info(f"Process ID: {os.getpid()}")
    logging.info(f"Environment RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")

    if rank == 0:
        logging.info("Rank 0 performing initial setup")
        outdir.mkdir(exist_ok=True, parents=True)
        tmpdir.mkdir(exist_ok=True, parents=True)
        try:
            deleted_files = list(tmpdir.glob('*'))
            for f in deleted_files:
                f.unlink()
            logging.info(f"Cleaned temp dir {tmpdir}. Deleted: {len(deleted_files)} items.")
        except Exception as e:
            logging.exception("Error cleaning temp directory")
        logging.info("Starting evaluation generation - Rank 0 Setup Done")

    # Log versions and device info after potential dist.init_process_group
    logging.info(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info(f"CUDA Version (available): {torch.version.cuda}")
        logging.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        try:
            # Get properties after setting device in main
            # device_id = torch.cuda.current_device() # This might need local_rank
            # logging.info(f"CUDA Device ID in setup: {device_id}")
            # logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(device_id)}")
            # logging.info(f"CUDA Device Properties: {torch.cuda.get_device_properties(device_id)}")
            pass # Log device props in main after setting device
        except Exception as e:
             logging.exception("Could not get CUDA device properties in setup")
    else:
        logging.warning("CUDA not available!")

    # Barrier moved to main after setup call and before main_worker
    # dist.barrier() # Don't barrier until dist is initialized

    logging.info("--- Finished Setup Function ---")
    return outdir, tmpdir

def cleanup():
    try:
        if dist.is_initialized():
            logging.info(f"Rank {dist.get_rank()} - Destroying process group.")
            dist.destroy_process_group()
        else:
            logging.info("Process group not initialized or already destroyed.")
    except Exception as e:
        logging.exception("Exception during distributed cleanup")


def main_worker(context: EvalContext, repetitions, outdir, tmpdir):
    rank = context.rank
    logging.info(f"Rank {rank} - Starting main_worker.")
    try:
        logging.info(f"Rank {rank} - Initializing Dataset.")
        dataset = mt.FastPackedSequenceShiftDataset("cache/pack_multitask_tensors/packed_hld")
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=context.batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, prefetch_factor=5, persistent_workers=True, pin_memory_device=f"cuda:{context.local_rank}") # Specify pin_memory_device

        logging.info(f"Rank {rank} - Initialized DataLoader with {len(dataloader)} batches per epoch.")
        logging.info(f"Rank {rank} - Using batch size: {context.batch_size}, num_workers=4")

        seen_inputs = set() # Consider if this is needed or causing issues
        batch_accum = []
        total_processed_count = 0

        for repeat in range(repetitions):
            logging.info(f"Rank {rank} - Starting repeat {repeat+1}/{repetitions}")
            sampler.set_epoch(repeat) # Important for shuffling with DistributedSampler if shuffle=True
            for i, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Rank {rank} Repeat {repeat+1}"):
                # logging.info(f"Rank {rank} - Processing batch {i} in repeat {repeat}") # Can be too verbose
                try:
                    raw_inp, _, raw_out = batch_data
                    # logging.debug(f"Rank {rank} - Batch {i} Data shapes: INP={raw_inp.shape}, OUT={raw_out.shape}") # DEBUG level maybe

                    # Filter based on seen_inputs (consider removing if not strictly necessary)
                    batch_tuples = tuple(map(lambda x, y: (tuple(x.tolist()), tuple(y.tolist())), raw_inp, raw_out))
                    new_inputs_mask = [t not in seen_inputs for t in batch_tuples]
                    seen_inputs.update(batch_tuples)

                    if any(new_inputs_mask):
                        new_raw_inp = raw_inp[new_inputs_mask]
                        new_raw_out = raw_out[new_inputs_mask]
                        # logging.info(f"Rank {rank} - Batch {i} - Found {new_raw_inp.shape[0]} new inputs out of {raw_inp.shape[0]}.")
                        batch_df = run_eval(i, new_raw_inp, new_raw_out, context)
                        if not batch_df.empty:
                             batch_accum.append(batch_df)
                             total_processed_count += new_raw_inp.shape[0] # Count processed items
                    # else:
                        # logging.info(f"Rank {rank} - Batch {i} - All inputs already seen.")

                    # Check accumulation size and save periodically
                    current_accum_rows = sum(len(df) for df in batch_accum)
                    if batch_accum and current_accum_rows > 1_000_000: # Save threshold
                        logging.info(f"Rank {rank} - Saving batch accumulation at step {i}, repeat {repeat}. Accumulated rows: {current_accum_rows}")
                        try:
                            pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{rank}_{repeat}_{i}_{uuid.uuid4()}.parquet", index=False)
                            batch_accum = [] # Clear after saving
                        except Exception as e:
                            logging.exception(f"Rank {rank} - Failed to save accumulated batch at step {i}, repeat {repeat}")
                            # Decide how to handle this - maybe try again later or just log

                except Exception as e:
                    logging.exception(f"Rank {rank} - Error processing batch {i} in repeat {repeat}")
                    # Continue to next batch or break? Depends on error severity.
                    continue # Example: try next batch

            # End of repeat loop
            logging.info(f"Rank {rank} - Finished repeat {repeat+1}/{repetitions}. Total items processed in this worker so far: {total_processed_count}")

        # Save any remaining accumulated data after all repetitions
        if batch_accum:
            logging.info(f"Rank {rank} - Saving final batch accumulation. Rows: {sum(len(df) for df in batch_accum)}")
            try:
                 pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{rank}_final_{uuid.uuid4()}.parquet", index=False)
            except Exception as e:
                 logging.exception(f"Rank {rank} - Failed to save final accumulated batch")

        logging.info(f"Rank {rank} - Finished main_worker loop.")

    except Exception as e:
        logging.exception(f"Rank {rank} - !!! Unhandled Exception in main_worker !!!")
    finally:
        logging.info(f"Rank {rank} - Exiting main_worker.")
        # Cleanup is usually called outside main_worker after completion or error in main script
        # cleanup() # Don't call cleanup here, call in main __name__ block


def finalize_output(rank, outdir, tmpdir): # Pass rank for logging
    # Only rank 0 should perform this
    if rank == 0:
        logging.info("Rank 0 - Starting finalize_output.")
        try:
            logging.info(f"Rank 0 - Searching for parquet files in {tmpdir}")
            parquet_files = glob.glob(str(tmpdir / "*.parquet"))
            if not parquet_files:
                 logging.warning("Rank 0 - No parquet files found in temp directory to concatenate.")
                 return

            logging.info(f"Rank 0 - Found {len(parquet_files)} parquet files. Concatenating...")
            # Consider reading in batches if total data is huge
            all_dfs = []
            for file in tqdm.tqdm(parquet_files, desc="Rank 0 Reading Parquets"):
                 try:
                      all_dfs.append(pd.read_parquet(file))
                 except Exception as e:
                      logging.error(f"Rank 0 - Failed to read parquet file {file}: {e}")
                      # Decide: skip this file or fail?
                      continue

            if not all_dfs:
                 logging.error("Rank 0 - Failed to read any parquet files successfully.")
                 return

            df = pd.concat(all_dfs, ignore_index=True)
            logging.info(f"Rank 0 - Concatenated DataFrame shape: {df.shape}")

            # Save single concatenated file (optional intermediate step)
            single_file_path = outdir / "multitask_predictions_single_file.parquet"
            logging.info(f"Rank 0 - Saving concatenated data to single file: {single_file_path}")
            df.to_parquet(single_file_path, index=False, engine="pyarrow", compression="zstd", compression_level=9)

            # Save partitioned dataset
            partitioned_dir = outdir / "multitask_predictions.parquet"
            logging.info(f"Rank 0 - Saving partitioned dataset to: {partitioned_dir}")
            if partitioned_dir.exists():
                 logging.warning(f"Rank 0 - Partitioned directory {partitioned_dir} exists. Overwriting.")
                 # shutil.rmtree(partitioned_dir) # Or let write_dataset handle overwrite

            table = pq.read_table(single_file_path) # Read back from single file
            ds.write_dataset(
                data=table,
                base_dir=partitioned_dir,
                format="parquet",
                file_options=ds.ParquetFileFormat().make_write_options(compression="zstd", compression_level=9),
                max_rows_per_file=25_000_000,
                existing_data_behavior="overwrite_or_ignore", # 'delete_matching' might be safer if re-running
                basename_template="part-{i}.parquet",
            )
            logging.info(f"Rank 0 - Finished saving partitioned dataset.")

            # Calculate AUC
            logging.info(f"Rank 0 - Calculating AUC score...")
            try:
                import sklearn.metrics
                # Read back the partitioned dataset for calculation
                # This ensures we are using the final written data
                final_df = ds.dataset(partitioned_dir, format="parquet").to_table().to_pandas()
                # df = pd.read_parquet(partitioned_dir) # Reading directory might also work depending on pandas/pyarrow version
                auc_by_nprop = final_df.groupby('nprops').apply(lambda x: sklearn.metrics.roc_auc_score(x['value'], x['probs']))
                logging.info(f"Rank 0 - AUC by number of properties:\n{auc_by_nprop}")
            except ImportError:
                 logging.warning("Rank 0 - sklearn not found, cannot calculate AUC score.")
            except Exception as e:
                 logging.exception("Rank 0 - Error calculating AUC score.")

            logging.info("Rank 0 - Finished finalize_output.")

        except Exception as e:
            logging.exception("Rank 0 - !!! Unhandled Exception in finalize_output !!!")
    else:
        logging.info(f"Rank {rank} - Skipping finalize_output.")


if __name__ == "__main__":
    # --- Early setup for rank info ---
    rank = int(os.environ.get("RANK", -1)) # Get rank early for initial logging setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    # --- Setup Logging ---
    # Note: Setup function configures file logging per rank
    outdir, tmpdir = setup(rank) # Pass rank to setup

    try:
        logging.info(f"--- Starting Main Execution Block (Rank {rank}) ---")
        logging.info(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")

        # --- Distributed Setup ---
        if not dist.is_available():
            logging.error("Distributed package is not available! Exiting.")
            exit(1)
        if not torch.cuda.is_available():
             logging.error("CUDA is not available! Exiting.")
             exit(1)

        logging.info("Initializing process group...")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        logging.info("Process group initialized.")

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logging.info(f"Set device to: {device}")
        logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(local_rank)}")
        logging.info(f"CUDA Device Properties: {torch.cuda.get_device_properties(local_rank)}")
        logging.info(f"Initial Memory: Allocated={torch.cuda.memory_allocated(device)/1e6:.2f}MB, Reserved={torch.cuda.memory_reserved(device)/1e6:.2f}MB")


        # --- Load Model ---
        logging.info("Loading model...")
        model_load_path = "cache/train_multitask_transformer_parallel/models/moe"
        try:
            model: me.MoE = me.MoE.load(model_load_path).to(device)
            logging.info(f"Model loaded successfully from {model_load_path} and moved to {device}")
        except Exception as e:
             logging.exception(f"Failed to load model from {model_load_path}")
             raise # Cannot continue without model


        # --- Compile/DDP ---
        # model = torch.compile(model) # Optional: Add logging around compile if used
        logging.info("Wrapping model with DDP...")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False) # Set find_unused_parameters=False if sure no params are unused, might speed up. Check if needed.
        model.eval()
        logging.info("Model wrapped in DDP and set to eval mode.")

        # --- Tokenizer ---
        logging.info("Setting up tokenizer...")
        try:
            tokenizer: SelfiesPropertyValTokenizer = model.module.tokenizer # Access tokenizer from underlying module
            tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()), device=device)
            tokenizer.value_indexes_tensor = torch.tensor(list(tokenizer.value_indexes().values()), device=device)
            logging.info("Tokenizer setup complete.")
        except AttributeError:
             logging.exception("Failed to access tokenizer from model.module.tokenizer")
             raise # Cannot continue without tokenizer


        # --- Configuration ---
        batch_size = 5  # FORCE BATCH SIZE TO 1 for detailed debugging
        nprops = 5
        repetitions=24 # FORCE REPETITIONS TO 1 for faster debugging cycle
        logging.info(f"Configuration: batch_size={batch_size}, nprops={nprops}, repetitions={repetitions}")


        # --- Create Context ---
        logging.info("Creating EvalContext...")
        perm_indices=list(itertools.permutations(range(nprops)))
        perm_count=math.factorial(nprops)
        context = EvalContext(
            rank=rank,
            local_rank=local_rank,
            model=model,
            tokenizer=tokenizer,
            device=device,
            perm_indices=perm_indices,
            perm_count=perm_count,
            nprops=nprops,
            batch_size=batch_size
        )
        logging.info(f"EvalContext created. Permutation count: {perm_count}")

        # --- Barrier before starting worker ---
        logging.info("Waiting at barrier before starting main worker...")
        dist.barrier()
        logging.info("Passed barrier.")

        # --- Run Main Worker ---
        logging.info(f"Starting evaluation generation on Rank {rank}")
        main_worker(context, repetitions=repetitions, outdir=outdir, tmpdir=tmpdir)
        logging.info(f"Finished main_worker call on Rank {rank}")

        # --- Barrier before finalization ---
        logging.info("Waiting at barrier before finalization...")
        dist.barrier()
        logging.info("Passed barrier.")

        # --- Finalize Output (Rank 0 only) ---
        finalize_output(rank, outdir, tmpdir)

        logging.info(f"--- Rank {rank} finished successfully ---")

    except Exception as e:
        logging.exception(f"--- !!! Unhandled Exception in __main__ (Rank {rank}) !!! ---")
        # Optional: try to signal other ranks about the error? DDP might handle this partially.

    finally:
        logging.info(f"Rank {rank} entering final cleanup.")
        cleanup() # Ensure cleanup happens even on error
        logging.info(f"Rank {rank} finished cleanup.")