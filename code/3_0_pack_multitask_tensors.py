import pathlib
import torch
import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from cvae.models.multitask_transformer import process_assay_vals
from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer
import cvae.tokenizer

def process_file(file_path, pad_idx, sep_idx, end_idx, nprops, epochs):
    file_data = torch.load(file_path, map_location="cpu", weights_only=False)
    selfies = file_data["selfies"]
    assay_vals = file_data["assay_vals"]

    selfies_out = []
    tch_out = []
    out_out = []

    for i in range(len(assay_vals)):
        for _ in range(epochs):
            t, o = process_assay_vals(assay_vals[i], pad_idx, sep_idx, end_idx, nprops)
            tch_out.append(t)
            out_out.append(o)
            selfies_out.append(selfies[i])

    return (
        torch.stack(selfies_out),
        torch.stack(tch_out),
        torch.stack(out_out),
    )

def pack_dataset(split_dir: pathlib.Path, out_path: pathlib.Path, tokenizer, nprops: int, epochs: int):
    logging.info(f"Starting to pack dataset from {split_dir}")
    logging.info(f"Output will be saved to {out_path}")
    logging.info(f"Using {nprops} properties and {epochs} epochs")

    pad_idx, sep_idx, end_idx = tokenizer.PAD_IDX, tokenizer.SEP_IDX, tokenizer.END_IDX
    file_paths = sorted(split_dir.glob("*.pt"))
    logging.info(f"Found {len(file_paths)} files to process")

    selfies_all = []
    tch_all = []
    out_all = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_file,
                file_path,
                pad_idx,
                sep_idx,
                end_idx,
                nprops,
                epochs
            ): file_path.name
            for file_path in file_paths
        }

        with tqdm.tqdm(total=len(futures), desc=f"Packing {split_dir.name}") as pbar:
            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    s, t, o = future.result()
                    selfies_all.append(s)
                    tch_all.append(t)
                    out_all.append(o)
                except Exception as e:
                    logging.error(f"Error processing file {file_name}: {e}")
                pbar.update(1)

    selfies_all = torch.cat(selfies_all, dim=0)
    tch_all = torch.cat(tch_all, dim=0)
    out_all = torch.cat(out_all, dim=0)

    logging.info(f"Final tensor shapes:")
    logging.info(f"  selfies: {selfies_all.shape}")
    logging.info(f"  tch: {tch_all.shape}")
    logging.info(f"  out: {out_all.shape}")

    torch.save({
        "selfies": selfies_all,
        "tch": tch_all,
        "out": out_all
    }, out_path)

    logging.info(f"Successfully saved {split_dir.name} to {out_path}")
    logging.info(f"Total records: {len(selfies_all)} ({len(selfies_all)//epochs} unique × {epochs} epochs)")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pack_multitask_tensors.log')
        ]
    )

    indir = pathlib.Path("cache/build_tensordataset/multitask_tensors")
    outdir = pathlib.Path("cache/pack_multitask_tensors")
    outdir.mkdir(parents=True, exist_ok=True)
    nprops = 20
    epochs = 100

    logging.info("Starting tensor packing process")
    logging.info(f"Input directory: {indir}")
    logging.info(f"Output directory: {outdir}")

    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    logging.info("Loaded tokenizer successfully")

    for split in ["trn", "tst", "hld"]:
        logging.info(f"\nProcessing {split} split")
        split_dir = indir / split
        out_path = outdir / f"packed_{split}.pt"
        pack_dataset(split_dir, out_path, tokenizer, nprops, epochs)