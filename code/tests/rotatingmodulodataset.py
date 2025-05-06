import cvae.tokenizer
import cvae.utils
from cvae.models.multitask_transformer import RotatingModuloSequenceShiftDataset
import torch
import bisect

# Example usage:
if __name__ == "__main__":
    # Create the dataset
    
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    tmpdir = "cache/tests/rotatingmodulodataset/multitask_tensors"
    trn = RotatingModuloSequenceShiftDataset(
        path="cache/build_tensordataset/multitask_tensors/trn",
        tokenizer=tokenizer,
        nprops=5
    )
    tst = RotatingModuloSequenceShiftDataset(
        path="cache/build_tensordataset/multitask_tensors/tst",
        tokenizer=tokenizer,
        nprops=5
    )

    inp1, tch1, out1 = trn.__getitem__(3)
    inp2, tch2, out2 = tst.__getitem__(0)
    
    # Test 1: Check that rotation is happening
def test_rotation(dataset, num_samples=100):
    dataset.reset_rotations()
    print("\n=== Testing Rotation ===")
    for idx in range(min(num_samples, len(dataset))):
        # Get the same sample multiple times
        prop_sequences = []
        print(f"idx{idx}")
        for _ in range(5):  # Get the same sample 5 times
            _, tchraw, outraw = dataset.__getitem__(idx)
            
            # remove padidx and eos and sos from out
            outmask = (outraw != tokenizer.PAD_IDX) & (outraw != tokenizer.END_IDX) & (outraw != tokenizer.SEP_IDX)
            out = outraw[outmask]
            
            tchmask = (tchraw != tokenizer.PAD_IDX) & (tchraw != tokenizer.END_IDX) & (tchraw != tokenizer.SEP_IDX) & (tchraw != 1)
            tch = tchraw[tchmask]

            # assert that tch and out are the same
            print(out)
            print(tch)
            assert torch.all(tch == out), f"Sample {idx} does not show rotation. tch and out are not the same"

            # Extract property IDs (every other element starting from index 1)
            props = out.tolist()[::2]  # Remove SOS/EOS tokens
            
            print(f"props are {props}")
            # Group into property-value pairs and extract just the properties
            if len(props) >= 2:
                prop_ids = props  # Take every other element (property IDs)
                prop_sequences.append(prop_ids)
        
        # Check if we have enough properties to test rotation
        if len(prop_sequences) < 2 or not all(prop_sequences):
            continue
            
        # Check that the first property changes across accesses
        first_props = [seq[0] if seq else None for seq in prop_sequences]
        unique_first_props = set(p for p in first_props if p is not None)
        
        # If we have multiple unique first properties, rotation is happening
        print(f"properties are {prop_sequences}")
        assert len(unique_first_props) > 1, f"Sample {idx} does not show rotation. First properties across accesses: {first_props}"
        print(f"Sample {idx} shows rotation: First properties across accesses: {first_props}")

def test_randomization(dataset, num_samples=100):
    print("\n=== Testing Randomization ===")
    randomizations_found = 0
    
    # Reset all rotation counters at the start
    dataset.reset_rotations()
    
    for idx in range(min(num_samples, len(dataset))):
        print(f"idx{idx}")
        
        # Get initial property sequence
        _, tchraw, outraw = dataset.__getitem__(idx)
        
        # Remove padidx, eos, and sos from out
        outmask = (outraw != tokenizer.PAD_IDX) & (outraw != tokenizer.END_IDX) & (outraw != tokenizer.SEP_IDX)
        out = outraw[outmask]
        
        # Extract property IDs (every other element)
        initial_props = out.tolist()[::2]
        
        print(f"Initial properties: {initial_props}")
        
        if len(initial_props) <= 1:
            print(f"Sample {idx} has too few properties for randomization test")
            continue
        
        # Get the file and local indices for this sample
        file_idx = bisect.bisect_right(dataset.cumulative_lengths, idx) - 1
        local_idx = idx - dataset.cumulative_lengths[file_idx]
        
        # Get the raw assay values to determine cycle length
        raw_assay_vals = dataset.data[file_idx][1][local_idx]
        mask = raw_assay_vals != dataset.pad_idx
        assay_vals = raw_assay_vals[mask][1:-1]  # Remove SOS/EOS
        
        # Count total property-value pairs
        if assay_vals.numel() < 2:
            continue
        
        total_pairs = assay_vals.view(-1, 2).size(0)
        
        # Get the global index
        global_idx = dataset._get_global_idx(file_idx, local_idx)
        
        # Save current shuffled indices
        initial_shuffled = list(dataset.sample_shuffled_indices.get(global_idx, []))
        
        print(f"Initial shuffle: {initial_shuffled}")
        
        # Now complete a full cycle by calling getitem multiple times
        for _ in range(total_pairs):
            _, _, _ = dataset.__getitem__(idx)
        
        # Get sample after cycle completion
        _, tchraw_after, outraw_after = dataset.__getitem__(idx)
        
        # Remove padidx, eos, and sos from out
        outmask_after = (outraw_after != tokenizer.PAD_IDX) & (outraw_after != tokenizer.END_IDX) & (outraw_after != tokenizer.SEP_IDX)
        out_after = outraw_after[outmask_after]
        
        # Assert that tch and out are still properly synchronized
        tchmask_after = (tchraw_after != tokenizer.PAD_IDX) & (tchraw_after != tokenizer.END_IDX) & (tchraw_after != tokenizer.SEP_IDX) & (tchraw_after != 1)
        tch_after = tchraw_after[tchmask_after]
        
        print(out_after)
        print(tch_after)
        assert torch.all(tch_after == out_after), f"Sample {idx} tch and out are not the same after cycle"
        
        # Extract property IDs after cycle
        after_props = out_after.tolist()[::2]
        
        print(f"Properties after cycle: {after_props}")
        
        # Get new shuffled indices
        new_shuffled = list(dataset.sample_shuffled_indices.get(global_idx, []))
        
        print(f"New shuffle: {new_shuffled}")
        
        # Check if there was a shuffle
        if (len(initial_shuffled) == len(new_shuffled) and 
            len(initial_shuffled) > 1 and 
            new_shuffled != initial_shuffled):
            print(f"Sample {idx} SHOWS randomization!")
            print(f"  Initial shuffle: {initial_shuffled}")
            print(f"  New shuffle: {new_shuffled}")
            randomizations_found += 1
        else:
            print(f"Sample {idx} does NOT show randomization")
            print(f"  Initial shuffle: {initial_shuffled}")
            print(f"  New shuffle: {new_shuffled}")
    
    print(f"\nRandomization Summary: Found randomization in {randomizations_found} samples")
    assert randomizations_found > 0, "No randomization found after cycle completion"

    test_rotation(trn)
    test_randomization(trn)

    test_rotation(tst)
    test_randomization(tst)

    # write a test here that goes through the first 100 items of each dataset and checks that randomization