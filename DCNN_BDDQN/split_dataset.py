import h5py
import os
import argparse
import random
from tqdm import tqdm

def split_h5_dataset(input_path: str, split_ratio: float = 0.8, seed: int = 42):
    """
    Splits an HDF5 dataset into training and test sets.

    Args:
        input_path (str): HDF5 file path.
        split_ratio (float, optional): Percentage of the dataset to be used for training.
                                       Default: 0.8.
        seed (int, optional): Seed for the random number generator. Default: 42.

    Output:
        Creates two new HDF5 files:
        - [filename]_train.h5: Contains the training data.
        - [filename]_test.h5: Contains the test data.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if not 0 < split_ratio < 1:
        raise ValueError("Split ratio must be between 0 and 1.")

    random.seed(seed)
    
    # Output files will be saved in the same directory as the input file
    output_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    train_output_path = os.path.join(output_dir, f"{base_name}_train.h5")
    test_output_path = os.path.join(output_dir, f"{base_name}_test.h5")

    print(f"Splitting {input_path} ...")
    print(f"  - Training set: {train_output_path}")
    print(f"  - Test set:          {test_output_path}")
    print(f"  - Report:             {split_ratio*100:.0f}% train / {(1-split_ratio)*100:.0f}% test")

    with h5py.File(input_path, 'r') as hf:
        # Assuming data is in the 'alignments' dataset
        if 'alignments' not in hf:
            raise KeyError("'alignments' not found in the HDF5 file.")
            
        dset = hf['alignments']
        num_items = dset.shape[0]
        item_shape = dset.shape[1:]
        dtype = dset.dtype

        # Generate a list of indices and shuffle it
        indices = list(range(num_items))
        random.shuffle(indices)

        # Calculate the split point and separate indices
        split_point = int(num_items * split_ratio)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        # Sort indices for more efficient disk reading
        train_indices.sort()
        test_indices.sort()

        # Create HDF5 file for the training set
        with h5py.File(train_output_path, 'w') as train_hf:
            train_dset = train_hf.create_dataset(
                'alignments', 
                shape=(len(train_indices), *item_shape), 
                dtype=dtype,
                chunks=dset.chunks # Maintains the same chunking structure for efficiency
            )
            # Copy data using indices
            for i, original_idx in enumerate(tqdm(train_indices, desc="Writing the training set...")):
                train_dset[i] = dset[original_idx]
        
        # Create HDF5 file for the test set
        with h5py.File(test_output_path, 'w') as test_hf:
            test_dset = test_hf.create_dataset(
                'alignments', 
                shape=(len(test_indices), *item_shape), 
                dtype=dtype,
                chunks=dset.chunks # Maintains the same chunking structure
            )
            for i, original_idx in enumerate(tqdm(test_indices, desc="Writing the test set...")):
                test_dset[i] = dset[original_idx]

    print("\nDone.")
    print(f"The training set contains {len(train_indices)} elements.")
    print(f"the test set contains {len(test_indices)} elements.")


def main():
    parser = argparse.ArgumentParser(
        description="Splits an HDF5 dataset into training and test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Input HDF5 file path.")
    parser.add_argument("--ratio", type=float, default=0.8, help="Percentage of the dataset to be used for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator.")
    
    args = parser.parse_args()
    
    try:
        split_h5_dataset(args.input_file, args.ratio, args.seed)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()