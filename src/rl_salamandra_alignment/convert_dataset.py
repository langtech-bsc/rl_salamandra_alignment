"""Script to convert HF datasets to a format that can be run locally in MN5"""
from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import os
from rl_salamandra_alignment import logger

def convert_dataset(
    dataset_input_path: str,
    dataset_output_path: str
) -> bool:
    """Convert HF datasets to a format that can be loaded locally in MN5

    Args:
        dataset_input_path (str): Path to dataset downloaded from HF
        dataset_output_path (str): Path to save converted dataset

    Raises:
        ValueError: Raised if there are errors while loading the dataset

    Returns:
        bool: True if conversion was successful
    """

    dataset = None
    try:
        dataset = load_from_disk(dataset_input_path)
    except:
        pass

    if dataset is None:
        try:
            dataset = load_dataset(dataset_input_path)
        except:
            pass

    if dataset:
        # create test and dev splits if not found
        if not isinstance(dataset, DatasetDict):
            if isinstance(dataset, Dataset):
                dataset = DatasetDict({"train": dataset})
            else:
                raise ValueError(f"Error loading the dataset:\n{dataset_input_path}")
        
        if "test" in dataset and "eval" in dataset:
            # all good
            pass
        elif "test" in dataset and "eval" not in dataset:
            full_train_split = dataset["train"]
            eval_size = int(DEV_SPLIT_SIZE * len(full_train_split))
            train_dev_split = full_train_split.train_test_split(
                test_size=eval_size,
                shuffle=True,
                seed=RANDOM_SEED
            )
            dataset = DatasetDict(
                {
                    "train": train_dev_split["train"], 
                    "eval": train_dev_split["test"],
                    "test": dataset["test"]
                }
            )
            
        elif "test" not in dataset and "eval" in dataset:
            full_train_split = dataset["train"]
            test_size = int(TEST_SPLIT_SIZE * len(full_train_split))
            train_dev_split = full_train_split.train_test_split(
                test_size=test_size,
                shuffle=True,
                seed=RANDOM_SEED
            )
            dataset = DatasetDict(
                {
                    "train": train_dev_split["train"], 
                    "eval": dataset["eval"],
                    "test": train_dev_split["test"]
                }
            )
            
        elif "test" not in dataset and "eval" not in dataset:
            # We will need to split two times
            full_train_split = dataset["train"]
            test_size = int(TEST_SPLIT_SIZE * len(full_train_split))
            eval_size = int(DEV_SPLIT_SIZE * len(full_train_split))
            train_TestAndEval = full_train_split.train_test_split(
                test_size= test_size + eval_size,
                shuffle=True,
                seed=RANDOM_SEED
            )
            TestAndEval = train_TestAndEval["test"].train_test_split(
                test_size=test_size,
                train_size=eval_size,
                shuffle=True,
                seed=RANDOM_SEED
            )
            
            dataset = DatasetDict(
                {
                    "train": train_TestAndEval["train"], 
                    "eval": TestAndEval["train"],
                    "test": TestAndEval["test"]
                }
            )
        else:
            raise(f"Check the splits for \n{dataset_input_path}\n{list(dataset.keys())}")
        
        output_path = dataset_output_path

        # Finally, save to disk
        try:
            dirname = os.path.dirname(output_path)
            os.makedirs(dirname, exist_ok=True)
        except:
            if dirname:
                logger.info("There may have been a problem creating the parent directory")
        dataset.save_to_disk(output_path)
    else:
        logger.warning(f"Problem loading this file:\n{dataset_input_path}")
        raise ValueError("Dataset could not be automatically loaded")

def main_convert():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        help="Path to dataset downloaded from HF",
    )
    parser.add_argument(
        "--output_path",
        help="Path to save converted dataset",
    )
    parser.add_argument(
        "--test_split",
        help="Test split size",
        default=0.1,
        type=float,
        required=False
    )
    parser.add_argument(
        "--dev_split",
        help="Test split size",
        default=0.1,
        type=float,
        required=False
    )  
    parser.add_argument(
        "--random_seed",
        help="Random seed for splitting",
        default=42,
        type=int,
        required=False
    )
    args = parser.parse_args()
    
    global DEV_SPLIT_SIZE
    global TEST_SPLIT_SIZE
    global RANDOM_SEED
    DEV_SPLIT_SIZE = args.dev_split
    TEST_SPLIT_SIZE = args.test_split
    RANDOM_SEED = args.random_seed
    
    print(f"""
        Splitting parameters:
        Random seed: {RANDOM_SEED}
        Dev split size: {DEV_SPLIT_SIZE}
        Test spplit size: {TEST_SPLIT_SIZE}
        If your dataset already contains splits train-test-eval, it will NOT be splitted again.
        """.replace("  ", "")
        )
    print(TEST_SPLIT_SIZE, DEV_SPLIT_SIZE, RANDOM_SEED)  
    
    convert_dataset(
        args.input_path,
        args.output_path
    )


if __name__ == "__main__":
    main_convert()