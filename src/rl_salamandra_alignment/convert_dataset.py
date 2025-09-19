"""Script to convert HF datasets to a format that can be run locally in MN5"""
from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
import os
from rl_salamandra_alignment import logger
import json
from tqdm import tqdm

def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def try_load_directory_of_jsons(path:str):
    filenames = os.listdir(path)
    jsons = [filename for filename in filenames if filename.endswith(".json")]
    jsonls = [filename for filename in filenames if filename.endswith(".jsonl")]
    if not jsons and not jsonls:
        raise ValueError(f"No json/jsonl files found in {path}")
    all_datasets = []
    for json in tqdm(jsons):
        sub_data = read_json(os.path.join(path, json))
        sub_ds = Dataset.from_list(sub_data)
        all_datasets.append(sub_ds)

    for jsonl in tqdm(jsonls):
        sub_data = read_jsonl(os.path.join(path, jsonl))
        sub_ds = Dataset.from_list(sub_data)
        all_datasets.append(sub_ds)
    
    ds = concatenate_datasets(all_datasets)
    print(f"There are {len(ds)} entries")

    return ds

def try_load_dataset(path):
    """Try to load a dataset using load_dataset or load_from_disk."""
    try:
        ds = load_from_disk(path)
        return ds
    except Exception:
        try:
            ds = load_dataset(path)
            return ds
        except Exception:
            try: 
                ds = try_load_directory_of_jsons(path)
                return ds
            except Exception as e:
                raise RuntimeError(f"Could not load dataset from {path}: {e}")

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
        dataset = try_load_dataset(dataset_input_path)
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
        """.replace("  ", "")
        )
    print(TEST_SPLIT_SIZE, DEV_SPLIT_SIZE, RANDOM_SEED)  
    
    convert_dataset(
        args.input_path,
        args.output_path
    )


if __name__ == "__main__":
    main_convert()