"""Script to convert HF datasets to a format that can be run locally in MN5"""
from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
import os
from rl_salamandra_alignment import logger
import json
from tqdm import tqdm

DATASET_KEYS_FOR_RL = [
    "prompt",
    "chosen",
    "rejected",
    "messages",
    "completions",
    "completion",
    "label"
]

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

def determine_dataset_columns(dataset_list: list[Dataset])-> list[str]:
    """Given a list of datasets, return the column names common to _all_ datasets, _and_ only those required for RL alignment.

    Args:
        dataset_list (list[Dataset]): _description_

    Returns:
        list[str]: _description_
    """
    print("Determining column names")
    original_column_names = [
        dataset.column_names for dataset in dataset_list
    ]
    # Get all unique elements from first list
    print(
    "Your datasets have the following keys:\n" + "\n".join([str(sublist) for sublist in original_column_names])
    )
    unique_elements = set(original_column_names[0])

    # Check which elements exist in all lists
    common_columns = [
        elem
        for elem in unique_elements
        if all(elem in sublist for sublist in original_column_names)
    ]
    columns_for_rl = [
        elem
        for elem in common_columns
        if  elem in DATASET_KEYS_FOR_RL
    ]
    

    if not columns_for_rl:
        raise ValueError(
            f"Your datasets do not have the column names needed for Reinforcement Learning.\nAre any of your datasets missing any of these columns?\n   {DATASET_KEYS_FOR_RL}\n\n"
        )
    return columns_for_rl

def try_load_directory_of_jsons(path:str):
    filenames = os.listdir(path)
    jsons = [filename for filename in filenames if filename.endswith(".json")]
    jsonls = [filename for filename in filenames if filename.endswith(".jsonl")]
    if not jsons and not jsonls:
        raise ValueError(f"No json/jsonl files found in {path}")
    all_datasets = []
    tqdm_bar = tqdm(jsons)
    for json in tqdm_bar:
        tqdm_bar.set_description(f"Loading: {json}")
        sub_data = read_json(os.path.join(path, json))
        sub_ds = Dataset.from_list(sub_data)
        all_datasets.append(sub_ds)

    tqdm_bar = tqdm(jsonls)
    for jsonl in tqdm_bar:
        tqdm_bar.set_description(f"Loading: {jsonl}")
        sub_data = read_jsonl(os.path.join(path, jsonl))
        sub_ds = Dataset.from_list(sub_data)
        all_datasets.append(sub_ds)
    
    dataset_columns = determine_dataset_columns(
        all_datasets
    )
    print(dataset_columns)

    all_datasets_new = [
        dataset.select_columns(dataset_columns)
        for dataset in all_datasets
    ]
    print(all_datasets_new)
    import time
    time.sleep(10)
    try:
        ds = concatenate_datasets(all_datasets_new)
    except Exception as e:
        print(e)
        raise e
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
        print(f"Raw Dataset loaded!")
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
            if DEV_SPLIT_SIZE == 0:
                dataset = DatasetDict({"train": full_train_split, "test": dataset["test"]})
            else:
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
            if TEST_SPLIT_SIZE == 0:
                dataset = DatasetDict({"train": dataset["train"], "test": dataset["test"]})
            else:
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
            if TEST_SPLIT_SIZE == 0 and DEV_SPLIT_SIZE == 0:
                print("Producing only `train` split, because test and dev size were set to zero")
                dataset = DatasetDict(
                    {"train": full_train_split}
                )
            elif TEST_SPLIT_SIZE == 0:
                print("Producing only `train` and `dev` splits, because test size was set to zero")
                train_dev = full_train_split.train_test_split(
                    test_size = eval_size,
                    shuffle = True,
                    seed = RANDOM_SEED
                )
                dataset = DatasetDict(
                    {"train": train_dev["train"], "dev": train_dev["test"]}
                )
            elif DEV_SPLIT_SIZE == 0:
                print("Producing only `train` and `test` splits, because dev size was set to zero")
                train_test = full_train_split.train_test_split(
                    test_size=test_size,
                    shuffle=True,
                    seed=RANDOM_SEED
                )
                dataset = DatasetDict(
                    {"train": train_test["train"], "test": train_test["test"]}
                )
            else:
                print("Producing `train`, `test`, and `dev` splits")
                
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

def to_nemo_rl_format_sft(
    dataset_split: Dataset,
    output_path: str
)-> None:
    dataset_split.to_json(
        output_path,
        lines = True # make sure it is JSONL
    )

# Nemo-RL uses these indices to specify which answer is "chosen" and which is "rejected"
CHOSEN_RANK_INT = 0
REJECTED_RANK_INT = 1 

def format_content(content, role: str) -> list:
    """Convert content to openai format"""
    if isinstance(content, list): return content
    if isinstance(content, str):
        # Convert to
        return [{"role": role, "content": content}]
        
        
        
def preference_entry_to_nemo_rl(
    entry : dict
)-> dict:
    # Based on example from
    # https://github.com/NVIDIA-NeMo/RL/blob/d843f02e6dd4e7b0cd4879139e986e3bb2bd267d/docs/guides/dpo.md#direct-preference-optimization-in-nemo-rl
    entry["context"] = format_content(entry["prompt"], "user")
    entry["completions"] = [
        {
            "rank": CHOSEN_RANK_INT,
            "completion": format_content(entry["chosen"], "assistant")
        },
        {
            "rank": REJECTED_RANK_INT,
            "completion": format_content(entry["rejected"], "assistant")
        }
    ]
    return entry
        
def to_nemo_rl_format_preference(
    dataset_split: Dataset,
    output_path: str
)-> None:
    rl_format_preference = dataset_split.map(
        preference_entry_to_nemo_rl
    )
    rl_format_preference.to_json(
        output_path,
        lines = True # make sure it is JSONL
    )
    
    
def convert_to_nemo_rl_format(
    dataset_path : str
)-> None:
    """Convert HF Dataset to the format needed for the Nemo-RL framework.
    Note that SFT datasets and DPO datasets are handled differently.

    Args:
        dataset_path (str): _description_
    """
    
    dataset_dict = load_from_disk(dataset_path)
    dataset_columns = dataset_dict["train"].column_names
    if "messages" in dataset_columns:
        print(f"Detected SFT Dataset")
        # In this case, simply convert to JSONL
        for split_name in dataset_dict:
            split = dataset_dict[split_name]
            to_nemo_rl_format_sft(
                split,
                os.path.join(dataset_path, f"{split_name}.jsonl")
            )
        
    if "chosen" in dataset_columns and "rejected" in dataset_columns:
        print(f"Detected Preference Dataset")
        
        for split_name in dataset_dict:
            split = dataset_dict[split_name]
            to_nemo_rl_format_preference(
                split,
                os.path.join(dataset_path, f"{split_name}.jsonl")
            )
    
    

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
    parser.add_argument(
        "--to_nemo_rl",
        help="Convert to NemoRL dataset format",
        action="store_true",
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
    if args.to_nemo_rl:
        convert_to_nemo_rl_format(
            args.output_path
        )
        


if __name__ == "__main__":
    main_convert()