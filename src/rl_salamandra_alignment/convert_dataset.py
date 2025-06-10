"""Script to convert HF datasets to a format that can be run locally in MN5"""
from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk
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
        dataset = load_dataset(dataset_input_path)
    except:
        pass

    try:
        dataset = load_from_disk(dataset_input_path)
    except:
        pass

    if dataset:
        # create test split if not found
        try:
            dataset["test"]
        except:
            logger.info(
                f"Creating a test split for {os.path.basename(dataset_input_path)}")
            dataset = dataset["train"].train_test_split(
                test_size=0.1,
                seed=42
            )
        output_path = dataset_output_path

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
    args = parser.parse_args()
    convert_dataset(
        args.input_path,
        args.output_path
    )


if __name__ == "__main__":
    main_convert()