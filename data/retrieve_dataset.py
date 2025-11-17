from configs.dataset_config import DatasetConfig
from datasets import DatasetDict,load_dataset, Dataset
from data.indexing import add_index_column
import pandas as pd
import logging

def retrieve_dataset(cfg: DatasetConfig, is_parquet = False) -> tuple[Dataset,Dataset,Dataset]:
    
    ds = load_dataset(path = 'parquet' if is_parquet else cfg.source, data_files= cfg.source if is_parquet else None)
    ds = _filter_nulls(ds, cfg)

    if len(ds) == 1:
        logging.info("Dataset has only one split. Splitting into train, validation, and test sets.")
        train_and_temp_dataset = ds[cfg.train_split].train_test_split(test_size=0.2,shuffle=True,seed=42)
        train = train_and_temp_dataset['train']
        val_and_test_dataset = train_and_temp_dataset['test'].train_test_split(test_size=0.5,shuffle=True,seed=42)
        val= val_and_test_dataset['train']
        test = val_and_test_dataset['test']
    
    else:
        train = ds[cfg.train_split]
        val   = ds[cfg.val_split]
        test  = ds[cfg.test_split]

    train = _safe_select(train, getattr(cfg,'train_size',None))
    val = _safe_select(val, getattr(cfg,'val_size',None))
    test = _safe_select(test, getattr(cfg,'test_size',None))

    train = _deduplicate(train)
    train = add_index_column(train)

    return train,val,test

def _filter_nulls(ds_dict: DatasetDict, cfg: DatasetConfig) -> DatasetDict:
    for split_name, dataset in ds_dict.items():
        initial_len = len(dataset)
        
        filtered_dataset = dataset.filter(lambda ex: ex[cfg.text_col] and ex[cfg.target_col])
        filtered_len = len(filtered_dataset)
        if initial_len > filtered_len:
            removed_count = initial_len - filtered_len
            logging.warning(f'[{split_name} split] Filtered out {removed_count} null or empty examples.')
        ds_dict[split_name] = filtered_dataset
    return ds_dict

def _deduplicate(train_dataset: Dataset) -> Dataset:
    train_dataset = train_dataset.to_pandas()
    original_len = len(train_dataset)

    train_dataset = train_dataset.drop_duplicates(keep='first')
    new_len = len(train_dataset)

    if original_len > new_len:
        logging.warning((f"Deduplication: Removed {original_len-new_len} duplicates."))
    
    train_dataset = train_dataset.reset_index(drop=True)
    return Dataset.from_pandas(train_dataset, preserve_index=False)

def _safe_select(ds: Dataset, n: int = None):
    if n is None or n >= len(ds):
        return ds
    return ds.select(range(n))