from configs.dataset_config import DatasetConfig
from datasets import DatasetDict,load_dataset, Dataset
from data.indexing import add_index_column

def retrieve_dataset(cfg: DatasetConfig) -> tuple[Dataset,Dataset,Dataset]:
    
    ds = load_dataset(cfg.source)

    train = ds[cfg.train_split]
    val   = ds[cfg.val_split]
    test  = ds[cfg.test_split]

    train = _safe_select(train, getattr(cfg,'train_size',None))
    val = _safe_select(val, getattr(cfg,'val_size',None))
    test = _safe_select(test, getattr(cfg,'test_size',None))

    train = add_index_column(train)

    return train,val,test

def _safe_select(ds: Dataset, n: int = None):
    if n is None or n >= len(ds):
        return ds
    return ds.select(range(n))