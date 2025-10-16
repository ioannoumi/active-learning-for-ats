from datasets import Dataset

def add_index_column(dataset: Dataset, column_name: str = 'idxs') -> Dataset:
    return dataset.add_column(column_name, list(range(len(dataset))))
