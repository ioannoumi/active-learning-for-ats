from dataclasses import dataclass

@dataclass
class DatasetConfig():
    source: str
    text_col: str
    target_col: str
    train_split: str
    val_split: str
    test_split: str
    max_source_length: int
    max_target_length: int
    train_size: int = None
    val_size: int = None
    test_size: int = None