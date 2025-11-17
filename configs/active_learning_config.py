from dataclasses import dataclass
from typing import Optional

@dataclass
class ActiveLearningConfig:
    strategy: str
    iterations: int
    samples_per_iteration: int
    warmup_samples: int = 10
    hard_examples_topk: int = 0
    bas_num_samples_to_rank: int = 0
    bas_num_samples_mc_dropout: int = 0
