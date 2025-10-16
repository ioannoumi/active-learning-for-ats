from dataclasses import dataclass
from typing import Optional

@dataclass
class ActiveLearningConfig:
    strategy: str
    iterations: int
    samples_per_iteration: int
    warmup_samples: int = 0
    hard_examples_topk: int = 0
