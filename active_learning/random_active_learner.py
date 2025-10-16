import random
from typing import List
from active_learning.base_active_learner import BaseActiveLearner


class RandomActiveLearner(BaseActiveLearner):
    def needs_embeddings(self):
        return False
    def needs_warmup(self):
        return False
    def select_idxs(self) -> List[int]:
        return self.data_handler.sample_from_unlabelled(self.active_learning_cfg.samples_per_iteration)
    