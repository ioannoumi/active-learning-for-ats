import random
from typing import List
from active_learning.base_active_learner import BaseActiveLearner


class RandomActiveLearner(BaseActiveLearner):
    def needs_embeddings(self):
        return False
    def needs_warmup(self):
        return False
    def select_idxs(self) -> List[int]:
        _,sample_dataset_idxs = self.data_handler.sample_from_unlabeled(self.active_learning_cfg.samples_per_iteration)
        return sample_dataset_idxs
    