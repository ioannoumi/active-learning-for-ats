from active_learning.loss_with_random_active_learner import LossWithRandomActiveLearner
from active_learning.idds_active_learner import IDDSActiveLearner
import logging
import torch
from typing import List


class LossWithIDDSActiveLearner(LossWithRandomActiveLearner, IDDSActiveLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquired_samples_neighbor_idxs = []
        self.idds_neighborhood_size = 100

    # def perform_warmup(self):
    #     logging.info("LossWithRandomActiveLearnerVAR2 requires warmup, performing warmup using the IDDDS strategy...")
    #     selected_idxs = IDDSActiveLearner.select_idxs(self)
    #     self.data_handler.update_labeled_idxs(selected_idxs)
    #     self.train_and_evaluate()
    #     self._write_data_to_file(selected_idxs, '/selected_idxs.json')
    #     self._log_examples_json(selected_idxs)
    
    def _get_candiadate_neighborhood(self,acquired_idxs:List[int]) -> List[int]:
        idds_dataset_score = self._compute_idds_scores(self.acquired_samples_neighbor_idxs)
        idds_asc_scores_idxs = idds_dataset_score.argsort()
        idds_unlabeled_neighborhood__idxs = idds_asc_scores_idxs[-self.idds_neighborhood_size:].tolist()
        return idds_unlabeled_neighborhood__idxs

    def select_idxs(self) -> List[int]:
        safe_neighborhood_idxs = self._get_candiadate_neighborhood(self.acquired_samples_neighbor_idxs)
        assert len(safe_neighborhood_idxs) == self.idds_neighborhood_size

        loss_calculation_dataloader = self._get_data_after_training()
        loss_tuples = self._per_example_loss(self.model,loss_calculation_dataloader)
        dict_loss_tuples = dict(loss_tuples)

        hard_examples = self._get_topk_hard_examples(loss_tuples,self.active_learning_cfg.hard_examples_topk)
        hard_idxs = torch.tensor([idx for idx, _ in hard_examples], device=self.embeddings.device)

        hard_embeddings = self.embeddings[hard_idxs]
        safe_neighborhood_embeddings = self.embeddings[safe_neighborhood_idxs]

        acquired_idxs = self._select_similar_unlabeled(
        hard_idxs,
        hard_embeddings,
        safe_neighborhood_embeddings,
        torch.tensor(safe_neighborhood_idxs, device=self.embeddings.device),
        self.active_learning_cfg.samples_per_iteration)

        self.acquired_samples_neighbor_idxs.extend(safe_neighborhood_idxs)
        self._log_and_save_metrics(dict_loss_tuples, hard_examples, acquired_idxs)

        return acquired_idxs.tolist()


    # def select_idxs(self) -> List[int]:
    #     random_per = self._random_samples_per()
    #     len_random = int(random_per * self.active_learning_cfg.samples_per_iteration) 

    #     acquired_idxs = []
    #     for i in range(self.active_learning_cfg.samples_per_iteration - len_random):
    #         logging.info(f"---Acquiring sample {i+1}---")
            
    #         #labeled_samples_scores will be 0
    #         idds_unlabeled_neighborhood_idxs = self._get_candiadate_neighborhood(acquired_idxs)
            
            