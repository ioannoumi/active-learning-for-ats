from active_learning.idds_active_learner import IDDSActiveLearner
from active_learning.bas_active_learner import BASActiveLearner
from active_learning.base_active_learner import BaseActiveLearner
import numpy as np
from typing import List
import logging
import json

class DUALActiveLearner(IDDSActiveLearner, BASActiveLearner, BaseActiveLearner):
    def __init__(self, *args, **kwargs):
        IDDSActiveLearner.__init__(self, *args, **kwargs)
        BASActiveLearner.__init__(self, *args, **kwargs)
        self.acquired_samples_neighbor_idxs = []

    def needs_embeddings(self):
        return True

    def needs_warmup(self):
        return True

    def select_idxs(self) -> List[int]:
        bas_num_samples_to_rank = self.active_learning_cfg.bas_num_samples_to_rank
        active_learning_samples_per_iteration = self.active_learning_cfg.samples_per_iteration

        acquired_samples_idxs = []
        for i in range(active_learning_samples_per_iteration):
            logging.info(f"---Acquiring sample {i+1}---")
            if i < 0.5 * active_learning_samples_per_iteration:
                logging.info(f"Computing IDDS scores and selecting the top {bas_num_samples_to_rank} highest scoring documents")
                assume_already_labelled_idxs = list(set(acquired_samples_idxs + self.acquired_samples_neighbor_idxs))
                idds_scores = self._compute_idds_scores(assume_already_labelled_idxs)
                logging.info(f"Max IDDS score: {idds_scores.max()}")
                logging.info(f"Min IDDS score: {idds_scores.min()}")
                idds_asc_scores_idxs = idds_scores.argsort()
                idds_sample_dataset_idxs = idds_asc_scores_idxs[-bas_num_samples_to_rank:].tolist()
                logging.info(f"Computing BLEUvar scores for the selected documents")
                bleuvars = self.compute_bleuvar_scores(idds_sample_dataset_idxs)
            
                self._log_metrics_for_samples(idds_sample_dataset_idxs, idds_scores[idds_sample_dataset_idxs].tolist(), bleuvars, "ActiveLearningStrategyDUAL")
            
                bleuvars = np.array(bleuvars)
                bas_asc_scores_idxs = bleuvars.argsort()
                bas_asc_scores_idxs = [idx for idx in bas_asc_scores_idxs if bleuvars[idx] < 0.96] # threshold filter
                highest_score_idx = bas_asc_scores_idxs[-1]
                idx = idds_sample_dataset_idxs[highest_score_idx]
                logging.info(f"Selecting the document with the highest BLEUvar score (index: {idx}, score: {bleuvars[highest_score_idx]})")

                self.acquired_samples_neighbor_idxs.extend(idds_sample_dataset_idxs)
                self.acquired_samples_neighbor_idxs.remove(idx)
            else:
                _, idxs = self.data_handler.sample_from_unlabeled(1)
                idx = idxs[0]
                logging.info(f"Selecting random document (index: {idx})")

            acquired_samples_idxs.append(idx)

        return acquired_samples_idxs
    
    def _log_metrics_for_samples(self, sample_dataset_idxs: List[int], sample_dataset_scores: List[float], bleuvars: List[float], source: str):
        super()._log_metrics_for_samples(sample_dataset_idxs, sample_dataset_scores, source)

        # Log BLEUvar scores
        # We expect to have relatively high variance, because the BLEUVar distribution 
        # does not necessarily match the embeddings variance distribution
        logging.info(f"BLEUvar scores: {bleuvars}")
        json_scores = json.dumps({
            "source": source,
            "bleuvar_scores": bleuvars,
        })
        with open(self.train_args.output_dir + "/bleuvar_scores.json", "a") as outfile:
            outfile.write(json_scores + '\n')
