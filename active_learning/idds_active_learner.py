from typing import List
import logging
from active_learning.base_active_learner import BaseActiveLearner
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import torch.nn as nn
import numpy as np
import time

class IDDSActiveLearner(BaseActiveLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(self.embeddings).to(self.train_args.device)

        logging.info("IDDS: Starting one-time pre-computation of similarity sum...")
        start_time = time.time()
        self.similarities_sum = self._compute_embeddings_similarities_sum()
        end_time = time.time()
        precomputation_time = end_time - start_time
        logging.info(f"IDDS: Finished pre-computation in {precomputation_time:.2f} seconds.")
        self._write_data_to_file({"idds_precomputation_time_seconds": precomputation_time}, '/acquisition_times.json')

    def needs_embeddings(self):
        return True

    def needs_warmup(self):
        return False
    
    def select_idxs(self) -> List[int]:
        samples_per_iteration = self.active_learning_cfg.samples_per_iteration

        logging.info(f"Acquiring {samples_per_iteration} samples using IDDS")
        sample_dataset_idxs = []
        sample_dataset_scores = []

        for i in range(samples_per_iteration):
            idds_scores = self._compute_idds_scores(sample_dataset_idxs)
            max_score_idx = int(idds_scores.argmax().item())
            max_idds_score = idds_scores[max_score_idx].item()
            sample_dataset_idxs.append(max_score_idx)
            sample_dataset_scores.append(max_idds_score)
            logging.info(f"Selecting the document with the highest IDDS score (index: {max_score_idx}, score: {max_idds_score})")
            logging.info(f"For reference, the IDDS score of the document with the lowest score is {idds_scores.min()}")
        
        IDDSActiveLearner._log_metrics_for_samples(self, sample_dataset_idxs, sample_dataset_scores, "ActiveLearningStrategyIDDS")

        return sample_dataset_idxs

    def _compute_embeddings_similarities_sum(self,batch_size = 7500) -> None:
        dataset_len = len(self.embeddings)
        similarities_sum = torch.zeros(dataset_len).to(self.train_args.device)
        tensor_dataset = TensorDataset(self.embeddings)
        tensor_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,shuffle =False)

        for index, (batch_embeddings,) in tqdm(enumerate(tensor_dataloader)):
            similarities = torch.mm(batch_embeddings, self.embeddings.t())
            assert(similarities.shape == (len(batch_embeddings), dataset_len))

            similarities_sum[index * batch_size : (index+1) * batch_size] = similarities.sum(dim=1)

        documents = self.data_handler.get_dataset()[self.dataset_cfg.text_col]
        logging.info(f"Document with lowest similarity sum: {documents[similarities_sum.argmin().item()]}")
        logging.info(f"Document with highest similarity sum: {documents[similarities_sum.argmax().item()]}")

        return similarities_sum
    
    def _compute_idds_scores(self, new_labeled_idxs: List[int]) -> torch.Tensor:
        dataset_len = len(self.embeddings)
        labeled_idxs = self.data_handler.get_labeled_idxs() + new_labeled_idxs
        labeled_count = len(labeled_idxs)
        labeled_embeddings = self.embeddings[labeled_idxs]

        if len(labeled_embeddings) > 0:
            similarities_with_labeled = torch.mm(self.embeddings, labeled_embeddings.T)
            assert(similarities_with_labeled.shape == (dataset_len, labeled_count))

            similarities_sum_with_labeled = torch.sum(similarities_with_labeled, dim=1).to(self.train_args.device)
        else:
            similarities_sum_with_labeled = torch.zeros(dataset_len).to(self.train_args.device)
        
        similarities_sum_with_unlabeled = self.similarities_sum - similarities_sum_with_labeled

        unlabeled_count = dataset_len - labeled_count
        unlabeled_scores = similarities_sum_with_unlabeled / unlabeled_count

        labeled_scores = similarities_sum_with_labeled / (labeled_count if labeled_count>0 else 1)

        idds_scores = 0.66 * unlabeled_scores - 0.33 * labeled_scores
        assert idds_scores.shape == (dataset_len,)

        idds_scores[labeled_idxs] = 0
        return idds_scores


    def _log_metrics_for_samples(self, sample_dataset_idxs: List[int], sample_dataset_scores: List[float], source: str) -> None:
        sample_embeddings = self.embeddings[sample_dataset_idxs]
        sample_embeddings_variance = self._compute_embeddings_variance(sample_embeddings)
        sample_embeddings_avg_pairwise_distance = self._compute_embeddings_avg_pairwise_distance(sample_embeddings)
        sample_embeddings_avg_cosine_similarity = self._compute_embeddings_avg_cosine_similarity(sample_embeddings)
        sample_embeddings_avg_distance_to_centroid = self._compute_embeddings_avg_distance_to_centroid(sample_embeddings)
        json_metrics = json.dumps({
            "source": source,
            "variance": sample_embeddings_variance,
            "avg_pairwise_dist": sample_embeddings_avg_pairwise_distance,
            "avg_cosine_similarity": sample_embeddings_avg_cosine_similarity,
            "avg_distance_to_centroid": sample_embeddings_avg_distance_to_centroid,
        })
        with open(self.train_args.output_dir + "/idds_embeddings_stats.json", "a") as outfile:
            outfile.write(json_metrics + '\n')

        index_and_scores = [{"index": idx, "score": score} for idx, score in zip(sample_dataset_idxs, sample_dataset_scores)]
        json_scores = json.dumps({
            "source": source,
            "index_and_scores": index_and_scores,
        })
        with open(self.train_args.output_dir + "/idds_scores.json", "a") as outfile:
            outfile.write(json_scores + '\n')

    def _compute_embeddings_variance(self, embeddings: torch.Tensor) -> float:
        return embeddings.var(dim=0).mean().item()
    
    def _compute_embeddings_avg_pairwise_distance(self, embeddings: torch.Tensor) -> float:
        return torch.nn.functional.pdist(embeddings).mean().item()
    
    def _compute_embeddings_avg_cosine_similarity(self, embeddings: torch.Tensor) -> float:
        similarities = torch.mm(embeddings, embeddings.T)
        return similarities.mean().item()

    def _compute_embeddings_avg_distance_to_centroid(self, embeddings: torch.Tensor) -> float:
        centroid = embeddings.mean(dim=0)
        distances = torch.nn.functional.pairwise_distance(embeddings, centroid)
        return distances.mean().item()