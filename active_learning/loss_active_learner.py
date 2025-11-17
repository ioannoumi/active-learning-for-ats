from typing import List
from active_learning.base_active_learner import BaseActiveLearner
import torch
import torch.nn as nn
import logging
import json
import numpy as np
from torch.utils.data import DataLoader
from data.collators import IdDataCollator

class LossActiveLearner(BaseActiveLearner):

    def needs_embeddings(self):
        return True

    def needs_warmup(self):
        return True
    
    def select_idxs(self) -> List[int]:
        loss_calculation_dataloader = self._get_data_after_training()
        loss_tuples = self._per_example_loss(self.model,loss_calculation_dataloader)
        hard_examples = self._get_topk_hard_examples(loss_tuples,self.active_learning_cfg.hard_examples_topk)
        
        hard_idxs = torch.tensor([idx for idx, _ in hard_examples], device=self.embeddings.device)

        unlabeled_idxs = self.data_handler.get_unlabeled_idxs()
        hard_embeddings = self.embeddings[hard_idxs]
        
        unlabeled_idxs_tensor = torch.tensor(unlabeled_idxs, device=self.embeddings.device)
        unlabeled_embeddings = self.embeddings[unlabeled_idxs_tensor]

        acquired_idxs = self._select_similar_unlabeled(
            hard_embeddings,
            unlabeled_embeddings,
            unlabeled_idxs_tensor,
            self.active_learning_cfg.samples_per_iteration)

        self._log_and_save_metrics(hard_examples, acquired_idxs)
        return acquired_idxs.tolist()

    def _get_data_after_training(self) -> DataLoader:
        id_collator = IdDataCollator(self.tok, model=self.model)
        loss_calculation_dataloader = DataLoader(
        self.tokenized_train_dataset,
        batch_size=self.train_args.per_device_train_batch_size, 
        collate_fn=id_collator,
        shuffle=False)
        return loss_calculation_dataloader

    @torch.no_grad()
    def _per_example_loss(self,model,dataloader,ignore_index = -100): 
        model.eval()
        ce = nn.CrossEntropyLoss(reduction="none", ignore_index= ignore_index)
        results = []

        device = next(model.parameters()).device 
        for batch in dataloader:

            labels = batch['labels'].to(device).long()

            outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=labels,
            output_hidden_states=True,
            return_dict=True
            )

            logits = outputs.logits # (B,T,V)

            assert logits.shape[0] == labels.shape[0], f"Batch size mismatch: {logits.shape[0]} != {labels.shape[0]}"
            assert logits.shape[1] == labels.shape[1], f"Seq len mismatch: {logits.shape[1]} != {labels.shape[1]}"

            per_token = ce(logits.reshape(-1,logits.size(-1)),labels.reshape(-1)).reshape(labels.size())

            assert per_token.shape == labels.shape, "Loss shape does not match labels shape"

            valid = (labels != ignore_index)
            num_valid = valid.sum(dim=1)
            assert torch.all(num_valid > 0), "Found sequence with 0 valid tokens, will cause division by zero."
            per_seq = (per_token * valid).sum(dim=1) / num_valid

            batch_size = labels.shape[0]
            idxs = batch['ids']
            for i in range(batch_size):
                loss = per_seq[i].item()
                idx = idxs[i]
                results.append((idx,loss))

        return results
  
    def _get_topk_hard_examples(self,loss_tuples,k) -> list[tuple[int,float]]:    
        loss_tuples.sort(key=lambda x: x[1], reverse=True)
        hard_examples = loss_tuples[:k]
        return hard_examples #(IDX,LOSS)

    def _select_similar_unlabeled(self,hard_embeddings: torch.Tensor, unlabeled_embs: torch.Tensor,unlabeled_idxs_tensor: torch.Tensor,num_to_select: int) -> torch.Tensor:
        similarities = torch.mm(hard_embeddings, unlabeled_embs.T)
        max_sim_scores, _ = torch.max(similarities, dim=0)
        sorted_indices = torch.argsort(max_sim_scores, descending=True)[:num_to_select]

        return unlabeled_idxs_tensor[sorted_indices]
    
    def _get_batch_similarity_metrics(self,embeddings: torch.Tensor) -> dict:
        avg_similarity = self._compute_batch_avg_cosine_similarity(embeddings)
        nn_avg_similarity = self._nn_avg_cosine(embeddings)
        return {
            'avg_similarity': avg_similarity,
            'nn_avg_similarity': nn_avg_similarity,
        }
    
    def _compute_batch_avg_cosine_similarity(self,embeddings: torch.Tensor) -> float:
        similarities = torch.mm(embeddings, embeddings.T)
        n = similarities.shape[0]
        iu = torch.triu_indices(n, n, offset=1, device=embeddings.device)
        return similarities[iu[0], iu[1]].mean().item()
    
    def _nn_avg_cosine(self, embeddings: torch.Tensor) -> float:
        """
        on average, how similar is each vector to its closest neighbor in the batch?
        """
        similarities = torch.mm(embeddings, embeddings.T)
        similarities.fill_diagonal_(float('-inf'))
        return similarities.max(dim=1).values.mean().item()
    
    def _log_and_save_metrics(self, hard_examples: list[tuple[int, float]], acquired_idxs: list[int]) -> None:
        self._log_hard_examples_json(hard_examples)
        self._write_data_to_file(dict(hard_examples), f'/top{self.active_learning_cfg.hard_examples_topk}_example_losses.json')
        acquired_embeddings = self.embeddings[acquired_idxs]
        batch_similarity_metrics = self._get_batch_similarity_metrics(acquired_embeddings)
        self._write_data_to_file(batch_similarity_metrics, '/batch_similarity_metrics.json')

    def _log_hard_examples_json(self, hard_examples: list[tuple[int, float]]) -> None:
        documents = self.data_handler.get_dataset()
        
        logging.info('\n')
        for rank, (idx,loss) in enumerate(hard_examples,1):
            rec = documents[idx]
            logging.info(json.dumps({
            "tag": "AL_HARD",
            "rank": rank,
            "id": idx,
            "loss": round(loss, 6),
            "doc_len": len(rec[self.dataset_cfg.text_col]),
            "summ_len": len(rec[self.dataset_cfg.target_col]),
            "doc_preview": str(rec[self.dataset_cfg.text_col]).replace("\n", " ")[:300],
            "summ_preview": str(rec[self.dataset_cfg.target_col]).replace("\n", " ")[:200],
        }, ensure_ascii=False))
        logging.info('\n')