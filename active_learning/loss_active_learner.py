from typing import List
from active_learning.base_active_learner import BaseActiveLearner
import torch
import torch.nn as nn
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
        hard_idxs = [idx for idx,_ in hard_examples]
        # print(hard_idxs)

        unlabeled_idxs = self.data_handler.get_unlabeled_idxs()
        hard_embeddings = self.embeddings[hard_idxs]
        unlabelled_embeddings = self.embeddings[unlabeled_idxs]

        acquired_idxs = self._select_similar_unlabeled(
        hard_embeddings,
        unlabelled_embeddings,
        unlabeled_idxs,
        self.active_learning_cfg.samples_per_iteration)

        return acquired_idxs

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
        
            # print('===Logits shape:===')
            # print(shift_logits.shape)
            # print('===Labels shape:===')
            # print(shift_labels.shape)

            per_token = ce(logits.reshape(-1,logits.size(-1)),labels.reshape(-1)).reshape(labels.size())

            # print('Loss per token shape (should be B,T-1)')
            # print(per_token.shape)

            valid = (labels != ignore_index)
            num_valid = valid.sum(dim=1)
            per_seq = (per_token * valid).sum(dim=1) / num_valid

            batch_size = labels.shape[0]
            idxs = batch['ids']
            for i in range(batch_size):
                loss = per_seq[i].item()
                idx = idxs[i]
                results.append((idx,loss))

        return results
  
    def _get_topk_hard_examples(self,loss_tuples,k):
        #The parameter k will  later be passes as arg
    
        loss_tuples.sort(key=lambda x: x[1], reverse=True)
        hard_examples = loss_tuples[:k]
        return hard_examples #(IDX,LOSS)

    def _select_similar_unlabeled(self,hard_embeddings, unlabeled_embs,unlabeled_idxs,num_to_select):
        similarities = np.dot(hard_embeddings, unlabeled_embs.T)
        max_sim_scores = np.max(similarities, axis = 0)
        sorted_indices = np.argsort(max_sim_scores)[::-1][:num_to_select]

        unlabeled_idxs_chosen = [unlabeled_idxs[idx] for idx in sorted_indices]
        return unlabeled_idxs_chosen

    def _get_data_after_training(self) -> DataLoader:
        id_collator = IdDataCollator(self.tok, model=self.model)
        loss_calculation_dataloader = DataLoader(
        self.tokenized_train_dataset,
        batch_size=self.train_args.per_device_train_batch_size, 
        collate_fn=id_collator,
        shuffle=False)
        return loss_calculation_dataloader
    