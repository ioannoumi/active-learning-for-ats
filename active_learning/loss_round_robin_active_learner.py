from active_learning.loss_with_random_active_learner import LossWithRandomActiveLearner
import torch

class LossRoundRobinActiveLearner(LossWithRandomActiveLearner):
    
    def _select_similar_unlabeled(self,hard_idxs: torch.Tensor,hard_embeddings: torch.Tensor, unlabeled_embs: torch.Tensor,unlabeled_idxs_tensor: torch.Tensor,num_to_select: int) -> torch.Tensor:
        num_of_hard_examples = len(hard_idxs)
        similarities = torch.mm(hard_embeddings, unlabeled_embs.T)
        selected_idxs = []
        selected_mask = torch.zeros(unlabeled_idxs_tensor.shape[0], dtype=torch.bool, device=unlabeled_idxs_tensor.device)

        responsible_idxs = []

        while len(selected_idxs) < num_to_select:
            for i in range(num_of_hard_examples):
                if len(selected_idxs) == num_to_select:
                    break
                
                row_similarities = similarities[i]
                row_similarities[selected_mask] = float('-inf')
                
                best_match_idx = torch.argmax(row_similarities)
                best_match_similarity = row_similarities[best_match_idx].item()

                if best_match_similarity == float('-inf'):
                    continue

                if best_match_similarity >= 0.95:
                    selected_mask[best_match_idx] = True
                    continue

                selected_idxs.append(unlabeled_idxs_tensor[best_match_idx].item())
                selected_mask[best_match_idx] = True

                responsible_idxs.append(hard_idxs[i].item())
        
        selection_log = [ {"selected_unlabeled_idx": selected, "responsible_hard_idx": responsible}   for selected, responsible in zip(selected_idxs, responsible_idxs)]

        self.log_responsible_examples(selection_log, '/responsible_hard_examples.json')

        return torch.tensor(selected_idxs, device=unlabeled_idxs_tensor.device)