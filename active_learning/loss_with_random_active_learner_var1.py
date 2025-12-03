from active_learning.loss_with_random_active_learner import LossWithRandomActiveLearner
import torch
import logging


class LossWithRandomActiveLearnerVAR1(LossWithRandomActiveLearner):
    def _select_similar_unlabeled(self, hard_embeddings, unlabeled_embs, unlabeled_idxs_tensor, num_to_select):
        print('LOSS_VARIANT1 CALLED')
        similarities = torch.mm(hard_embeddings, unlabeled_embs.T)
        scores = torch.max(similarities, dim=0).values

        scores[scores >= 0.95] = -1.0
        selected_idxs = []

        for _ in range(num_to_select):
            best_idx = torch.argmax(scores).item()

            if scores[best_idx] == -1.0:
                logging.warning(f"MMR stopped early. Found {len(selected_idxs)}/{num_to_select} diverse neighbors.")
                break
            
            selected_idxs.append(best_idx)
            scores[best_idx] = -1.0

            best_embedding = unlabeled_embs[best_idx].unsqueeze(0)
            similarities_to_winner = torch.mm(best_embedding, unlabeled_embs.T).squeeze(0)
            
            scores[similarities_to_winner >= 0.90] = -1.0

        return unlabeled_idxs_tensor[selected_idxs]