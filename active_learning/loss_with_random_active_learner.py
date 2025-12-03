from active_learning.loss_active_learner import LossActiveLearner
import torch

class LossWithRandomActiveLearner(LossActiveLearner):
    def select_idxs(self):
        random_per = self._random_samples_per()
        len_random = int(random_per * self.active_learning_cfg.samples_per_iteration)

        loss_calculation_dataloader = self._get_data_after_training()
        loss_tuples = self._per_example_loss(self.model,loss_calculation_dataloader)
        dict_losses = dict(loss_tuples)

        hard_examples = self._get_topk_hard_examples(loss_tuples,self.active_learning_cfg.hard_examples_topk)
        hard_idxs = torch.tensor([idx for idx, _ in hard_examples], device=self.embeddings.device)

        unlabeled_idxs = self.data_handler.get_unlabeled_idxs()
        hard_embeddings = self.embeddings[hard_idxs]
        unlabeled_idxs_tensor = torch.tensor(unlabeled_idxs, device=self.embeddings.device)
        unlabeled_embeddings = self.embeddings[unlabeled_idxs_tensor]

        acquired_idxs = self._select_similar_unlabeled(
        hard_idxs,
        hard_embeddings,
        unlabeled_embeddings,
        unlabeled_idxs_tensor,
        self.active_learning_cfg.samples_per_iteration - len_random)
        
        acquired_idxs = acquired_idxs.tolist()
        total_number_of_examples = self.active_learning_cfg.samples_per_iteration
        number_of_examples_needed = total_number_of_examples - len(acquired_idxs)

        exclude_set = set(acquired_idxs)
        random_idxs = []

        while len(random_idxs) < number_of_examples_needed:
            _,random_idx = self.data_handler.sample_from_unlabeled(1)
            random_idx = random_idx[0]
            if random_idx not in exclude_set:
                random_idxs.append(random_idx)
                exclude_set.add(random_idx)
        print(acquired_idxs+random_idxs)
        self._log_and_save_metrics(dict_losses,hard_examples, acquired_idxs)

        return acquired_idxs + random_idxs
        
    def _random_samples_per(self):
        return 0.5