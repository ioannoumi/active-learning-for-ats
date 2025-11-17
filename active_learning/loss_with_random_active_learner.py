from active_learning.loss_active_learner import LossActiveLearner

class LossWithRandomActiveLearner(LossActiveLearner):
    def select_idxs(self):
        random_per = self._random_samples_per()
        len_random = int(random_per * self.active_learning_cfg.samples_per_iteration)

        loss_calculation_dataloader = self._get_data_after_training()
        loss_tuples = self._per_example_loss(self.model,loss_calculation_dataloader)
        hard_examples = self._get_topk_hard_examples(loss_tuples,self.active_learning_cfg.hard_examples_topk)
        hard_idxs = [idx for idx,_ in hard_examples]

        unlabeled_idxs = self.data_handler.get_unlabeled_idxs()
        hard_embeddings = self.embeddings[hard_idxs]
        unlabeled_embeddings = self.embeddings[unlabeled_idxs]

        acquired_idxs = self._select_similar_unlabeled(
        hard_embeddings,
        unlabeled_embeddings,
        unlabeled_idxs,
        self.active_learning_cfg.samples_per_iteration - len_random)
        
        exclude_set = set(acquired_idxs)
        random_idxs = []

        while len(random_idxs) < len_random:
            _,random_idx = self.data_handler.sample_from_unlabeled(1)
            random_idx = random_idx[0]
            if random_idx not in exclude_set:
                random_idxs.append(random_idx)
                exclude_set.add(random_idx)
        print(acquired_idxs+random_idxs)
        self._log_and_save_metrics(hard_examples, acquired_idxs+random_idxs)

        return acquired_idxs + random_idxs
        
    def _random_samples_per(self):
        return 0.5