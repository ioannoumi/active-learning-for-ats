from datasets import Dataset
import random

class ActiveLearningDatasetHandler():
    def __init__(self,dataset: Dataset,seed = 42):
        self.rn = random.Random(seed)
        self.dataset = dataset
        self.labeled_idxs = set()

    def sample_from_unlabelled(self,num_samples):
        unlabeled_idxs = self.get_unlabeled_idxs()
        self.rn.shuffle(unlabeled_idxs)
        return unlabeled_idxs[:num_samples]

    def update_labeled_idxs(self,acquired_idxs):
        self.labeled_idxs.update(acquired_idxs)

    def get_unlabeled_idxs(self):
        return [i for i in range(len(self.dataset)) if i not in self.labeled_idxs]

    def get_labeled_idxs(self):
        return sorted(self.labeled_idxs)

    def get_labeled_data(self):
      return self.dataset.select(sorted(self.labeled_idxs))

    def get_unlabeled_data(self):
      return self.dataset.select(sorted(self.get_unlabeled_idxs()))
    
    def get_dataset(self):
      return self.dataset

    def summary(self):
      print(f"Labeled: {len(self.labeled_idxs)}, Unlabeled: {len(self.get_unlabeled_idxs())}")