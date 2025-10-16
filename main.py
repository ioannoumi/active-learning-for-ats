from sentence_transformers import SentenceTransformer
from embeddings.get_embeddings import get_embeddings
from active_learning.active_learning_dataset_handler import ActiveLearningDatasetHandler
from active_learning.random_active_learner import RandomActiveLearner
from active_learning.loss_active_learner import LossActiveLearner
from configs.dataset_config import DatasetConfig
from data.retrieve_dataset import retrieve_dataset
from configs.training_config import TrainingConfig
from utils.get_args import get_train_args
import torch
from configs.active_learning_config import ActiveLearningConfig

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

datasets = [
    ('FiscalNote/billsum', 'text', 'summary'),
    ('Yale-LILY/aeslc', 'email_body','subject_line')
]

dataset = datasets[1]
dataset_cfg = DatasetConfig(
    source= dataset[0],
    text_col= dataset[1], target_col= dataset[2],
    train_split= 'train', val_split= 'validation',test_split= 'test',
    max_source_length=512,max_target_length=32
)

train_dataset,val_dataset,eval_dataset = retrieve_dataset(dataset_cfg)
training_cfg = TrainingConfig()
train_args = get_train_args(training_cfg)

active_learning_cfg = ActiveLearningConfig(
    strategy= 'Random Strategy',
    iterations= 15,
    samples_per_iteration=10,
    warmup_samples=16,
    hard_examples_topk=3
)

model_name = training_cfg.model_name_or_path

data_handler = ActiveLearningDatasetHandler(train_dataset)
random_al = RandomActiveLearner(
    data_handler,
    model_name,
    dataset_cfg,
    active_learning_cfg,
    train_args,
    False,
    val_dataset,
    eval_dataset
    )
random_al.run()