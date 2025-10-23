from active_learning.active_learning_dataset_handler import ActiveLearningDatasetHandler
from active_learning.random_active_learner import RandomActiveLearner
from active_learning.loss_active_learner import LossActiveLearner
from configs.dataset_config import DatasetConfig
from data.retrieve_dataset import retrieve_dataset
from configs.training_config import TrainingConfig
from utils.get_args import get_train_args
import torch
from configs.active_learning_config import ActiveLearningConfig
from utils.set_seed import set_seed

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

SEED = 42
set_seed(SEED)

MODELS = {
    "flan_t5_large": "google/flan-t5-large",
    "bart_base": "facebook/bart-base",
    "pegasus_large": "google/pegasus-large",
} 

DATASETS = {
    'billsum': ('FiscalNote/billsum', 'text', 'summary'),
    'aeslc': ('Yale-LILY/aeslc', 'email_body','subject_line')
}

LEARNERS = {
    'random': RandomActiveLearner,
    'loss': LossActiveLearner,
}

TRAIN_ARGS = {
    'per_device_train_batch_size': 6,
    'per_device_eval_batch_size': 16,
    'learning_rate': 3e-5,
    'num_train_epochs': 3,
    'gradient_accumulation_steps': 1,
    'optim': "adafactor",
    'generation_num_beams': 3,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'train_validation': False,
}

model_name = MODELS['bart_base']

dataset_source, dataset_text_col, dataset_target_col = DATASETS['aeslc']
dataset_cfg = DatasetConfig(
    source= dataset_source,
    text_col= dataset_text_col, target_col= dataset_target_col,
    train_split= 'train', val_split= 'validation',test_split= 'test',
    max_source_length=512,max_target_length=32
)

active_learning_cfg = ActiveLearningConfig(
    strategy= 'random',
    iterations= 15,
    samples_per_iteration=10,
    warmup_samples=10,
    hard_examples_topk=5
)

train_dataset,val_dataset,eval_dataset = retrieve_dataset(dataset_cfg)
training_cfg = TrainingConfig(
    seed= SEED,
    output_dir=f"./experiments/{dataset_cfg.source}/{model_name if '/' not in model_name else model_name.split('/')[-1]}/{active_learning_cfg.strategy}",
    evaluation_strategy= 'epoch' if TRAIN_ARGS['train_validation'] else 'no',
    save_strategy= 'epoch' if TRAIN_ARGS['train_validation'] else 'no',
    **TRAIN_ARGS
)
train_args = get_train_args(training_cfg)

data_handler = ActiveLearningDatasetHandler(train_dataset,seed=SEED)

learner_cls = LEARNERS.get(active_learning_cfg.strategy)
if learner_cls is None:
    raise ValueError(f"Unknown active learning strategy: {active_learning_cfg.strategy}")

learner = learner_cls(
    data_handler,
    model_name,
    dataset_cfg,
    active_learning_cfg,
    train_args,
    TRAIN_ARGS['train_validation'],
    val_dataset,
    eval_dataset
    )
learner.run()
