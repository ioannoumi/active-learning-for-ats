import os
# This must be set before torch is imported.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from active_learning.active_learning_dataset_handler import ActiveLearningDatasetHandler
from active_learning.random_active_learner import RandomActiveLearner
from active_learning.loss_active_learner import LossActiveLearner
from active_learning.idds_active_learner import IDDSActiveLearner
from active_learning.bas_active_learner import BASActiveLearner 
from active_learning.dual_active_learner import DUALActiveLearner
from active_learning.loss_with_random_active_learner import LossWithRandomActiveLearner   
from active_learning.loss_with_random_active_learner_var1 import LossWithRandomActiveLearnerVAR1
from active_learning.loss_with_idds_active_learner import LossWithIDDSActiveLearner
from active_learning.loss_round_robin_active_learner import LossRoundRobinActiveLearner
from configs.dataset_config import DatasetConfig
from data.retrieve_dataset import retrieve_dataset
from configs.training_config import TrainingConfig
from utils.get_args import get_train_args
import torch
import logging  
import nltk
from configs.active_learning_config import ActiveLearningConfig
from utils.set_seed import set_seed
from utils.get_model_train_args import get_model_train_args
from utils.logging_utils import setup_logging_to_file

MODELS = {
"bart_base": "facebook/bart-base"
# "pegasus_large": "google/pegasus-large"
} 

DATASETS = {
'billsum': ('FiscalNote/billsum', 'text', 'summary'),
'aeslc': ('Yale-LILY/aeslc', 'email_body','subject_line'),
'wikihow': ('wikihow_clean_LLM_ready.parquet','text','summary'),
'xsum': ('EdinburghNLP/xsum','document','summary')
}

LEARNERS = {
# 'loss': LossActiveLearner,
# 'bas': BASActiveLearner,
# 'loss_with_random': LossWithRandomActiveLearner
# 'loss_with_random_var2': LossWithRandomActiveLearnerVAR2
# 'dual': DUALActiveLearner,
# 'random': RandomActiveLearner
# 'idds': IDDSActiveLearner
'loss_round_robin':LossRoundRobinActiveLearner
}

def main(learner_strategy: str, dataset_key: str, model_key: str):
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_name = MODELS[model_key]
    TRAIN_ARGS = get_model_train_args(model_name.split('/')[-1])

    dataset_source, dataset_text_col, dataset_target_col = DATASETS[dataset_key]
    dataset_cfg = DatasetConfig(
        source=dataset_source,
        text_col=dataset_text_col,
        target_col=dataset_target_col,
        train_split='train',
        val_split='validation',
        test_split='test',
        max_source_length=512,
        max_target_length=32 if dataset_key == 'aeslc' else 128,
        val_size=None if dataset_key == 'aeslc' else 1000,
        test_size=None if dataset_key == 'aeslc' else 1000
    )
    print(dataset_cfg)
    active_learning_cfg = ActiveLearningConfig(
        strategy= learner_strategy,
        iterations=15,
        samples_per_iteration=10,
        warmup_samples=10,
        hard_examples_topk=5,
        bas_num_samples_to_rank=10  if learner_strategy == 'dual' else 100,
        bas_num_samples_mc_dropout=10,    
    )

    #17,23,42,1, 123
    seeds = [42,17]
    for seed in seeds:
        print(f"Running experiment with seed {seed}")
        if seed == 42 and (learner_strategy == 'idds' or learner_strategy == 'random'):
            continue
        is_parquet = dataset_source.endswith('.parquet')
        train_dataset, val_dataset, eval_dataset = retrieve_dataset(dataset_cfg, is_parquet=is_parquet)
        set_seed(seed)

        output_dir_path = f"./experiments/{dataset_cfg.source}/{model_name if '/' not in model_name else model_name.split('/')[-1]}/seed_{seed}/{active_learning_cfg.strategy}"
        # output_dir_path = f"./experiments/{dataset_cfg.source}/{model_name if '/' not in model_name else model_name.split('/')[-1]}/full_dataset_train"
        training_cfg = TrainingConfig(
            seed=seed,
            output_dir= output_dir_path,
            evaluation_strategy='epoch' if TRAIN_ARGS['has_validation'] else 'no',
            save_strategy='epoch' if TRAIN_ARGS['has_validation'] else 'no',
            **TRAIN_ARGS,
        )
        print(training_cfg)
        train_args = get_train_args(training_cfg)
        print(f' BF16 IS: {train_args.bf16}')
     
        setup_logging_to_file(training_cfg.output_dir)
        logging.info(f"Running experiment with seed {seed}")

        data_handler = ActiveLearningDatasetHandler(train_dataset, seed=seed)

        learner_cls = LEARNERS.get(active_learning_cfg.strategy)
        if learner_cls is None:
            raise ValueError(f"Unknown active learning strategy: {active_learning_cfg.strategy}")

        learner = learner_cls(
            data_handler,
            model_name,
            dataset_cfg,
            active_learning_cfg,
            train_args,
            TRAIN_ARGS['has_validation'],
            TRAIN_ARGS['val_size'],
            TRAIN_ARGS['min_train_steps'],
            val_dataset,
            eval_dataset,
        )
        learner.run()

        del learner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    dataset_key = 'aeslc'

    for model_key in MODELS:
        nltk.download('punkt_tab')
        for learner_strategy in LEARNERS:
            print(f"\n{'='*20} Running Experiment: Learner='{learner_strategy}', Model='{model_key}', Dataset='{dataset_key}' {'='*20}\n")
            main(learner_strategy, dataset_key, model_key)
