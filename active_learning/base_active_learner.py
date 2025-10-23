from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from active_learning.active_learning_dataset_handler import ActiveLearningDatasetHandler
from abc import ABC, abstractmethod
import evaluate
import json
import gc
import torch
import torch.utils
import numpy as np
from typing import List
from datasets import Dataset
from embeddings.get_embeddings import get_embeddings
from sentence_transformers import SentenceTransformer
from configs.dataset_config import DatasetConfig
from configs.active_learning_config import ActiveLearningConfig

class BaseActiveLearner(ABC):
    def __init__(
            self,
            data_handler: ActiveLearningDatasetHandler,
            model_name: str,
            dataset_cfg: DatasetConfig,
            active_learning_cfg: ActiveLearningConfig,
            train_args: Seq2SeqTrainingArguments,
            has_validation: bool,
            val_dataset: Dataset,
            eval_dataset: Dataset
        ):
        self.data_handler = data_handler
        self.model_name = model_name
        self.dataset_cfg = dataset_cfg
        self.active_learning_cfg = active_learning_cfg
        self.train_args = train_args

        self.tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.has_validation = has_validation

        if self.has_validation:
            self.tokenized_val_dataset = self._get_tokenized_dataset(val_dataset)
        self.tokenized_eval_dataset = self._get_tokenized_dataset(eval_dataset)
        self.tokenized_train_dataset = None

        self.rouge_metric = evaluate.load("rouge")
        self.experiment_metrics = []

        if not self.needs_embeddings():
            return
        
        emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = get_embeddings(self.data_handler.get_dataset(),dataset_cfg.text_col,emb_model)

    def _tokenize(self,batch):
        prompt = "summarize: " if 't5' in self.model_name else ''

        inputs = [prompt+m for m in batch[self.dataset_cfg.text_col]]

        model_inputs = self.tok(inputs, max_length=self.dataset_cfg.max_source_length, truncation=True)
        labels = self.tok(text_target=batch[self.dataset_cfg.target_col],max_length=self.dataset_cfg.max_target_length,truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def _get_tokenized_dataset(self,dataset: Dataset):
        return dataset.map(self._tokenize, batched=True, remove_columns=[column for column in dataset.column_names if column not in ['idxs']])

    def _compute_metrics(self,eval_pred):
        preds,labels = eval_pred
        # print("preds shape:", getattr(preds, "shape", None), "dtype:", getattr(preds, "dtype", None))
        # print("labels shape:",getattr(labels, "shape", None), "dtype:", getattr(labels, "dtype", None))

        # print(preds[0])
        # print(labels[0])

        preds = np.where(preds != -100, preds, self.tok.pad_token_id)
        decoded_preds = self.tok.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tok.pad_token_id)
        decoded_labels = self.tok.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        new_dict = {}
        for (k, v) in result.items():
            new_dict[k] = round(v*100, 2)

        # print('Reference:',decoded_labels[0])
        # print('Prediction:',decoded_preds[0])

        return new_dict

    def train_and_evaluate(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        data_collator = DataCollatorForSeq2Seq(self.tok, model=self.model)

        self.tokenized_train_dataset = self._get_tokenized_dataset(self.data_handler.get_labeled_data())

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset if self.has_validation else None,
            processing_class=self.tok,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate(
            eval_dataset = self.tokenized_eval_dataset,
            max_length = self.dataset_cfg.max_target_length,
            num_beams = self.train_args.generation_num_beams
        )

        self.experiment_metrics.append(metrics)
        self._write_data_to_file(metrics, '/eval_metrics.json')
        print("TEST ROUGE:", metrics)

        del trainer, data_collator
        gc.collect()
        torch.cuda.empty_cache()

    def _write_data_to_file(self,data,path):
        json_data = json.dumps(data)
        with open(self.train_args.output_dir + path, "a") as f:
            f.write(json_data + "\n")

    @abstractmethod
    def select_idxs(self) -> List[int]:
        """ To be implemented by subclasses """
        pass
    
    @abstractmethod
    def needs_embeddings(self) -> bool:
        pass

    @abstractmethod
    def needs_warmup(self) -> bool:
        pass

    def run(self):
        print(f"===ACTIVE LEARNING STRATEGY: {self.active_learning_cfg.strategy}===")
        if self.needs_warmup():
            print("=== WARM UP ===")
            selected_idxs = self.data_handler.sample_from_unlabelled(self.active_learning_cfg.warmup_samples)
            self.data_handler.update_labeled_idxs(selected_idxs)
            self.train_and_evaluate()
            self._write_data_to_file(selected_idxs, '/selected_idxs.json')

        for i in range(self.active_learning_cfg.iterations):
            print(f'=== ACTIVE LEARNING ITERATION: {i+1} ===')
            selected_idxs = self.select_idxs()
            self.data_handler.update_labeled_idxs(selected_idxs)
            self.train_and_evaluate()
            self._write_data_to_file(selected_idxs, '/selected_idxs.json')
            
    def _chosen_data(self, selected_idxs):
        chosen_data = self.data_handler.get_dataset().select(selected_idxs)
        return chosen_data[self.dataset_cfg.target_col]
