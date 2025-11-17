from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, PegasusTokenizer)
from active_learning.active_learning_dataset_handler import ActiveLearningDatasetHandler
from abc import ABC, abstractmethod
import evaluate
import json
import gc
import torch
import torch.utils
import numpy as np
import random
import logging
import math
import warnings
from typing import List
import time
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
            validation_size: int,
            min_train_steps: int,
            val_dataset: Dataset,
            eval_dataset: Dataset
        ):
        self.data_handler = data_handler
        self.model_name = model_name
        self.dataset_cfg = dataset_cfg
        self.active_learning_cfg = active_learning_cfg
        self.train_args = train_args
        self.has_validation = has_validation
        self.validation_size = validation_size
        self.min_train_steps = min_train_steps
        self.val_dataset = val_dataset
        self.eval_dataset = eval_dataset

        if 'pegasus' not in self.model_name:
            self.tok = AutoTokenizer.from_pretrained(self.model_name, use_fast= True)
        else:
            self.tok = PegasusTokenizer.from_pretrained(self.model_name)
        
        if self.has_validation:
            n_total = len(self.val_dataset)
            n = self.validation_size

            if n<=0:
                raise ValueError(f"train_validation_samples must be > 0, got {n}.")
            if n > n_total:
                warnings.warn(
                    f"Requested {n} validation samples; capping to dataset size {n_total}.",
                    RuntimeWarning,
                )
                n = n_total
            ds = self.val_dataset.shuffle(seed=self.train_args.seed)
            val_subset = ds.select(range(n))
            self.tokenized_val_dataset = self._get_tokenized_dataset(val_subset)

        self.tokenized_eval_dataset = self._get_tokenized_dataset(eval_dataset)
        self.tokenized_train_dataset = None

        self.rouge_metric = evaluate.load("rouge")
        self.experiment_metrics = []
        self.given_num_train_epochs = self.train_args.num_train_epochs

        if not self.needs_embeddings():
            return
        
        emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = get_embeddings(self.data_handler.get_dataset(),dataset_cfg.text_col,emb_model)
    
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

    def _tokenize(self,batch):
        prompt = "summarize: " if 't5' in self.model_name else ''

        inputs = [prompt+m for m in batch[self.dataset_cfg.text_col]]

        model_inputs = self.tok(inputs, max_length=self.dataset_cfg.max_source_length, truncation=True)
        labels = self.tok(text_target=batch[self.dataset_cfg.target_col],max_length=self.dataset_cfg.max_target_length,truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _get_tokenized_dataset(self,dataset: Dataset):
        return dataset.map(self._tokenize, batched=True, remove_columns=[column for column in dataset.column_names if column not in ['idxs']])

    def _calculate_num_train_epochs(self) -> int:
        if self.min_train_steps is None:
            return int(self.given_num_train_epochs)
        
        effective_batch_size = self.train_args.per_device_train_batch_size * self.train_args.gradient_accumulation_steps
        steps_per_epoch = math.ceil(self.data_handler.get_labeled_count() / effective_batch_size)
        min_epochs = math.ceil(self.min_train_steps / steps_per_epoch)
        return max(min_epochs, int(self.given_num_train_epochs))

    def _compute_metrics(self,eval_pred):
        preds,labels = eval_pred
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
        return new_dict
    
    def train_and_evaluate(self):
        self.train_args.num_train_epochs = self._calculate_num_train_epochs()

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
            compute_metrics=self._compute_metrics,
            callbacks= [EarlyStoppingCallback(early_stopping_patience=4)] if self.has_validation else None
        )

        trainer.train()
        metrics = trainer.evaluate(
            eval_dataset = self.tokenized_eval_dataset,
            max_length = self.dataset_cfg.max_target_length,
            num_beams = self.train_args.generation_num_beams
        )

        self.experiment_metrics.append(metrics)
        self._write_data_to_file(metrics, '/eval_metrics.json')
        # self.log_sample_generations(5)

        del trainer, data_collator
        gc.collect()
        torch.cuda.empty_cache()

    def run(self):
        logging.info(f"\n\n\nRUNNING ACTIVE LEARNING STRATEGY {self.active_learning_cfg.strategy}\n\n")
        if self.needs_warmup():
            logging.info("Strategy requires warmup, performing warmup...")
            _,selected_idxs = self.data_handler.sample_from_unlabeled(self.active_learning_cfg.warmup_samples)
            self.data_handler.update_labeled_idxs(selected_idxs)
            self.train_and_evaluate()
            self._write_data_to_file(selected_idxs, '/selected_idxs.json')
            self._log_examples_json(selected_idxs)

        for i in range(self.active_learning_cfg.iterations):
            logging.info(f"\n\n\nSTARTING ACTIVE LEARNING ITERATION {i+1}\n\n")
            
            start_time = time.time()
            selected_idxs = self.select_idxs()
            end_time = time.time()
            acquisition_time = end_time - start_time
            self._write_data_to_file({"iteration": i+1, "acquisition_time_seconds": acquisition_time}, '/acquisition_times.json')

            self.data_handler.update_labeled_idxs(selected_idxs)
            self.train_and_evaluate()
            self._write_data_to_file(selected_idxs, '/selected_idxs.json')
            self._log_examples_json(selected_idxs,i+1)

    def _write_data_to_file(self,data,path):
        json_data = json.dumps(data)
        with open(self.train_args.output_dir + path, "a") as f:
            f.write(json_data + "\n")

    def _log_examples_json(self, examples: list[int],iteration: int = None) -> None:
        documents = self.data_handler.get_dataset()
        json_lines_to_write = []

        logging.info('\n')
        for rank,idx in enumerate(examples,start=1):
            rec = documents[idx]
            json_data = json.dumps({
            "tag": "Selected Sample",
            "rank": rank,
            "id": idx,
            "doc_len": len(rec[self.dataset_cfg.text_col]),
            "summ_len": len(rec[self.dataset_cfg.target_col]),
            "doc_preview": str(rec[self.dataset_cfg.text_col]).replace("\n", " ")[:300],
            "summ_preview": str(rec[self.dataset_cfg.target_col]).replace("\n", " ")[:200],
            }, ensure_ascii=False)
            logging.info(json_data)
            json_lines_to_write.append(json_data + '\n')
            
        with open(self.train_args.output_dir + '/selected_samples.json', "a") as f:
            f.write(f'ACTIVE LEARNING ITERATION: {iteration}\n') if iteration is not None else f.write('WARMUP\n')
            f.writelines(json_lines_to_write)
            f.write('\n')
        logging.info('\n')
    
    def log_sample_generations(self, num_to_log: int) -> None:
        self.model.eval()

        random_idxs = random.sample(range(len(self.tokenized_eval_dataset)),num_to_log)
        subset = self.tokenized_eval_dataset.select(random_idxs)

        model_inputs = {
            "input_ids": subset["input_ids"],
            "attention_mask": subset["attention_mask"],
        }
        padded = self.tok.pad(model_inputs,padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(self.train_args.device)
        attention_mask = padded["attention_mask"].to(self.train_args.device)
        assert isinstance(input_ids, torch.Tensor)

        gens = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_length=self.dataset_cfg.max_target_length,
            num_beams=self.train_args.generation_num_beams,
            early_stopping = True,
            do_sample= False
        )

        decoded_gens = self.tok.batch_decode(gens, skip_special_tokens=True)
        decoded_golden_summaries = self.eval_dataset.select(random_idxs)[self.dataset_cfg.target_col]
        decoded_documents = self.eval_dataset.select(random_idxs)[self.dataset_cfg.text_col]

        logging.info("Logging sample generations: ")
        for i in range(num_to_log):
            logging.info(f"Original document: {decoded_documents[i]}")
            logging.info(f"Golden summary: {decoded_golden_summaries[i]})")
            logging.info(f"Generated summary: {decoded_gens[i]}")
            logging.info("") 