from configs.training_config import TrainingConfig
from transformers import Seq2SeqTrainingArguments

def get_train_args(train_cfg: TrainingConfig) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        # ==== Paths ====
        output_dir=train_cfg.output_dir,
        overwrite_output_dir=train_cfg.overwrite_output_dir,

        # ==== Training ====
        seed=train_cfg.seed,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.num_train_epochs,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,

        # ==== Generation ====
        predict_with_generate=train_cfg.predict_with_generate,
        generation_num_beams=train_cfg.generation_num_beams,
        generation_max_length=train_cfg.generation_max_length,

        # ==== Evaluation and Checkpoints ====
        eval_strategy=train_cfg.evaluation_strategy,
        save_strategy=train_cfg.save_strategy,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        save_total_limit=train_cfg.save_total_limit,

        # ==== Optimization ====
        optim=train_cfg.optim,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,

        # ==== Logging ====
        logging_strategy=train_cfg.logging_strategy,
        logging_steps=train_cfg.logging_steps,
        report_to=train_cfg.report_to,
        bf16=train_cfg.bf16
    )