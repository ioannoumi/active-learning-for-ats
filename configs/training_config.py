from dataclasses import dataclass

@dataclass
class TrainingConfig():
    output_dir: str 
    overwrite_output_dir: bool = True

    # ==== Core Training ====
    seed: int = 42
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 16
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 1
    train_validation: bool = False

    # ==== Generation ====
    predict_with_generate: bool = True
    generation_num_beams: int = 3
    generation_max_length: int = 32

    # ==== Evaluation / Saving ====
    evaluation_strategy: str = "no"
    save_strategy: str = "no"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "rouge1"
    greater_is_better: bool = True
    save_total_limit: int = 1

    # ==== Optimization ====
    optim: str = "adafactor"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # ==== Logging ====
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: str = "none"