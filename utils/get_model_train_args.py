BASE_ARGS = {
    'has_validation': False,
    'val_size': 0,
    'warmup_ratio': 0.1,
    'gradient_accumulation_steps': 1
}

TRAIN_ARGS= {
    'flan-t5-large': {
    **BASE_ARGS,
    'per_device_train_batch_size': 6,
    'per_device_eval_batch_size': 16,
    'learning_rate': 3e-5,
    'num_train_epochs': 3,
    'optim': "adafactor",
    'generation_num_beams': 3,
    'weight_decay': 0.01,
    'min_train_steps': None
    },
    'bart-base': { 
    **BASE_ARGS,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 25,
    'learning_rate': 2e-5,
    'num_train_epochs': 6,
    'optim': "adamw_torch",
    'generation_num_beams': 4,
    'weight_decay': 0.028,
    'min_train_steps': 350
    },
    'pegasus-large': {
    **BASE_ARGS,   
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 16,
    'learning_rate': 5e-4,
    'num_train_epochs': 4,
    'optim': "adamw_torch",
    'generation_num_beams': 4,
    'weight_decay': 0.03,
    'min_train_steps': 200
    },
    't5-small': {
    **BASE_ARGS,
    'per_device_train_batch_size': 6,
    'per_device_eval_batch_size': 16,
    'learning_rate': 3e-5,
    'num_train_epochs': 3,
    'optim': "adafactor",
    'generation_num_beams': 3,
    'weight_decay': 0.01,
    'min_train_steps': None
    }
}

def get_model_train_args(model_name):
    return TRAIN_ARGS[model_name]   