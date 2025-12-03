import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _get_mean_acquisition_time_per_iteration(dataset_source: str, model_key: str, learner_key: str, seed: int):
    path_seed = PROJECT_ROOT / "experiments" / dataset_source / model_key / f'seed_{seed}' / learner_key / 'acquisition_times.json'
    if not os.path.exists(path_seed):
        print(f"WARNING: File not found, skipping: {path_seed}")
        return None

    try:
        with open(path_seed, 'r') as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"ERROR: Could not read or parse {path_seed}: {e}")
        return None

    acquisition_times = []
    precomputation_time = 0
    for record in lines:
        if 'idds_precomputation_time_seconds' in record:
            precomputation_time = record['idds_precomputation_time_seconds']
        if 'acquisition_time_seconds' in record:
            acquisition_times.append(record['acquisition_time_seconds'])

    if not acquisition_times:
        return 0
    
    amortized_precomputation = precomputation_time / len(acquisition_times) if acquisition_times else 0
    
    mean_acquisition_time = np.mean([t + amortized_precomputation for t in acquisition_times])
    return mean_acquisition_time

def _plot_acquisition_times(learners_mean_run_time: dict[str, float], model_key: str, dataset_name: str) -> None:
    sorted_items = sorted(learners_mean_run_time.items(), key=lambda item: item[1])
    
    learner_names = [item[0] for item in sorted_items]
    mean_times = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(learner_names, mean_times, color='skyblue')
    ax.set_ylabel('Average Acquisition Time per Iteration (seconds)')
    ax.set_title(f'Mean Acquisition Time per AL Iteration\nModel: {model_key}, Dataset: {dataset_name}')
    ax.set_xticklabels(learner_names, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center')
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / "plots" / dataset_name  / "mean_acquisition_times" / f"{model_key}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

MODELS = {
    'bart-base': 'bart-base',
    'pegasus-large': 'pegasus-large'
}

LEARNERS = {
    'Loss': 'loss',
    'IDDS': 'idds',
    'BAS': 'bas',
    'DUAL': 'dual',
    'Loss_Random': 'loss_with_random'
}

SEEDS = [42, 17, 23]

dataset_source = 'EdinburghNLP/xsum'
dataset_name = dataset_source.split('/')[-1]

for model_key in MODELS:
    print(f"\n--- Processing Model: {model_key} ---")
    learners_mean_times = {}
    for learner_strategy in LEARNERS:
        learner_key = LEARNERS[learner_strategy]
        seed_times = []
        for seed in SEEDS:
            mean_acquisition_time = _get_mean_acquisition_time_per_iteration(dataset_source, model_key, learner_key, seed)
            if mean_acquisition_time is not None:
                seed_times.append(mean_acquisition_time)

        if seed_times:
            learners_mean_times[learner_strategy] = np.mean(seed_times)
        else:
            print(f"INFO: No data found for learner '{learner_strategy}' with model '{model_key}'.")

    _plot_acquisition_times(learners_mean_times, model_key, dataset_name)