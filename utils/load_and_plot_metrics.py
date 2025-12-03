import json
from pathlib import Path
from typing import List
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(path: str, metric: str):
    if not os.path.exists(path):
        return None
    
    records = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return [record.get(metric) for record in records if record.get(metric) is not None]

def plot_metrics(project_root: str,
                 dataset_name: str,
                 model_name : str,
                 scores_by_learner: dict,
                   metric: str = 'rouge1'
                ):

    fig,ax = plt.subplots(figsize=(8, 4.5))
    for learner_name, scores in scores_by_learner.items():
        if scores:
            x_axis = range(1, len(scores) + 1)
            ax.plot(x_axis, scores, marker="o", label=learner_name)

    ax.set_title(f"{model_name} on {dataset_name}")
    ax.set_xlabel("AL Iterations")
    ax.set_ylabel(f"Rouge-1")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.grid(True,alpha=0.4)
    ax.legend()
    output_path = project_root / "plots" / dataset_name / metric / f"{model_name}-rouge1_mean_scores.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    datasets = {
        'aeslc':'aeslc',
        'billsum':'billsum',
        'xsum':'xsum'
    }
    models = {
        # 'flan-t5-large': 'flan-t5-large',
        'bart-base': 'bart-base'
        # 'pegasus-large': 'pegasus-large'
    }
    learners = {
        # 'Random': 'random',
        # 'Loss': 'loss',
        # 'IDDS': 'idds',
        # 'BAS': 'bas',
        'DUAL': 'dual',
        # 'Loss_Random': 'loss_with_random',
        # 'Loss_IDDS': 'loss_with_random_var2',
        'Loss_RR': 'loss_round_robin'
    }
    
    seeds = [42, 17, 23] # Using the seeds from your main.py context
    dataset_name = datasets['xsum']
    
    for model_key in models:
        model_name = models[model_key] 
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        
        all_scores_by_learner_and_seed = {display_name: [] for display_name in learners.keys()}

        for seed in seeds:
            print(f"Loading metrics for model: {model_name}, seed: {seed}")
            for display_name, learner_key in learners.items():
                path = PROJECT_ROOT / "experiments" / "EdinburghNLP" / dataset_name / model_key / f'seed_{seed}'/ learner_key / "eval_metrics.json"
                scores = load_metrics(str(path), 'eval_rouge1')
                if scores:
                    all_scores_by_learner_and_seed[display_name].append(scores)
                else:
                    print(f"WARNING: No metrics found for {display_name} with seed {seed} at {path}")
        
        mean_scores_by_learner = {}
        for display_name, learner_key in learners.items():
            list_of_scores_for_seeds = all_scores_by_learner_and_seed[display_name]
            if list_of_scores_for_seeds:
                
                min_len = min(len(s) for s in list_of_scores_for_seeds)
                trimmed_scores = [s[:min_len] for s in list_of_scores_for_seeds]
                
                mean_scores = np.mean(trimmed_scores, axis=0).tolist()
                mean_scores_by_learner[display_name] = mean_scores
            else:
                print(f"INFO: No scores available for {display_name} across any seed, skipping mean calculation.")

      
        plot_metrics(PROJECT_ROOT, dataset_name, model_key, mean_scores_by_learner, metric='rouge1')
