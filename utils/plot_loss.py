import json
import matplotlib.pyplot as plt
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_loss(dataset_key:str, model_key:str, seed:int, learner_strategy:str):
    losses_per_iteration = []
    path_loss = PROJECT_ROOT/'experiments'/dataset_key/model_key/f'seed_{seed}'/learner_strategy/'all_examples_losses.json'
    
    try:
        with open(path_loss, 'r') as f:
            lines = [json.loads(line) for line in f if line.strip()]
            
        for line in lines:
            iteration_losses = list(line.values())
            losses_per_iteration.append(iteration_losses)
        return losses_per_iteration
        
    except FileNotFoundError:
        print(f"Error: File not found at {path_loss}")
        return []

def plot_loss(losses_per_iteration: list[list[float]], dataset_key: str, model_key: str, seed: int, learner_strategy: str) -> None:
    if not losses_per_iteration:
        print("No loss data to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(losses_per_iteration)
    
    plt.title('Distribution of Top Example Losses per AL Iteration')
    plt.xlabel('AL Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = PROJECT_ROOT/'plots'/dataset_key/f'loss_plot_{model_key}_seed_{seed}_{learner_strategy}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

dataset_key = 'EdinburghNLP/xsum'
model_key = 'bart-base'
seed = 17
learner_strategy = 'loss_with_random'

losses_per_iteration = get_loss(dataset_key, model_key, seed, learner_strategy)
if losses_per_iteration:
    plot_loss(losses_per_iteration, dataset_key, model_key, seed, learner_strategy)