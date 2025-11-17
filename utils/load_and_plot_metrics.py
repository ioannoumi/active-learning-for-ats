import json
from pathlib import Path
from typing import List
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def load_metrics(path: str, metric: str):
    raw = Path(path).read_text()
    records = json.loads('[' + raw.replace('}\n{', '},\n{') + ']')
    rouge1_scores = [record[metric] for record in records]
    return rouge1_scores

def plot_metrics(project_root: str,
                 dataset_name: str,
                 model_name : str,
                 random_scores: List[float] = None,
                 loss_scores:List[float] = None,
                idds_scores:List[float] = None,
                   bas_scores:List[float] = None, 
                   dual_scores:List[float] = None,
                   loss_with_random: List[float] = None,
                   metric: str = 'rouge1'
                ):
    x1 = range(1, len(random_scores) + 1) if random_scores is not None else None
    x2 = range(1, len(loss_scores) + 1) if loss_scores is not None else None
    x3 = range(1, len(idds_scores) + 1) if idds_scores is not None else None
    x4 = range(1, len(bas_scores) + 1) if bas_scores is not None else None
    x5 = range(1, len(dual_scores) + 1) if dual_scores is not None else None
    x6 = range(1, len(loss_with_random) + 1) if loss_with_random is not None else None

    fig,ax = plt.subplots(figsize=(8, 4.5))
    if random_scores is not None:
        ax.plot(x1, random_scores, marker="o", label="Random")
    if loss_scores is not None: 
        ax.plot(x2, loss_scores, marker="o", label="Loss")
    if idds_scores is not None:
        ax.plot(x3, idds_scores, marker="o", label="IDDS")
    if bas_scores is not None:
        ax.plot(x4, bas_scores, marker="o", label="BAS")
    if dual_scores is not None:
        ax.plot(x5, dual_scores, marker="o", label="DUAL")
    if loss_with_random is not None:
        ax.plot(x6, loss_with_random,marker="o", label="Loss_Random")

    ax.set_title(f"{model_name} on {dataset_name}")
    ax.set_xlabel("AL Iterations")
    ax.set_ylabel(f"Rouge-1")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.grid(True,alpha=0.4)
    ax.legend()
    output_path = project_root / "plots" / dataset_name / metric/f"{model_name}-rouge1_scores.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    datasets = {
        'aeslc':'aeslc',
        'billsum':'billsum'
    }
    models = {
        'flan-t5-large': 'flan-t5-large',
        'bart-base': 'bart-base',
        'pegasus-large': 'pegasus-large'
    }
    seeds = [42]
    seed = seeds[0]
    dataset_name = datasets['aeslc']
    model_name = models['bart-base']

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    random_path = PROJECT_ROOT / "experiments" / "Yale-LILY" / dataset_name / model_name / f'seed_{seed}'/"random" / "eval_metrics.json"
    loss_path = PROJECT_ROOT / "experiments" / "Yale-LILY" / dataset_name / model_name / f'seed_{seed}'/"loss" / "eval_metrics.json"
    idds_path = PROJECT_ROOT / "experiments" / "Yale-LILY" / dataset_name / model_name / f'seed_{seed}'/"idds" / "eval_metrics.json"
    bas_path = PROJECT_ROOT / "experiments" / "Yale-LILY" / dataset_name / model_name / f'seed_{seed}'/"bas" / "eval_metrics.json"
    dual_path = PROJECT_ROOT / "experiments" / "Yale-LILY" / dataset_name / model_name / f'seed_{seed}'/"dual" / "eval_metrics.json"
    loss_with_random_path = PROJECT_ROOT / "experiments" / "Yale-LILY" / dataset_name / model_name / f'seed_{seed}'/"loss_with_random" / "eval_metrics.json"

    random_scores = load_metrics(str(random_path), 'eval_rouge1')
    loss_scores = load_metrics(str(loss_path), 'eval_rouge1')
    idds_scores = load_metrics(str(idds_path),'eval_rouge1')
    bas_scores = load_metrics(str(bas_path),'eval_rouge1')
    dual_scores = load_metrics(str(dual_path),'eval_rouge1')
    loss_with_random_scores = load_metrics(str(loss_with_random_path),'eval_rouge1')

    plot_metrics(PROJECT_ROOT,dataset_name,model_name,loss_scores=loss_scores,loss_with_random=loss_with_random_scores,metric = 'rouge1')
