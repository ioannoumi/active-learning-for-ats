import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from embeddings.get_embeddings import get_embeddings
from data.retrieve_dataset import retrieve_dataset
from configs.dataset_config import DatasetConfig
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

MODELS = {
# "flan-t5_large": "google/flan-t5-large",
"bart-base": "facebook/bart-base"
# "pegasus-large": "google/pegasus-large"
} 

DATASETS = {
    'billsum': ('FiscalNote/billsum', 'text', 'summary'),
    'aeslc': ('Yale-LILY/aeslc', 'email_body','subject_line'),
    'xsum': ('EdinburghNLP/xsum','document','summary')
}

LEARNES = {
    # 'random':'Random',
    # 'loss':'Loss'
    # 'idds':'IDDS'
    # 'bas':'BAS',
    # 'dual':'DUAL'
    # 'loss_with_random':'Loss_Random',
    # 'loss_with_random_var2':'Loss_IDDS',
    'loss_round_robin':'Loss_RR'
}

HIGHLIGHT_COLOR = "red"
BASE_COLOR = "steelblue"
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_hard_examples_idxs(path_hard_examples: str):
    hard_examples_idxs = set()
    try:
        with open(path_hard_examples, 'r') as f:
            lines = [json.loads(line) for line in f if line.strip()]

        for line in lines:
            for hard_example_idx in line:
                hard_examples_idxs.add(hard_example_idx)
        return hard_examples_idxs
    except FileNotFoundError:
        print(f"Error: File not found at {path_hard_examples}")
        return set()

def plot_embeddings(dataset_source: str,learner_strategy_key: str, model_key: str, all_embeddings: np.ndarray, seed: int):
    dataset_name = dataset_source.split('/')[-1]
    learner = LEARNES[learner_strategy_key]
    hard_examples_idxs = set()

    # if 'loss' in learner_strategy_key:
    #     hard_examples_idxs = get_hard_examples_idxs(PROJECT_ROOT / "experiments" / "EdinburghNLP" / dataset_name / model_key /  f"seed_{seed}" / learner_strategy_key / "top5_example_losses.json")
    #     hard_examples_idxs = list(hard_examples_idxs)
    #     hard_examples_idxs = [int(idx) for idx in hard_examples_idxs]

    path_idxs = PROJECT_ROOT / "experiments" / "EdinburghNLP" / dataset_name / model_key /  f"seed_{seed}" / learner_strategy_key / "selected_idxs.json"

    if not path_idxs.exists():
        print(f"WARNING: File not found, skipping plot: {path_idxs}")
        return

    print(f"Loading selected indices from: {path_idxs}")
    with path_idxs.open() as f:
        selected_groups = [json.loads(line) for line in f if line.strip()]

    IDXS_SELECTED = [idx for group in selected_groups for idx in group]

    print("Performing PCA on embeddings...")
    xy = PCA(n_components=2).fit_transform(all_embeddings)

    highlight_mask = np.zeros(len(xy), dtype=bool)
    if IDXS_SELECTED:
        highlight_mask[IDXS_SELECTED] = True


    plt.figure(figsize=(10, 8))
    base_points = xy[~highlight_mask]
    plt.scatter(
        base_points[:, 0],
        base_points[:, 1],
        s=20,
        alpha=0.6,
        color=BASE_COLOR,
        label="All Samples",
    )

    if IDXS_SELECTED:
        highlight_points = xy[highlight_mask]
        plt.scatter(
            highlight_points[:, 0],
            highlight_points[:, 1],
            s=60,
            alpha=1.0,
            color=HIGHLIGHT_COLOR,
            label="Samples selected for annotation",
            edgecolors="black", 
            linewidths=1
        )

    if hard_examples_idxs:
        highlight_mask = np.zeros(len(xy), dtype=bool)
        highlight_mask[hard_examples_idxs] = True
        hard_points = xy[highlight_mask]
        plt.scatter(
            hard_points[:, 0],
            hard_points[:, 1],
            s=60,
            alpha=1.0,
            color='green',
            label="Hard samples",
            edgecolors="black", 
            linewidths=1
        )

    plt.legend()
    plt.title(f"{learner} on {dataset_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)


    output_path = PROJECT_ROOT / "plots" /  dataset_name/ 'embeddings'/model_key/f"{learner}/seed_{seed}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":

    seeds = [42,17,23]

    dataset_source, dataset_text_col, dataset_target_col = DATASETS['xsum']
    is_parquet = dataset_source.endswith('.parquet')
    safe_dataset_source = dataset_source.replace('/', '_')
    embeddings_path = f'embeddings/cached_embeddings_{safe_dataset_source}.pt'

    dataset_cfg = DatasetConfig(
        source=dataset_source,
        text_col=dataset_text_col,
        target_col=dataset_target_col,
        train_split='train',
        val_split='validation',
        test_split='test',
        max_source_length=512,
        max_target_length=128
    )
    print("Retrieving dataset...")
    train_dataset, _, _ = retrieve_dataset(dataset_cfg,is_parquet=is_parquet)

    print("Getting embeddings (this may take a while on the first run)...")
    embeddings_np = get_embeddings(train_dataset, dataset_text_col, model,embeddings_path=embeddings_path).detach().cpu().numpy()

    for model_key in MODELS:
        for learner_strategy_key in LEARNES:
            for seed in seeds:
                print(f"--- Generating plot for Model: '{model_key}', Learner: '{learner_strategy_key}', Seed: {seed}---")
                plot_embeddings(dataset_source,learner_strategy_key, model_key, embeddings_np,seed = seed)