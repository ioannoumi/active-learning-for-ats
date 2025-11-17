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

DATASETS = {
    'billsum': ('FiscalNote/billsum', 'text', 'summary'),
    'aeslc': ('Yale-LILY/aeslc', 'email_body','subject_line')
}

LEARNES = {
    'random':'Random',
    'loss':'Loss',
    'idds':'IDDS',
    'bas':'BAS',
    'dual':'DUAL'
}

HIGHLIGHT_COLOR = "red"
BASE_COLOR = "steelblue"
model = SentenceTransformer("all-MiniLM-L6-v2")

dataset_source, dataset_text_col, dataset_target_col = DATASETS['aeslc']
dataset_name = dataset_source.split('/')[-1]
learner = LEARNES['loss']
learner_lower = learner.lower()

dataset_cfg = DatasetConfig(
    source=dataset_source,
    text_col=dataset_text_col,
    target_col=dataset_target_col,
    train_split='train',
    val_split='validation',
    test_split='test',
    max_source_length=512,
    max_target_length=32,
)

train_dataset, val_dataset, eval_dataset = retrieve_dataset(dataset_cfg)

all_embeddings = get_embeddings(train_dataset, dataset_text_col, model)

path_idxs = PROJECT_ROOT / "experiments" / "Yale-LILY" / dataset_name / 'bart-base' /  "seed_42" / learner_lower / "selected_idxs.json"

with path_idxs.open() as f:
    selected_groups = [json.loads(line) for line in f if line.strip()]

IDXS_SELECTED = [idx for group in selected_groups for idx in group][0:120]

xy = PCA(n_components=2).fit_transform(all_embeddings)

highlight_mask = np.zeros(len(xy), dtype=bool)
valid_highlights = [idx for idx in IDXS_SELECTED]

if valid_highlights:
    highlight_mask[valid_highlights] = True


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

if valid_highlights:
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

plt.legend()
plt.title(f"{learner} on {dataset_name}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle='--', alpha=0.3)


output_path = PROJECT_ROOT / "plots" /  dataset_name/ 'embeddings'/f"{learner}-plot_embeddings_new.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
plt.show()

