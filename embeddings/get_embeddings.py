from datasets import Dataset
from sentence_transformers import SentenceTransformer

def get_embeddings(dataset: Dataset, key_column: str, model: SentenceTransformer, batch_size = 64):
  embeddings = model.encode(
    dataset[key_column],
    batch_size=batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
  )
  return embeddings