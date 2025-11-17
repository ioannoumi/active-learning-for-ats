from datasets import Dataset
from sentence_transformers import SentenceTransformer
import torch
import logging

def get_embeddings(dataset: Dataset, key_column: str, model: SentenceTransformer, batch_size = 64):
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  
  model.to(device)
  logging.info(f"SentenceTransformer model moved to {device}.")
  
  embeddings = model.encode(
    dataset[key_column],
    batch_size=batch_size,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
  )
  logging.info(f"Embeddings have been generated on device: {embeddings.device}")
  return embeddings