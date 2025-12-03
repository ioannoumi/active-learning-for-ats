from datasets import Dataset
from sentence_transformers import SentenceTransformer
import torch
import os
import logging

def get_embeddings(dataset: Dataset, key_column: str, model: SentenceTransformer, embeddings_path: str, batch_size = 256) -> torch.Tensor:
   
  if os.path.exists(embeddings_path):
    logging.info(f"Embeddings loaded from {embeddings_path}")
    print(f"Embeddings loaded from {embeddings_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = torch.load(embeddings_path, map_location=device)
    return embeddings
   
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
  torch.save(embeddings, embeddings_path)
  logging.info(f"Embeddings saved to {embeddings_path}")
  print(f"Embeddings saved to {embeddings_path}")
  return embeddings