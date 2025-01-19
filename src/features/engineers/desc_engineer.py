"""Text feature engineer for loan descriptions."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import scan_cache_dir
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


class DescriptionFeatureEngineer:
    """Text feature engineer for loan descriptions."""

    def __init__(
        self,
        n_embedding_features: int = 5,
        n_tfidf_features: int = 10,
        device: str = "cpu",
    ):
        """Initialize the engineer."""
        self.n_embedding_features = n_embedding_features
        self.n_tfidf_features = n_tfidf_features
        self.model_name = "huawei-noah/TinyBERT_General_4L_312D"
        self.tokenizer = None
        self.model = None
        self.tfidf = None
        self.feature_names = None
        self.device = torch.device(device)

    # Function to check if the model exists in the cache

    def check_model_in_cache(self):
        try:
            # Scan the cache directory for models
            cache_info = scan_cache_dir()
            repo_id = self.model_name.replace(
                "/", "--"
            )  # Convert repo name to cache directory format
            model_dir = None

            # Search for the specific model in the cache
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    model_dir = Path(repo.repo_dir)
                    break

            if model_dir and model_dir.exists():
                print(f"{self.model_name} is already downloaded in cache: {model_dir}")
                return model_dir
            else:
                print(f"{self.model_name} is not downloaded yet.")
                return None

        except Exception as e:
            print(f"Error checking cache: {e}")
            return None

    def fit(self, descriptions: pd.Series) -> "DescriptionFeatureEngineer":
        """Fit the engineer on training descriptions."""
        cache_dir = self.check_model_in_cache()
        if not cache_dir:
            print(f"Downloading {self.model_name}...")
            # Download TinyBERT
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            print(f"{self.model_name} downloaded.")
        else:
            print(f"{self.model_name} already downloaded.")
            # Initialize TinyBERT
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        # Fit TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=self.n_tfidf_features, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf.fit(descriptions.fillna("[NO_DESCRIPTION]"))
        self.feature_names = self.tfidf.get_feature_names_out()

        return self

    def transform(self, descriptions: pd.Series) -> pd.DataFrame:
        """Transform descriptions into features."""
        if self.tokenizer is None or self.model is None or self.tfidf is None:
            raise ValueError("Transformer must be fitted before transform")

        descriptions = descriptions.fillna("[NO_DESCRIPTION]")

        # Get embeddings and TF-IDF features
        embeddings = self._get_embeddings(descriptions)
        tfidf_matrix = self.tfidf.transform(descriptions)

        # Create output DataFrame
        df_features = pd.DataFrame()

        # Add embedding features
        for i in range(self.n_embedding_features):
            df_features[f"desc_emb_{i}"] = embeddings[:, i]

        # Add TF-IDF features
        for i, feature in enumerate(self.feature_names):
            df_features[f"desc_keyword_{feature}"] = (
                tfidf_matrix[:, i].toarray().flatten()
            )

        # Add basic text features
        df_features["desc_length"] = descriptions.str.len()
        df_features["desc_word_count"] = descriptions.str.split().str.len()

        return df_features

    def _get_embeddings(self, descriptions: pd.Series) -> np.ndarray:
        """Get TinyBERT embeddings for all descriptions."""
        batch_size = 64
        embeddings = []

        for i in range(0, len(descriptions), batch_size):
            batch_texts = descriptions.iloc[i : i + batch_size].tolist()
            batch_embeddings = [self._get_embedding(text) for text in batch_texts]
            embeddings.extend(batch_embeddings)

        return np.vstack(embeddings)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get TinyBERT embedding for a single text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        # Move inputs to correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
