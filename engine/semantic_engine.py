"""Sunwa-AI — Multilingual Semantic Embedding Engine.
Uses: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Falls back to TF-IDF (scikit-learn) if transformer unavailable.
Lazy-loaded and cached to avoid model reloads on Streamlit reruns.
"""
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

_MODEL = None
_IS_FALLBACK = False


def _load_model(force_fallback: bool = False):
    global _MODEL, _IS_FALLBACK
    if _MODEL is not None:
        return _MODEL, _IS_FALLBACK
    if force_fallback:
        _MODEL, _IS_FALLBACK = _make_fallback_model(), True
        return _MODEL, _IS_FALLBACK
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading multilingual transformer model...")
        _MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        _IS_FALLBACK = False
        logger.info("Transformer model loaded.")
    except Exception as exc:
        logger.warning("Transformer unavailable (%s), using TF-IDF fallback.", exc)
        _MODEL = _make_fallback_model()
        _IS_FALLBACK = True
    return _MODEL, _IS_FALLBACK


class _FallbackTfidf:
    """CPU TF-IDF n-gram vectorizer as embedding fallback."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        self.vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=8000,
            sublinear_tf=True,
        )
        self._fitted = False
        self.is_fallback = True

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if not self._fitted:
            self.vec.fit(texts)
            self._fitted = True
        mat = self.vec.transform(texts).toarray()
        # Normalize rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def _make_fallback_model() -> _FallbackTfidf:
    return _FallbackTfidf()


class SemanticEngine:
    """Encapsulates embedding model loading and cosine similarity computation."""

    def __init__(self, force_fallback: bool = False) -> None:
        self._model, self.is_fallback = _load_model(force_fallback)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        if self.is_fallback:
            return self._model.encode(texts)
        # Transformer model
        return self._model.encode(texts, show_progress_bar=False, batch_size=32)

    def cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms
        return normed @ normed.T

    def compute_similarities(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embed(texts)
        return self.cosine_similarity_matrix(embeddings)
