import numpy as np
from .matching import normalize_embedding


def cluster_embeddings(vectors, *, eps: float = .72, min_samples: int = 2):
    """Return integer cluster labels; -1 denotes individual noise samples."""
    if not len(vectors):
        return np.array([], dtype=int)
    from sklearn.cluster import DBSCAN
    matrix = np.stack([normalize_embedding(vector) for vector in vectors])
    return DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(matrix)
