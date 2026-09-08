import numpy as np
from face_id_kit.clustering import cluster_embeddings


def test_clustering_groups_similar_samples_and_preserves_noise():
    labels = cluster_embeddings([np.array([1., 0., 0.]), np.array([1., .01, 0.]), np.array([0., 0., 1.])])
    assert labels[0] == labels[1] >= 0
    assert labels[2] == -1
