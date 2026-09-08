import json
from pathlib import Path
import pytest

def test_android_clustering_fixture():
    pytest.importorskip("sklearn")
    from face_id_kit.clustering import cluster_embeddings
    data = json.loads((Path(__file__).parents[1] / "fixtures/clustering_parity.json").read_text())
    assert cluster_embeddings(data["vectors"]).tolist() == data["labels"]
