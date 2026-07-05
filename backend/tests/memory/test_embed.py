import numpy as np

from magenta.memory.embed import LocalEmbedder


def test_encode_shape_and_determinism():
    emb = LocalEmbedder()
    v1 = emb.encode(["my bill is too high"])
    v2 = emb.encode(["my bill is too high"])
    assert v1.shape == (1, 384)
    np.testing.assert_allclose(v1, v2, atol=1e-6)  # deterministic
    np.testing.assert_allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)  # normalized


def test_similarity_orders_correctly():
    emb = LocalEmbedder()
    near = emb.similarity("my bill went up", "the invoice is higher this month")
    far = emb.similarity("my bill went up", "the weather is nice today")
    assert near > far
