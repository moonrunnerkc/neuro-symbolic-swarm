# Author: Bradley R. Kinnard
"""Tests for the embedder module."""

import numpy as np
import pytest

from src.embedder import cosine_similarity


class TestCosineSimilarity:
    """tests for the cosine similarity helper (no model needed)."""

    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_normalized_vectors(self):
        a = np.array([0.6, 0.8], dtype=np.float32)
        b = np.array([0.8, 0.6], dtype=np.float32)
        expected = float(np.dot(a, b))
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-5)

    def test_high_dimensional(self):
        rng = np.random.default_rng(42)
        a = rng.random(384).astype(np.float32)
        b = rng.random(384).astype(np.float32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        result = cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0
