# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Vendi feature helpers dispatch to the existing embedding and kernel contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

pytestmark = pytest.mark.unit
PACKAGE = Path(__file__).resolve().parents[3] / "worldfoundry/evaluation/tasks/metrics/vendi_score/vendi_score"


def _load_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_test_vendi_{name}", PACKAGE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def optional_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, fields in {
        "datasets": {},
        "nltk.tokenize": {"word_tokenize": Mock()},
        "nltk.translate": {"bleu_score": Mock()},
        "sklearn": {"preprocessing": SimpleNamespace(normalize=Mock(side_effect=lambda x, axis: x))},
        "sklearn.feature_extraction.text": {"CountVectorizer": Mock()},
        "sklearn.preprocessing": {"normalize": Mock()},
        "transformers": {"AutoModel": Mock(), "AutoTokenizer": Mock()},
        "vendi_score": {"data_utils": Mock(), "vendi": Mock()},
        "vendi_score.data_utils": {"Example": object},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)


def test_add_embeddings_forwards_model_path_and_preserves_examples(
    optional_providers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = _load_module("text_utils")
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    embed = create_autospec(text.get_embeddings, return_value=embeddings)
    monkeypatch.setattr(text, "get_embeddings", embed)
    examples = [SimpleNamespace(x=value, features={"existing": value}) for value in ("first", "second")]
    model, tokenizer = object(), object()

    result = text.add_embeddings_to_examples(
        examples,
        model=model,
        tokenizer=tokenizer,
        model_name="chosen-model",
        batch_size=2,
        device="cpu",
        feature_name="custom",
    )

    assert result is examples
    embed.assert_called_once_with(
        ["first", "second"], model=model, tokenizer=tokenizer, model_path="chosen-model", batch_size=2, device="cpu"
    )
    for example, embedding in zip(examples, embeddings, strict=False):
        assert example.features["existing"] == example.x
        np.testing.assert_array_equal(example.features["custom"], embedding)


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_intdiv_x_uses_kernel_formula(optional_providers: None, q: int, weighted: bool, normalize: bool) -> None:
    vendi = _load_module("vendi")
    vectors = np.array([[1.0, 0.0], [0.6, 0.8]])
    probabilities = np.array([0.25, 0.75]) if weighted else None
    weights = probabilities if probabilities is not None else np.array([0.5, 0.5])
    expected = 1 - np.sum((vectors @ vectors.T) ** q * np.outer(weights, weights))

    result = vendi.intdiv_X(vectors, q=q, p=probabilities, normalize=normalize)

    assert result == expected
    np.testing.assert_array_equal(vectors, [[1.0, 0.0], [0.6, 0.8]])
    provider: Any = vendi.preprocessing.normalize
    if normalize:
        provider.assert_called_once_with(vectors, axis=1)
    else:
        provider.assert_not_called()
