# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Tanimoto kernels dispatch each molecule to the selected fingerprint callable."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def molecule_utils(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    rdkit = ModuleType("rdkit")
    chem = ModuleType("rdkit.Chem")
    all_chem = ModuleType("rdkit.Chem.AllChem")
    data_structs = ModuleType("rdkit.DataStructs")
    rd_base = ModuleType("rdkit.rdBase")
    vendi = ModuleType("vendi_score")
    data_utils = ModuleType("vendi_score.data_utils")

    def rdk_fingerprint(molecule: str) -> frozenset[str]:
        return frozenset(molecule)

    def morgan_fingerprint(molecule: str, radius: int, nBits: int) -> frozenset[str]:
        assert (radius, nBits) == (2, 1024)
        return frozenset(molecule)

    def similarity(left: frozenset[str], right: frozenset[str]) -> float:
        return len(left & right) / len(left | right)

    chem.__dict__["RDKFingerprint"] = rdk_fingerprint
    all_chem.__dict__["GetMorganFingerprintAsBitVect"] = morgan_fingerprint
    data_structs.__dict__["FingerprintSimilarity"] = similarity
    data_utils.__dict__["Example"] = object
    rdkit.__dict__.update(Chem=chem, DataStructs=data_structs, rdBase=rd_base)
    for module in (rdkit, chem, all_chem, data_structs, rd_base, vendi, data_utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    source = (
        Path(__file__).resolve().parents[2]
        / "worldfoundry/evaluation/tasks/metrics/vendi_score/vendi_score/molecule_utils.py"
    )
    spec = importlib.util.spec_from_file_location("_test_vendi_molecule_utils", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_fingerprint_receives_each_molecule(molecule_utils: ModuleType) -> None:
    molecules = ["ab", "bc", "ab"]
    calls: list[str] = []

    def fingerprint(molecule: str) -> frozenset[str]:
        calls.append(molecule)
        return frozenset(molecule)

    kernel = molecule_utils.get_tanimoto_K(molecules, fp=fingerprint)

    assert calls == molecules
    np.testing.assert_array_equal(kernel, [[1.0, 1 / 3, 1.0], [1 / 3, 1.0, 1 / 3], [1.0, 1 / 3, 1.0]])


@pytest.mark.parametrize("fingerprint", ["rdk", "morgan"])
def test_builtin_fingerprints_preserve_kernel(molecule_utils: ModuleType, fingerprint: str) -> None:
    kernel = molecule_utils.get_tanimoto_K(["ab", "bc"], fp=fingerprint)

    np.testing.assert_array_equal(kernel, [[1.0, 1 / 3], [1 / 3, 1.0]])


@pytest.mark.parametrize("fingerprint", ["rdk", "morgan"])
def test_empty_molecule_batch_preserves_shape(molecule_utils: ModuleType, fingerprint: str) -> None:
    assert molecule_utils.get_tanimoto_K([], fp=fingerprint).shape == (0, 0)
