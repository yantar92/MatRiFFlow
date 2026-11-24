# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of MatRiFFlow package
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Tests for matrifflow.pymatgen.io.vasp.diagnostics.

These records and containers are our own data structures, so the
MSONable round-trip contract is exercised here directly (see the
testing principle in tests/conftest.py).
"""

from __future__ import annotations

import pickle
import warnings

import pytest
from pymatgen.util.testing import MatSciTest

from matrifflow.pymatgen.io.vasp.diagnostics import (
    VaspWarning,
    VaspWarningRecord,
    VaspWarnings,
)


class TestVaspWarningRecord(MatSciTest):
    def test_merge_accumulates_count_and_takes_newer(self) -> None:
        """message/tips/metadata take the newer value; count accumulates.

        source uses the newer non-empty value, falling back to the original.
        """
        left = VaspWarningRecord(
            name="x", message="old", tips=["a"], count=2,
            source="s1", metadata={"k": 1},
        )
        right = VaspWarningRecord(
            name="x", message="new", tips=["b"], count=3,
            source="s2", metadata={"k": 2, "j": 3},
        )
        merged = left.merge(right)
        assert merged.name == "x"
        assert merged.message == "new"
        assert merged.tips == ["b"]
        assert merged.count == 5
        assert merged.source == "s2"
        assert merged.metadata == {"k": 2, "j": 3}

    def test_merge_empty_tips_keeps_original(self) -> None:
        """A record with no tips keeps the original tips on merge."""
        left = VaspWarningRecord(name="x", message="m", tips=["a"])
        right = VaspWarningRecord(name="x", message="m2")
        assert left.merge(right).tips == ["a"]

    def test_merge_preserves_missing_details_without_mutating_inputs(self) -> None:
        """Missing newer details retain old values; metadata merges by key."""
        left = VaspWarningRecord(
            name='x', message='old', tips=['check INCAR'], count=2,
            source='OUTCAR', metadata={'retained': 1, 'replaced': 2},
        )
        right = VaspWarningRecord(
            name='x', message='new', count=3,
            metadata={'replaced': 4, 'added': 5},
        )

        merged = left.merge(right)

        assert merged == VaspWarningRecord(
            name='x', message='new', tips=['check INCAR'], count=5,
            source='OUTCAR', metadata={'retained': 1, 'replaced': 4, 'added': 5},
        )
        assert left.message == 'old'
        assert left.count == 2
        assert left.metadata == {'retained': 1, 'replaced': 2}
        assert right.metadata == {'replaced': 4, 'added': 5}

    def test_as_dict_copies_mutable_fields(self) -> None:
        """Editing serialized tips or metadata does not change the record."""
        record = VaspWarningRecord(
            name='x', message='message', tips=['original'], metadata={'steps': 2},
        )
        serialized = record.as_dict()
        serialized['tips'].append('new')
        serialized['metadata']['steps'] = 3

        assert record.tips == ['original']
        assert record.metadata == {'steps': 2}

    def test_from_dict_preserves_input(self) -> None:
        """Reading MSON metadata does not remove it from the caller's dict."""
        record = VaspWarningRecord(name='x', message='message')
        serialized = record.as_dict()
        original = serialized.copy()

        assert VaspWarningRecord.from_dict(serialized) == record
        assert serialized == original

    def test_merge_mismatched_names_raises(self) -> None:
        """Merging records with different names is rejected."""
        with pytest.raises(ValueError):
            VaspWarningRecord(name="a", message="m").merge(
                VaspWarningRecord(name="b", message="m"))

    def test_msonable(self) -> None:
        record = VaspWarningRecord(
            name="x", message="m", tips=["t"], count=2,
            source="s", metadata={"k": 1},
        )
        self.assert_msonable(record)


class TestVaspWarnings(MatSciTest):
    def test_add_merges_same_name(self) -> None:
        """Adding two records with the same name merges them."""
        warnings = VaspWarnings()
        warnings.add(VaspWarningRecord(name="x", message="m", count=1))
        warnings.add(VaspWarningRecord(name="x", message="m2", count=1))
        assert len(warnings) == 1
        assert warnings["x"].count == 2
        assert warnings["x"].message == "m2"

    def test_overwrite_replaces(self) -> None:
        """overwrite replaces a record instead of merging it."""
        warnings = VaspWarnings()
        warnings.add(VaspWarningRecord(name="x", message="m", count=1))
        warnings.overwrite(VaspWarningRecord(name="x", message="m3", count=7))
        assert warnings["x"].count == 7
        assert warnings["x"].message == "m3"

    def test_iter_names_len_has(self) -> None:
        """The container exposes iteration, names, length, and membership."""
        warnings = VaspWarnings()
        warnings.add(VaspWarningRecord(name="x", message="m"))
        warnings.add(VaspWarningRecord(name="y", message="m"))
        assert list(warnings) == ["x", "y"]
        assert warnings.names() == {"x", "y"}
        assert warnings.has("x")
        assert not warnings.has("z")

    def test_init_from_records(self) -> None:
        """A constructor list is folded through add(), merging duplicates."""
        warnings = VaspWarnings([
            VaspWarningRecord(name="x", message="m"),
            VaspWarningRecord(name="x", message="m2"),
        ])
        assert len(warnings) == 1
        assert warnings["x"].count == 2

    def test_emit(self) -> None:
        """emit warns with the VaspWarning category per record."""
        warnings = VaspWarnings([VaspWarningRecord(name="x", message="hello")])
        with pytest.warns(VaspWarning, match="hello"):
            warnings.emit()

    @pytest.mark.parametrize('operation', ['add', 'overwrite'])
    @pytest.mark.parametrize('emit', [False, True])
    def test_mutation_emission_policy(self, operation: str, emit: bool) -> None:
        """Mutations store records but notify only when requested."""
        diagnostics = VaspWarnings()
        original = VaspWarningRecord(name='x', message='old', count=2)
        incoming = VaspWarningRecord(name='x', message='new', count=3)
        diagnostics.add(original, emit=False)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            getattr(diagnostics, operation)(incoming, emit=emit)

        assert diagnostics['x'].count == (5 if operation == 'add' else 3)
        assert diagnostics['x'].message == 'new'
        assert len(captured) == int(emit)
        if emit:
            assert captured[0].category is VaspWarning
            assert str(captured[0].message) == 'new'

    def test_overwrite_inserts_and_preserves_order(self) -> None:
        """Overwrite can insert a name and does not reorder existing names."""
        diagnostics = VaspWarnings()
        first = VaspWarningRecord(name='x', message='first')
        second = VaspWarningRecord(name='y', message='second')
        replacement = VaspWarningRecord(name='x', message='replacement')
        diagnostics.overwrite(first, emit=False)
        diagnostics.add(second, emit=False)
        diagnostics.overwrite(replacement, emit=False)

        assert list(diagnostics.items()) == [('x', replacement), ('y', second)]
        assert list(diagnostics.values()) == [replacement, second]
        with pytest.raises(KeyError, match='missing'):
            diagnostics['missing']

    def test_emit_formats_relative_sources_and_uses_custom_category(self) -> None:
        """Explicit emission reports sources relative to the working directory."""
        source = self.tmp_path / 'calculation' / 'OUTCAR'
        diagnostics = VaspWarnings()
        diagnostics.add(
            VaspWarningRecord(name='x', message='problem', source=str(source)),
            emit=False,
        )
        diagnostics.add(VaspWarningRecord(name='y', message='plain'), emit=False)

        with pytest.warns(RuntimeWarning) as captured:
            diagnostics.emit(category=RuntimeWarning)

        assert [str(item.message) for item in captured] == [
            'calculation/OUTCAR: problem', 'plain',
        ]
        assert all(item.category is RuntimeWarning for item in captured)
        assert diagnostics['x'].source == str(source)

    @pytest.mark.parametrize('serialization', ['mson', 'pickle'])
    def test_deserialization_reemits_restored_records(self, serialization: str) -> None:
        """Both persistence paths restore full records and notify once per name."""
        diagnostics = VaspWarnings()
        records = [
            VaspWarningRecord(
                name='x', message='first', tips=['retry'], count=3,
                source='OUTCAR', metadata={'steps': 12},
            ),
            VaspWarningRecord(name='y', message='second'),
        ]
        for record in records:
            diagnostics.add(record, emit=False)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            payload = diagnostics.as_dict() if serialization == 'mson' \
                else pickle.dumps(diagnostics)
        assert not captured

        with pytest.warns(VaspWarning) as captured:
            restored = VaspWarnings.from_dict(payload) if serialization == 'mson' \
                else pickle.loads(payload)

        assert isinstance(restored, VaspWarnings)
        assert list(restored.values()) == records
        assert [str(item.message) for item in captured] == ['OUTCAR: first', 'second']
        assert all(item.category is VaspWarning for item in captured)

    def test_empty_container_emits_nothing(self) -> None:
        """An empty diagnostic collection stays silent, including on restore."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            diagnostics = VaspWarnings.from_dict(VaspWarnings().as_dict())
            diagnostics.emit()

        assert not captured
        assert len(diagnostics) == 0
        assert list(diagnostics.items()) == []

    def test_msonable(self) -> None:
        warnings = VaspWarnings([
            VaspWarningRecord(
                name="x", message="m", tips=["t"], count=1,
                source="s", metadata={"k": 1},
            )
        ])
        self.assert_msonable(warnings)
