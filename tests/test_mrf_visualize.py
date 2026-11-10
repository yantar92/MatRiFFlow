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


"""Tests for matrifflow.pymatgen.cli.mrf_visualize."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Force a headless backend so plots never try to open a GUI window.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from ase import Atoms
from pymatgen.core import Lattice, Structure

import matrifflow.pymatgen.cli.mrf_visualize as mrf_visualize


# ---------------------------------------------------------------------------
# synthetic ATAT data frames and files
# ---------------------------------------------------------------------------


def _predstr_df() -> pd.DataFrame:
    """A minimal predstr.out table with busy/error/ground-state rows."""
    return pd.DataFrame({
        "concentration": [0.0, 0.5, 1.0],
        "energy": [-1.0, -2.0, -3.0],
        "predicted energy": [-1.05, -2.1, -3.2],
        "index": [0, 1, 2],
        "status": ["b", "e", "bg"],
    })


def _gs_df() -> pd.DataFrame:
    """A minimal gs.out table."""
    return pd.DataFrame({
        "concentration": [0.0, 1.0],
        "energy": [-1.0, -3.0],
        "fitted energy": [-1.05, -3.2],
        "index": [0, 2],
        "sublattice deviation": [0.1, 0.3],
    })


def _fit_df() -> pd.DataFrame:
    """A minimal fit.out table with sublattice deviations."""
    return pd.DataFrame({
        "concentration": [0.0, 0.5, 1.0],
        "energy": [-1.0, -2.0, -3.0],
        "fitted energy": [-1.05, -2.1, -3.2],
        "energy delta": [0.05, 0.1, 0.2],
        "weight": [1.0, 2.0, 3.0],
        "index": [0, 1, 2],
        "sublattice deviation": [0.1, 0.2, 0.3],
    })


def _write_atat_dir(path: Path) -> None:
    """Write predstr.out / gs.out / fit.out to *path*."""
    (path / "predstr.out").write_text(
        "0.0 -1.0 -1.05 0 b\n"
        "0.5 -2.0 -2.1 1 e\n"
        "1.0 -3.0 -3.2 2 bg\n")
    (path / "gs.out").write_text(
        "0.0 -1.0 -1.05 0\n"
        "1.0 -3.0 -3.2 2\n")
    (path / "fit.out").write_text(
        "0.0 -1.0 -1.05 0.05 1.0 0\n"
        "0.5 -2.0 -2.1 0.1 2.0 1\n"
        "1.0 -3.0 -3.2 0.2 3.0 2\n")


# ---------------------------------------------------------------------------
# _to_subscript
# ---------------------------------------------------------------------------


def test_to_subscript_wraps_digits_in_latex() -> None:
    """Integer groups become LaTeX subscript markers."""
    assert mrf_visualize._to_subscript("LiC6") == "LiC$_{6}$"
    assert mrf_visualize._to_subscript("Li2C12") == "Li$_{2}$C$_{12}$"


def test_to_subscript_leaves_pure_letters_unchanged() -> None:
    """A formula without digits is returned verbatim."""
    assert mrf_visualize._to_subscript("LiC") == "LiC"


# ---------------------------------------------------------------------------
# ATAT file readers
# ---------------------------------------------------------------------------


def test_atat_read_predstr_columns(tmp_path) -> None:
    """predstr.out parses into the five documented columns."""
    _write_atat_dir(tmp_path)
    df = mrf_visualize._atat_read_predstr(tmp_path)
    assert list(df.columns) == [
        "concentration", "energy", "predicted energy", "index", "status"]
    assert df["index"].tolist() == [0, 1, 2]
    assert df["status"].tolist() == ["b", "e", "bg"]


def test_atat_read_gs_columns(tmp_path) -> None:
    """gs.out parses into the four documented columns."""
    _write_atat_dir(tmp_path)
    df = mrf_visualize._atat_read_gs(tmp_path)
    assert list(df.columns) == ["concentration", "energy", "fitted energy", "index"]
    assert df["index"].tolist() == [0, 2]


def test_atat_read_fit_columns(tmp_path) -> None:
    """fit.out parses into the six documented columns."""
    _write_atat_dir(tmp_path)
    df = mrf_visualize._atat_read_fit(tmp_path)
    assert list(df.columns) == [
        "concentration", "energy", "fitted energy", "energy delta", "weight", "index"]
    assert df["energy delta"].tolist() == [0.05, 0.1, 0.2]


def test_atat_read_clusters_filters_singlets(tmp_path) -> None:
    """Cluster size 1 is dropped; size 2+ is kept with its ECI."""
    (tmp_path / "clusters.out").write_text(
        "0\n1.5\n2\n0.1\n0.2\n")
    (tmp_path / "eci.out").write_text("0.5\n")
    df = mrf_visualize._atat_read_clusters(tmp_path)
    assert df.to_dict("list") == {
        "cluster_size": [2],
        "cluster_diameter": [1.5],
        "eci_weight": [0.5],
    }


def test_atat_read_clusters_multiple_clusters(tmp_path) -> None:
    """A single-atom cluster is skipped between two kept pairs."""
    (tmp_path / "clusters.out").write_text(
        "0\n10.0\n1\n0.9\nX\n"   # size-1 cluster, skipped
        "1\n1.5\n2\n0.1\n0.2\nY\n"  # kept pair
        "2\n2.0\n2\n0.3\n0.4\n")   # kept pair, EOF terminates
    (tmp_path / "eci.out").write_text("0.5\n0.7\n0.9\n")
    df = mrf_visualize._atat_read_clusters(tmp_path)
    assert df["cluster_size"].tolist() == [2, 2]
    assert df["cluster_diameter"].tolist() == [1.5, 2.0]
    assert df["eci_weight"].tolist() == [0.7, 0.9]


# ---------------------------------------------------------------------------
# __blue_orrd_cmap
# ---------------------------------------------------------------------------


def test_blue_orrd_cmap_none_when_range_is_empty() -> None:
    """A degenerate range has no meaningful colour mapping."""
    assert mrf_visualize.__blue_orrd_cmap(1.0, 1.0) is None


def test_blue_orrd_cmap_returns_colormap() -> None:
    """A non-degenerate range produces a ListedColormap."""
    from matplotlib.colors import ListedColormap
    cmap = mrf_visualize.__blue_orrd_cmap(0.0, 1.0)
    assert isinstance(cmap, ListedColormap)


# ---------------------------------------------------------------------------
# ATAT plotting helpers
# ---------------------------------------------------------------------------


def test_atat_plot_clusters_sets_axes() -> None:
    """The ECI panel draws and labels its axes."""
    fig, ax = plt.subplots()
    clusters = pd.DataFrame({
        "cluster_size": [2, 3],
        "cluster_diameter": [1.5, 2.5],
        "eci_weight": [0.5, 0.1],
    })
    mrf_visualize._atat_plot_clusters(ax, clusters)
    assert ax.get_title() == "ECI vs cluster diameter"
    assert ax.get_xlabel() == "Diameter, Å"
    plt.close(fig)


def test_atat_plot_fitted_energies() -> None:
    """The fitted-energies panel plots predicted and fitted series."""
    fig, ax = plt.subplots()
    mrf_visualize._atat_plot_fitted_energies(
        ax, _predstr_df(), _gs_df(), _fit_df())
    assert ax.get_title() == "Fitted Energies"
    assert len(ax.lines) == 4  # predicted, known, fitted gs, predicted gs
    plt.close(fig)


def test_atat_plot_calc_vs_fit_energies() -> None:
    """The calc-vs-fit panel plots calculated and fitted series."""
    fig, ax = plt.subplots()
    mrf_visualize._atat_plot_calc_vs_fit_energies(ax, _fit_df())
    assert ax.get_title() == "Calculated and Fitted Energies"
    assert len(ax.lines) == 2
    plt.close(fig)


def test_atat_plot_residuals_concentration() -> None:
    """Default residuals plot energy delta vs concentration."""
    fig, ax = plt.subplots()
    mrf_visualize._atat_plot_residuals(ax, "CV: 0.12", _fit_df())
    assert ax.get_title() == "Residuals of the fit\nCV: 0.12"
    assert ax.get_xlabel() == "Concentration"
    plt.close(fig)


def test_atat_plot_residuals_classic() -> None:
    """Classic residuals plot predicted vs DFT energy."""
    fig, ax = plt.subplots()
    mrf_visualize._atat_plot_residuals(
        ax, "CV: 0.12", _fit_df(), classic_residuals=True)
    assert ax.get_xlabel() == "DFT energy per reference cell, eV"
    assert ax.get_ylabel() == "Predicted energy per reference cell, eV"
    plt.close(fig)


def test_atat_plot_calculated_energies() -> None:
    """The calculated-energies panel plots known and ground-state series."""
    fig, ax = plt.subplots()
    mrf_visualize._atat_plot_calculated_energies(
        ax, _predstr_df(), _gs_df(), _fit_df())
    assert ax.get_title() == "Calculated Energies"
    plt.close(fig)


def test_atat_plot_sublattice_deviation() -> None:
    """The sublattice-deviation panel plots all and ground-state series."""
    fig, ax = plt.subplots()
    mrf_visualize._atat_plot_sublattice_deviation(ax, _gs_df(), _fit_df())
    assert ax.get_title() == "Deviation from ideal sublattice"
    assert len(ax.lines) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# hull entry readers
# ---------------------------------------------------------------------------


def test_hull_get_entries_from_pickle(tmp_path) -> None:
    """A pickle of ASE atoms becomes a ComputedStructureEntry list."""
    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    df = pd.DataFrame({
        "structure": [atoms], "total_energy": [-5.0], "fold_id": [7]})
    path = tmp_path / "db.pkl"
    df.to_pickle(path)

    entries = mrf_visualize._hull_get_entries_from_pickle(path)

    assert len(entries) == 1
    assert entries[0].composition.reduced_formula == "Li"
    assert entries[0].energy == pytest.approx(-5.0)
    assert entries[0].data["ID"] == "7"


def test_hull_get_entries_from_pickle_filter(tmp_path) -> None:
    """The filter expression is evaluated against the loaded DataFrame."""
    atoms_li = Atoms("Li", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    atoms_c = Atoms("C", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    df = pd.DataFrame({
        "structure": [atoms_li, atoms_c],
        "total_energy": [-5.0, -9.0],
        "fold_id": [1, 2],
    })
    path = tmp_path / "db.pkl"
    df.to_pickle(path)

    entries = mrf_visualize._hull_get_entries_from_pickle(
        path, filter_expr="df[df['total_energy'] < -6.0]")

    assert len(entries) == 1
    assert entries[0].composition.reduced_formula == "C"


def test_hull_get_entries_from_pickle_index_fallback(tmp_path) -> None:
    """A missing id column falls back to the DataFrame index."""
    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    df = pd.DataFrame({"structure": [atoms], "total_energy": [-5.0]})
    path = tmp_path / "db.pkl"
    df.to_pickle(path)

    entries = mrf_visualize._hull_get_entries_from_pickle(path)

    assert entries[0].data["ID"] == "0"


class _FakeVaspDir:
    """Minimal stand-in for MRFVaspDir in the recursive reader."""

    def __init__(self, structure, energy, converged=True) -> None:
        self.final_structure = structure
        self.final_energy = energy
        self.converged = converged
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_hull_get_entries_recursively_builds_entries(monkeypatch) -> None:
    """Converged runs become entries; skipped runs still get reset."""
    structure = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
    converged = _FakeVaspDir(structure, -5.0)
    unconverged = _FakeVaspDir(structure, -5.0, converged=False)
    no_energy = _FakeVaspDir(structure, None)
    vaspdirs = {
        Path("/x/a"): converged,
        Path("/x/b"): unconverged,
        Path("/x/c"): no_energy,
    }
    monkeypatch.setattr(
        mrf_visualize.MRFVaspDir, "read_vaspdirs",
        lambda root, path_filter=None: vaspdirs)

    entries = mrf_visualize._hull_get_entries_recursively(Path("/x"))

    assert len(entries) == 1
    assert entries[0].composition.reduced_formula == "Li"
    assert entries[0].energy == pytest.approx(-5.0)
    assert converged.reset_calls == 1
    assert unconverged.reset_calls == 1
    assert no_energy.reset_calls == 1


def test_hull_get_entries_recursively_path_filter(monkeypatch) -> None:
    """include/exclude regexps are applied to relative paths."""
    captured = {}

    def fake_read(root, path_filter=None):
        captured["path_filter"] = path_filter
        return {}

    monkeypatch.setattr(mrf_visualize.MRFVaspDir, "read_vaspdirs", fake_read)
    mrf_visualize._hull_get_entries_recursively(
        Path("/root"), include=r"keep", exclude=r"gorun_\d+")

    path_filter = captured["path_filter"]
    assert path_filter(Path("/root/gorun_123")) is False
    assert path_filter(Path("/root/keep_dir")) is True
    assert path_filter(Path("/root/other")) is False


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def test_add_args_dir_defaults_to_dot() -> None:
    """The positional dir defaults to the current directory."""
    parser = argparse.ArgumentParser()
    mrf_visualize.add_args(parser)
    args = parser.parse_args(["neb"])
    assert args.dir == "."
    assert args.func_derive is mrf_visualize.neb


def test_add_args_registers_all_subcommands() -> None:
    """Each subcommand wires its own dispatch function."""
    parser = argparse.ArgumentParser()
    mrf_visualize.add_args(parser)
    expected = {
        "neb": mrf_visualize.neb,
        "selective_dynamics": mrf_visualize.selective_dynamics,
        "atat": mrf_visualize.atat,
        "hull": mrf_visualize.hull,
        "voltage": mrf_visualize.voltage,
    }
    for name, func in expected.items():
        args = parser.parse_args(["somedir", name])
        assert args.func_derive is func
