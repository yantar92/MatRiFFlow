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
from pymatgen.analysis.phase_diagram import PDPlotter, PhaseDiagram
from pymatgen.core import Element, Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from unittest.mock import PropertyMock, create_autospec

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


def _visualize_args(*argv: str) -> argparse.Namespace:
    """Parse ``mrf visualize`` command-line arguments."""
    parser = argparse.ArgumentParser()
    mrf_visualize.add_args(parser)
    return parser.parse_args(list(argv))


def test_add_args_dir_defaults_to_dot() -> None:
    """The positional dir defaults to the current directory."""
    args = _visualize_args("neb")
    assert args.dir == "."
    assert args.func_derive is mrf_visualize.neb


def test_add_args_registers_all_subcommands() -> None:
    """Each subcommand wires its own dispatch function."""
    expected = {
        "neb": mrf_visualize.neb,
        "selective_dynamics": mrf_visualize.selective_dynamics,
        "atat": mrf_visualize.atat,
        "hull": mrf_visualize.hull,
        "voltage": mrf_visualize.voltage,
    }
    for name, func in expected.items():
        args = _visualize_args("somedir", name)
        assert args.func_derive is func


# ---------------------------------------------------------------------------
# synthetic hull / voltage entries
# ---------------------------------------------------------------------------


def _framework_entry(
        n_ion: int, energy_per_atom: float, ion: str = "Li"
) -> ComputedStructureEntry:
    """A six-carbon framework with N_ION intercalated ions."""
    species = ["C"] * 6 + [ion] * n_ion
    coords = [[i / 6, 0, 0] for i in range(6)] + [
        [0.5, 0.5, 0.5 + i * 0.05] for i in range(n_ion)]
    structure = Structure(Lattice.hexagonal(2.46, 6.7), species, coords)
    return ComputedStructureEntry(
        structure, energy_per_atom * len(structure),
        data={"ID": structure.composition.reduced_formula,
              "formula": structure.composition.formula})


def _pure_entry(symbol: str, energy_per_atom: float) -> ComputedStructureEntry:
    """A one-atom entry for a pure element."""
    structure = Structure(Lattice.cubic(3.5), [symbol], [[0.0, 0.0, 0.0]])
    return ComputedStructureEntry(
        structure, energy_per_atom,
        data={"ID": symbol, "formula": structure.composition.formula})


def _hull_entries() -> list[ComputedStructureEntry]:
    """A convex Li-C hull: C, LiC6, LiC2, LiC are stable; LiC3 is above."""
    return [
        _framework_entry(0, -10.0),
        _framework_entry(1, -9.2),
        _framework_entry(2, -8.35),
        _framework_entry(3, -7.95),
        _framework_entry(6, -6.8),
        _pure_entry("Li", -2.0),
    ]


def _mock_vaspdir(
        *, final_structure=None, initial_structure=None,
        energy=None, converged=True, nebp=False):
    """An MRFVaspDir mock exposing the attributes the CLI reads."""
    vaspdir = create_autospec(mrf_visualize.MRFVaspDir, instance=True)
    # MRFVaspDir is a Mapping, so the mock would otherwise be empty/falsy.
    vaspdir.__len__.return_value = 1
    vaspdir.final_structure = final_structure
    vaspdir.initial_structure = initial_structure
    vaspdir.final_energy = energy
    vaspdir.converged = converged
    vaspdir.nebp = nebp
    return vaspdir


def _patch_hull_entries(monkeypatch, entries) -> None:
    """Replace the recursive VASP scan with fixed entries."""
    monkeypatch.setattr(
        mrf_visualize, "_hull_get_entries_recursively",
        lambda path, include=None, exclude=None: list(entries))


def _patch_extra_vaspdirs(monkeypatch, mapping) -> None:
    """Resolve MRFVaspDir(path) from MAPPING[keyed by path.name]."""
    def factory(path):
        spec = mapping[path.name]
        if isinstance(spec, Exception):
            raise spec
        return spec
    monkeypatch.setattr(mrf_visualize, "MRFVaspDir", factory)


## hull


def _table_ids(path: Path) -> set[str]:
    """IDs recorded in the first column of a written energy table."""
    lines = path.read_text().splitlines()
    return {line.split(" ", 1)[0] for line in lines[1:]}


def test_hull_writes_plots_and_tables(monkeypatch, tmp_path, capsys) -> None:
    """The written tables agree with the phase diagram from the same entries."""
    monkeypatch.chdir(tmp_path)
    entries = _hull_entries()
    _patch_hull_entries(monkeypatch, entries)
    args = _visualize_args(str(tmp_path), "hull")

    assert mrf_visualize.hull(args) == 0

    out = capsys.readouterr().out
    assert "All energies saved to formation_en.txt" in out
    assert "GS energies saved to formation_en_gs.txt" in out
    assert "Min energies saved to formation_en_min.txt" in out
    for name in ("formation_en.png", "formation_en.svg",
                 "formation_en.txt", "formation_en_gs.txt",
                 "formation_en_min.txt"):
        assert (tmp_path / name).is_file()

    phase_diagram = PhaseDiagram(
        entries, elements=[Element("C"), Element("Li")])
    _, stable_entries, unstable_entries = PDPlotter(
        phase_diagram, show_unstable=0.2).pd_plot_data

    all_ids = _table_ids(tmp_path / "formation_en.txt")
    assert all_ids == {
        str(entry.data["ID"])
        for entry in [*unstable_entries, *stable_entries.values()]}

    ground_state_ids = _table_ids(tmp_path / "formation_en_gs.txt")
    assert ground_state_ids == {
        str(entry.data["ID"]) for entry in stable_entries.values()}

    min_entries = pd.read_csv(tmp_path / "formation_en_min.txt", sep=" ")
    assert set(min_entries["Reduced formula"]) == {
        entry.composition.reduced_formula for entry in entries}


def test_hull_svg_format_skips_png(monkeypatch, tmp_path) -> None:
    """--format svg skips the raster output."""
    monkeypatch.chdir(tmp_path)
    _patch_hull_entries(monkeypatch, _hull_entries())
    args = _visualize_args(str(tmp_path), "hull", "--format", "svg")

    assert mrf_visualize.hull(args) == 0

    assert (tmp_path / "formation_en.svg").is_file()
    assert not (tmp_path / "formation_en.png").exists()


def test_hull_forwards_plot_arguments(monkeypatch, tmp_path) -> None:
    """Plot options reach the custom plot helper with their parsed values."""
    monkeypatch.chdir(tmp_path)
    _patch_hull_entries(monkeypatch, _hull_entries())
    plot = create_autospec(mrf_visualize._hull_plot_custom_phase_diagram)
    monkeypatch.setattr(
        mrf_visualize, "_hull_plot_custom_phase_diagram", plot)
    args = _visualize_args(
        str(tmp_path), "hull", "--format", "svg",
        "--title", "Custom hull", "--font_size", "8",
        "--ymin", "-1200", "--ymax", "100",
        "--max_composition", "LiC2", "--show_unstable", "0.05")

    assert mrf_visualize.hull(args) == 0

    plot.assert_called_once()
    phase_diagram, _, ion, matrix = plot.call_args.args
    assert isinstance(phase_diagram, PhaseDiagram)
    assert ion == "Li"
    assert matrix == "C"
    kwargs = plot.call_args.kwargs
    assert kwargs["title"] == "Custom hull"
    assert kwargs["font_size"] == 8
    assert kwargs["ymin"] == -1200
    assert kwargs["ymax"] == 100
    assert kwargs["show_unstable"] == 0.05
    assert kwargs["max_conc"] == pytest.approx(1 / 3)


def test_hull_plot_applies_title_and_axis_limits(
        tmp_path, monkeypatch) -> None:
    """Title, axis limits, and max composition control the axes."""
    monkeypatch.chdir(tmp_path)
    entries = _hull_entries()
    phd = PhaseDiagram(entries, elements=[Element("C"), Element("Li")])
    _, ax = plt.subplots()

    mrf_visualize._hull_plot_custom_phase_diagram(
        phd, ax, "Li", "C", font_size=8, title="My hull",
        ymin=-1200, ymax=100, max_conc=0.4)

    assert ax.get_title() == "My hull"
    assert ax.get_xlim() == pytest.approx((-0.05, 0.45))
    low, high = ax.get_ylim()
    assert low < -1200
    assert high >= 100
    plt.close(plt.gcf())


def test_hull_reports_missing_matrix_entry(
        monkeypatch, tmp_path, capsys) -> None:
    """No pure non-ion element is a readable error, not a traceback."""
    monkeypatch.chdir(tmp_path)
    _patch_hull_entries(monkeypatch, [_pure_entry("Na", -1.3)])
    args = _visualize_args(str(tmp_path), "hull", "--ion", "Na")

    assert mrf_visualize.hull(args) == 1
    assert "No pure element (matrix) entry found" in capsys.readouterr().out


def test_hull_reports_missing_ion_entry(
        monkeypatch, tmp_path, capsys) -> None:
    """No pure working-ion element is a readable error."""
    monkeypatch.chdir(tmp_path)
    _patch_hull_entries(monkeypatch, [_pure_entry("C", -10.0)])
    args = _visualize_args(str(tmp_path), "hull", "--ion", "Na")

    assert mrf_visualize.hull(args) == 1
    assert "No pure Na (ion) entry found" in capsys.readouterr().out


def test_hull_reports_ambiguous_matrix_element(
        monkeypatch, tmp_path, capsys) -> None:
    """Two distinct pure non-ion elements leave the matrix ambiguous."""
    monkeypatch.chdir(tmp_path)
    _patch_hull_entries(monkeypatch, [
        _pure_entry("C", -10.0),
        _pure_entry("Na", -1.3),
        _pure_entry("Li", -2.0),
    ])
    args = _visualize_args(str(tmp_path), "hull", "--ion", "Li")

    assert mrf_visualize.hull(args) == 1
    out = capsys.readouterr().out
    assert "Multiple distinct matrix element candidates" in out
    assert "C, Na" in out


def test_hull_extra_dirs_add_valid_and_skip_invalid(
        monkeypatch, tmp_path, capsys) -> None:
    """--extra_dir entries are added, invalid ones skipped, all reset."""
    monkeypatch.chdir(tmp_path)
    _patch_hull_entries(monkeypatch, _hull_entries())
    structure = _framework_entry(5, -7.2).structure
    good = _mock_vaspdir(
        final_structure=structure, energy=-7.2 * len(structure))
    unconverged = _mock_vaspdir(
        final_structure=structure, energy=-80.0, converged=False)
    no_energy = _mock_vaspdir(final_structure=structure)
    _patch_extra_vaspdirs(monkeypatch, {
        "good": good,
        "unconverged": unconverged,
        "no_energy": no_energy,
        "boom": ValueError("broken run"),
    })
    args = _visualize_args(
        str(tmp_path), "hull", "--extra_dir",
        str(tmp_path / "good"), str(tmp_path / "unconverged"),
        str(tmp_path / "no_energy"), str(tmp_path / "boom"))

    assert mrf_visualize.hull(args) == 0

    out = capsys.readouterr().out
    assert f"Added {tmp_path / 'good'}" in out
    assert "unconverged" in out
    assert "no final energy" in out
    assert "broken run" in out
    good.reset.assert_called_once_with()
    unconverged.reset.assert_called_once_with()
    no_energy.reset.assert_called_once_with()
    # The extra ground state is written under its directory path.
    assert any(
        line.startswith(str(tmp_path / "good"))
        for line in (tmp_path / "formation_en_gs.txt").read_text().splitlines())


def _write_emc2(path: Path, temperature: float) -> None:
    """Write a minimal emc2 table with constant temperature."""
    rows = [(c, -1.0 - 2.0 * c, -0.9 - 2.0 * c) for c in (0.0, 0.5, 1.0)]
    path.write_text("".join(
        f"{temperature}\t0\t{e}\t{x}\t{f}\n"
        for (x, e, f) in [(2 * c - 1, e, f) for c, e, f in rows]))


@pytest.mark.parametrize("temperature,vibrational", [
    (0.0, False),
    (300.0, False),
    (300.0, True),
])
def test_hull_entropy_adjustments(
        monkeypatch, tmp_path, capsys, temperature, vibrational) -> None:
    """--entropy and --entropy_vibrational both reach PhaseDiagram."""
    monkeypatch.chdir(tmp_path)
    _patch_hull_entries(monkeypatch, _hull_entries())
    entropy_file = tmp_path / "emc2.txt"
    _write_emc2(entropy_file, temperature)
    argv = [str(tmp_path), "hull", "--entropy", str(entropy_file)]
    if vibrational:
        argv.append("--entropy_vibrational")
    args = _visualize_args(*argv)

    assert mrf_visualize.hull(args) == 0

    assert f"Adding entropy adjustments for T={temperature}" in \
        capsys.readouterr().out
    assert (tmp_path / "formation_en.png").is_file()


def test_hull_reads_pickle_input(tmp_path, monkeypatch) -> None:
    """A file positional is read as a DataFrame pickle of ASE structures."""
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame([
        {"structure": AseAtomsAdaptor.get_atoms(entry.structure),
         "total_energy": entry.energy,
         "fold_id": entry.data["ID"]}
        for entry in _hull_entries()])
    path = tmp_path / "db.pkl"
    df.to_pickle(path)

    args = _visualize_args(str(path), "hull")

    assert mrf_visualize.hull(args) == 0
    assert (tmp_path / "formation_en.png").is_file()


## voltage


@pytest.mark.parametrize(
    "xaxis,expected_column,expected_label", [
        ("capacity_grav", "capacity", "Capacity (mAh/g)"),
        ("capacity_vol", "capacity", "Capacity (Ah/l)"),
        ("x_form", "x", "x in Li<sub>x</sub>Host"),
        ("frac_x", "x", "Atomic Fraction of Li"),
    ])
def test_voltage_writes_profile_for_each_xaxis(
        monkeypatch, tmp_path, xaxis, expected_column, expected_label) -> None:
    """Every --xaxis choice plots and writes the matching quantity."""
    _patch_hull_entries(monkeypatch, _hull_entries())
    step = create_autospec(plt.step)
    xlabel = create_autospec(plt.xlabel)
    ylabel = create_autospec(plt.ylabel)
    monkeypatch.setattr(plt, "step", step)
    monkeypatch.setattr(plt, "xlabel", xlabel)
    monkeypatch.setattr(plt, "ylabel", ylabel)
    args = _visualize_args(str(tmp_path), "voltage", "--xaxis", xaxis)

    assert mrf_visualize.voltage(args) == 0

    for name in ("voltage.png", "voltage.svg", "voltage.out"):
        assert (tmp_path / name).is_file()
    df = pd.read_csv(tmp_path / "voltage.out", sep=" ")
    plotted_x, plotted_voltage = step.call_args.args
    assert list(plotted_x) == pytest.approx(df[expected_column].tolist())
    assert list(plotted_voltage) == pytest.approx(df["voltage"].tolist())
    assert xlabel.call_args.args[0] == expected_label
    assert ylabel.call_args.args[0] == "Voltage vs. Li/Li+ (V)"
    assert df["x"].is_monotonic_increasing


def test_voltage_svg_format_skips_png(monkeypatch, tmp_path) -> None:
    """--format svg skips the raster output."""
    _patch_hull_entries(monkeypatch, _hull_entries())
    args = _visualize_args(
        str(tmp_path), "voltage", "--format", "svg", "--dpi", "72")

    assert mrf_visualize.voltage(args) == 0

    assert (tmp_path / "voltage.svg").is_file()
    assert not (tmp_path / "voltage.png").exists()


def test_voltage_missing_ion_raises(monkeypatch, tmp_path) -> None:
    """Without a pure working-ion entry there is no voltage reference."""
    _patch_hull_entries(monkeypatch, [_framework_entry(0, -10.0)])
    args = _visualize_args(str(tmp_path), "voltage", "--ion", "Na")

    with pytest.raises(ValueError, match="No pure Na entry found"):
        mrf_visualize.voltage(args)


def test_voltage_extra_dirs_add_valid_and_skip_invalid(
        monkeypatch, tmp_path, capsys) -> None:
    """--extra_dir entries join the electrode; invalid ones are skipped."""
    _patch_hull_entries(monkeypatch, _hull_entries())
    structure = _framework_entry(4, -7.5).structure
    good = _mock_vaspdir(
        final_structure=structure, energy=-7.5 * len(structure))
    unconverged = _mock_vaspdir(
        final_structure=structure, energy=-80.0, converged=False)
    no_energy = _mock_vaspdir(final_structure=structure)
    _patch_extra_vaspdirs(monkeypatch, {
        "good": good,
        "unconverged": unconverged,
        "no_energy": no_energy,
        "boom": ValueError("broken run"),
    })
    args = _visualize_args(
        str(tmp_path), "voltage", "--extra_dir",
        str(tmp_path / "good"), str(tmp_path / "unconverged"),
        str(tmp_path / "no_energy"), str(tmp_path / "boom"))

    assert mrf_visualize.voltage(args) == 0

    out = capsys.readouterr().out
    assert f"Added {tmp_path / 'good'}" in out
    assert "unconverged" in out
    assert "no final energy" in out
    assert "broken run" in out
    good.reset.assert_called_once_with()
    unconverged.reset.assert_called_once_with()
    no_energy.reset.assert_called_once_with()
    # Li4C6 is a stable composition only present via --extra_dir.
    df = pd.read_csv(tmp_path / "voltage.out", sep=" ")
    assert any(abs(x - 4 / 10) < 1e-9 for x in df["x"])


def test_voltage_reads_pickle_and_writes_next_to_it(tmp_path) -> None:
    """A pickle positional writes its outputs into the pickle's directory."""
    df = pd.DataFrame([
        {"structure": AseAtomsAdaptor.get_atoms(entry.structure),
         "total_energy": entry.energy,
         "fold_id": entry.data["ID"]}
        for entry in _hull_entries()])
    path = tmp_path / "db.pkl"
    df.to_pickle(path)
    args = _visualize_args(str(path), "voltage")

    assert mrf_visualize.voltage(args) == 0

    assert (tmp_path / "voltage.out").is_file()
    assert (tmp_path / "voltage.png").is_file()


## visualise dispatch and remaining branches


def test_hull_recursive_reader_skips_raising_vaspdir(monkeypatch, capsys) -> None:
    """A malformed run is skipped, counted, and still reset."""
    broken = _mock_vaspdir()
    type(broken).final_energy = PropertyMock(
        side_effect=RuntimeError("corrupt vasprun.xml"))
    monkeypatch.setattr(
        mrf_visualize.MRFVaspDir, "read_vaspdirs",
        lambda root, path_filter=None: {Path("/x/broken"): broken})

    entries = mrf_visualize._hull_get_entries_recursively(Path("/x"))

    assert entries == []
    out = capsys.readouterr().out
    assert "Skipping /x/broken: corrupt vasprun.xml" in out
    assert "Read 1 runs" in out
    assert "Skipped: 1" in out
    broken.reset.assert_called_once_with()


def test_hull_plot_labels_stable_phases_right_of_center(
        tmp_path, monkeypatch) -> None:
    """A stable phase at x > 0.5 gets a left-aligned label above the point."""
    monkeypatch.chdir(tmp_path)
    entries = [
        _framework_entry(0, -10.0),
        _framework_entry(6, -6.8),
        _framework_entry(8, -6.128571428571429),
        _pure_entry("Li", -2.0),
    ]
    phd = PhaseDiagram(entries, elements=[Element("C"), Element("Li")])
    _, ax = plt.subplots()

    mrf_visualize._hull_plot_custom_phase_diagram(
        phd, ax, "Li", "C", font_size=8, title="Right labels")

    label = next(
        text for text in ax.texts if text.get_text() == "Li$_{4}$C$_{3}$")
    assert label.get_horizontalalignment() == "left"
    assert label.get_verticalalignment() == "top"
    entry = next(
        entry for entry in entries
        if entry.composition.reduced_formula == "Li4C3")
    x, y = label.xy
    assert x == pytest.approx(
        entry.composition.get_atomic_fraction(Element("Li")))
    assert y == pytest.approx(phd.get_form_energy_per_atom(entry) * 1000)
    plt.close(plt.gcf())


def test_visualize_dispatches_to_subcommand(monkeypatch) -> None:
    """visualize() calls the dispatch function stored on the namespace."""
    calls = []
    args = argparse.Namespace(func_derive=lambda a: calls.append(a) or 7)

    assert mrf_visualize.visualize(args) == 7
    assert calls == [args]


## neb


def test_neb_exports_converged_trajectory(
        monkeypatch, tmp_path, capsys, cscl) -> None:
    """Converged NEB runs get a CIF; the other runs are skipped."""
    # Create the directories os.walk will visit.
    for name in ("good", "unconverged", "not_neb"):
        (tmp_path / name).mkdir()
    displaced = cscl.copy()
    displaced.translate_sites(0, [0.05, 0.0, 0.0], frac_coords=True)
    good = _mock_vaspdir()
    good.nebp = True
    good.neb_dirs.return_value = [
        _mock_vaspdir(final_structure=cscl),
        _mock_vaspdir(initial_structure=displaced),
    ]
    unconverged = _mock_vaspdir(converged=False)
    unconverged.nebp = True
    not_neb = _mock_vaspdir()
    not_neb.nebp = False
    vaspdirs = {
        "good": good,
        "unconverged": unconverged,
        "not_neb": not_neb,
    }
    other = _mock_vaspdir()
    monkeypatch.setattr(
        mrf_visualize, "MRFVaspDir",
        lambda path: vaspdirs.get(Path(path).name, other))

    args = argparse.Namespace(dir=str(tmp_path))

    assert mrf_visualize.neb(args) == 0

    cif = tmp_path / "good" / "NEB_trajectory_converged.cif"
    merged = Structure.from_file(cif)
    # Two CsCl images with one displaced site merge into three sites.
    assert len(merged) == 3
    assert not (tmp_path / "unconverged" / "NEB_trajectory_converged.cif").exists()
    assert not (tmp_path / "not_neb" / "NEB_trajectory_converged.cif").exists()
    assert "Saved final trajectory to NEB_trajectory_converged.cif" in \
        capsys.readouterr().out


## selective dynamics


def test_selective_dynamics_writes_cif_for_constrained_sites(
        cscl, tmp_path, capsys) -> None:
    """A POSCAR with fixed sites produces selective_dynamics.cif."""
    Poscar(
        cscl,
        selective_dynamics=[
            [False, False, False],
            [True, True, True],
        ],
    ).write_file(tmp_path / "POSCAR")
    args = argparse.Namespace(dir=str(tmp_path))

    mrf_visualize.selective_dynamics(args)

    assert (tmp_path / "selective_dynamics.cif").is_file()
    assert "Saved selective_dynamics to selective_dynamics.cif" in \
        capsys.readouterr().out


def test_selective_dynamics_skips_unconstrained_and_missing_poscar(
        cscl, tmp_path, capsys) -> None:
    """No constraint or no POSCAR means no CIF and no output line."""
    (tmp_path / "plain").mkdir()
    Poscar(cscl).write_file(tmp_path / "plain" / "POSCAR")
    (tmp_path / "empty").mkdir()
    args = argparse.Namespace(dir=str(tmp_path))

    mrf_visualize.selective_dynamics(args)

    assert not (tmp_path / "plain" / "selective_dynamics.cif").exists()
    assert not (tmp_path / "empty" / "selective_dynamics.cif").exists()
    assert capsys.readouterr().out == ""
