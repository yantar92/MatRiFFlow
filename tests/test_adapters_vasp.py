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


"""Tests for matrifflow.gorun.adapters.vasp."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import pytest

import matrifflow.gorun.adapters.vasp as vasp_mod

from matrifflow.gorun.adapters.vasp import (
    VaspAdapter,
    _has_incar_chain,
    _modify_incar,
    _rotate_incar_chain,
)


def _adapter(**kwargs) -> VaspAdapter:
    return VaspAdapter(**kwargs)


# --- validate_environment ---------------------------------------------------


def test_validate_environment_missing_vars_raises(monkeypatch) -> None:
    """Missing VASP_PATH raises OSError."""
    monkeypatch.delenv("VASP_PATH", raising=False)
    with pytest.raises(OSError):
        _adapter().validate_environment()


def test_validate_environment_ok(monkeypatch, tmp_path: Path) -> None:
    """VASP_PATH with the selected binary present passes validation."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "vasp_ncl").touch()
    monkeypatch.setenv("VASP_PATH", str(tmp_path))
    _adapter().validate_environment()


def test_validate_environment_missing_binary(monkeypatch, tmp_path: Path) -> None:
    """VASP_PATH set but the selected binary absent raises OSError."""
    monkeypatch.setenv("VASP_PATH", str(tmp_path))
    with pytest.raises(OSError, match="binary not found"):
        _adapter().validate_environment()


# --- is_valid_input ---------------------------------------------------------


def test_is_valid_input_requires_incar(tmp_path: Path) -> None:
    """is_valid_input is True only when INCAR exists."""
    assert _adapter().is_valid_input(tmp_path) is False
    (tmp_path / "INCAR").touch()
    assert _adapter().is_valid_input(tmp_path) is True


# --- is_converged -----------------------------------------------------------


def test_is_converged_false_when_not_converged(monkeypatch, tmp_path: Path) -> None:
    """Not converged according to the cleanVASP helper."""
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: False)
    assert _adapter().is_converged(tmp_path) is False


def test_is_converged_md_true_without_chain(monkeypatch, tmp_path: Path) -> None:
    """An MD run that converged is True regardless of INCAR chains."""
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: True)
    (tmp_path / "INCAR.0").touch()
    assert _adapter().is_converged(tmp_path) is True


def test_is_converged_false_with_chain(monkeypatch, tmp_path: Path) -> None:
    """Converged but with INCAR chains remaining is not done."""
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: False)
    (tmp_path / "INCAR.0").touch()
    assert _adapter().is_converged(tmp_path) is False


def test_is_converged_true_without_chain(monkeypatch, tmp_path: Path) -> None:
    """Converged, non-MD, and no chains is done."""
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: False)
    assert _adapter().is_converged(tmp_path) is True


# --- has_previous_output ----------------------------------------------------


def test_has_previous_output_incar_py(tmp_path: Path) -> None:
    """An INCAR.py counts as previous output when not suppressed."""
    (tmp_path / "INCAR.py").touch()
    assert _adapter().has_previous_output(tmp_path) is True


def test_has_previous_output_incar_py_suppressed(tmp_path: Path) -> None:
    """--no-incar-py ignores INCAR.py."""
    (tmp_path / "INCAR.py").touch()
    assert _adapter(no_incar_py=True).has_previous_output(tmp_path) is False


def test_has_previous_output_from_vasp(monkeypatch, tmp_path: Path) -> None:
    """Without INCAR.py, the cleanVASP output check decides."""
    monkeypatch.setattr(
        vasp_mod, "directory_contains_vasp_outputp", lambda path: True
    )
    assert _adapter(no_incar_py=True).has_previous_output(tmp_path) is True


# --- _has_incar_chain / _rotate_incar_chain ---------------------------------


def test_has_incar_chain_false_when_empty(tmp_path: Path) -> None:
    """No INCAR.N files means no chain."""
    assert _has_incar_chain(tmp_path) is False


def test_has_incar_chain_true_with_numbered_files(tmp_path: Path) -> None:
    """INCAR.0 / INCAR.1 files mark a chain."""
    (tmp_path / "INCAR.0").touch()
    (tmp_path / "INCAR.1").touch()
    assert _has_incar_chain(tmp_path) is True


def test_has_incar_chain_ignores_incar_old(tmp_path: Path) -> None:
    """INCAR.old is not a chain member."""
    (tmp_path / "INCAR.old").touch()
    assert _has_incar_chain(tmp_path) is False


def test_rotate_incar_chain_noop_when_not_converged(monkeypatch, tmp_path: Path) -> None:
    """Not converged means no rotation."""
    (tmp_path / "INCAR.0").touch()
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: False)
    _rotate_incar_chain(tmp_path)
    assert (tmp_path / "INCAR").exists() is False
    assert (tmp_path / "INCAR.0").exists()


def test_rotate_incar_chain_rotates_first(monkeypatch, tmp_path: Path) -> None:
    """The first INCAR.N is renamed to INCAR; the old INCAR becomes INCAR.old."""
    (tmp_path / "INCAR").write_text("old")
    (tmp_path / "INCAR.0").write_text("first")
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    _rotate_incar_chain(tmp_path)
    assert (tmp_path / "INCAR").read_text() == "first"
    assert (tmp_path / "INCAR.old").read_text() == "old"
    assert not (tmp_path / "INCAR.0").exists()


# --- _modify_incar ----------------------------------------------------------


def test_modify_incar_changes_value(tmp_path: Path) -> None:
    """A key is set and written back to INCAR."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\n")
    _modify_incar(tmp_path, {"ALGO": "Normal"})
    content = (tmp_path / "INCAR").read_text()
    assert "ALGO = Normal" in content
    assert "ENCUT" in content


def test_modify_incar_none_removes_key(tmp_path: Path) -> None:
    """A value of the string None removes the key."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\nEDIFF = 1e-6\n")
    _modify_incar(tmp_path, {"EDIFF": "None"})
    assert "EDIFF" not in (tmp_path / "INCAR").read_text()


# --- setup_commands ---------------------------------------------------------


def test_setup_commands_vasp_section(tmp_path: Path) -> None:
    """The new [vasp] setup key is preferred."""
    cfg = {"vasp": {"setup": "module load vasp"}}
    assert _adapter().setup_commands(cfg) == "module load vasp"


def test_setup_commands_legacy_fallback() -> None:
    """Without [vasp], the legacy VASP-setup key is used."""
    cfg = {"VASP-setup": "module load legacy"}
    assert _adapter().setup_commands(cfg) == "module load legacy"


def test_setup_commands_no_vasp_config() -> None:
    """--no-vasp-config returns an empty string."""
    cfg = {"vasp": {"setup": "module load vasp"}}
    assert _adapter(no_vasp_config=True).setup_commands(cfg) == ""


# --- _direct_vasp_script ----------------------------------------------------


def test_direct_vasp_script_default_mpiexec(monkeypatch) -> None:
    """The direct script exports VASP_COMMAND with the default launcher."""
    monkeypatch.setenv("VASP_PATH", "/vasp")
    adapter = _adapter(vasp_variant="std")
    script = adapter._direct_vasp_script({})
    assert 'export VASP_COMMAND="mpiexec /vasp/bin/vasp_std"' in script
    assert script.endswith("$VASP_COMMAND")


def test_direct_vasp_script_custom_mpiexec(monkeypatch) -> None:
    """A server mpiexec key overrides the default launcher."""
    monkeypatch.setenv("VASP_PATH", "/vasp")
    adapter = _adapter()
    script = adapter._direct_vasp_script({"mpiexec": "srun"})
    assert 'export VASP_COMMAND="srun /vasp/bin/vasp_ncl"' in script


def test_direct_vasp_script_explicit_empty_mpiexec(monkeypatch) -> None:
    """An explicit empty mpiexec key disables the launcher (bare binary)."""
    monkeypatch.setenv("VASP_PATH", "/vasp")
    adapter = _adapter()
    script = adapter._direct_vasp_script({"mpiexec": ""})
    assert 'export VASP_COMMAND="/vasp/bin/vasp_ncl"' in script


# --- _mpiexec_command -------------------------------------------------------


def test_mpiexec_command_explicit(monkeypatch) -> None:
    """An explicit mpiexec key is used verbatim, including empty."""
    assert _adapter()._mpiexec_command({"mpiexec": "srun"}) == "srun"
    assert _adapter()._mpiexec_command({"mpiexec": ""}) == ""


def test_mpiexec_command_default() -> None:
    """Without an explicit key, mpiexec is always returned."""
    assert _adapter()._mpiexec_command({}) == "mpiexec"


# --- _incar_py_script -------------------------------------------------------


def test_incar_py_script_string(monkeypatch) -> None:
    """The INCAR.py script wraps the python heredoc and exports VASP_COMMAND."""
    adapter = _adapter()
    script = adapter._incar_py_script(Path("."))
    assert script.startswith('export VASP_COMMAND="gorun vasp --local')
    assert "python <<EOF" in script
    assert "$(cat INCAR.py)" in script
    assert "Path(\"UNCONVERGED\").touch()" in script


def test_generate_run_commands_prefers_incar_py(monkeypatch, tmp_path: Path) -> None:
    """With INCAR.py present, the ASE wrapper path is used."""
    (tmp_path / "INCAR.py").touch()
    adapter = _adapter()
    script = adapter.generate_run_commands(tmp_path, {})
    assert "python <<EOF" in script


def test_generate_run_commands_direct_without_incar_py(monkeypatch, tmp_path: Path) -> None:
    """Without INCAR.py the direct VASP path is used."""
    monkeypatch.setenv("VASP_PATH", "/vasp")
    adapter = _adapter()
    script = adapter.generate_run_commands(tmp_path, {})
    assert "export VASP_COMMAND=" in script


# --- prepare_inputs ---------------------------------------------------------


def test_prepare_inputs_potcar_keep_skip(
    monkeypatch, vasp_input_dir: Path
) -> None:
    """--keep POTCAR does not regenerate an existing POTCAR."""
    (vasp_input_dir / "POTCAR").write_text("placeholder")
    regenerate = create_autospec(vasp_mod._regenerate_potcar, return_value=None)
    monkeypatch.setattr(vasp_mod, "_regenerate_potcar", regenerate)

    _adapter().prepare_inputs(vasp_input_dir, keep={"POTCAR"})
    regenerate.assert_not_called()


def test_prepare_inputs_potcar_missing_regenerates(
    monkeypatch, vasp_input_dir: Path
) -> None:
    """A missing POTCAR triggers regeneration via _regenerate_potcar."""
    regenerate = create_autospec(vasp_mod._regenerate_potcar, return_value=None)
    monkeypatch.setattr(vasp_mod, "_regenerate_potcar", regenerate)

    _adapter().prepare_inputs(vasp_input_dir, keep=set())
    regenerate.assert_called_once_with(str(vasp_input_dir))


# --- _rotate_incar_chain: converged with no chain ---------------------------


def test_rotate_incar_chain_converged_no_chains_noop(
    monkeypatch, tmp_path: Path
) -> None:
    """Converged but with no INCAR.N files means nothing to rotate."""
    (tmp_path / "INCAR").write_text("old")
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: True)
    _rotate_incar_chain(tmp_path)
    assert (tmp_path / "INCAR").read_text() == "old"
    assert not (tmp_path / "INCAR.old").exists()


# --- _modify_incar: no change ----------------------------------------------


def test_modify_incar_unchanged(tmp_path: Path, capsys) -> None:
    """Setting a tag to its existing value leaves INCAR unchanged."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\n")
    _modify_incar(tmp_path, {"ENCUT": "400"})
    assert "Leaving INCAR unchanged" in capsys.readouterr().out


# --- _regenerate_potcar -----------------------------------------------------


def test_regenerate_potcar_keep_existing_false(
    monkeypatch, tmp_path: Path
) -> None:
    """_regenerate_potcar passes keep_existing=False explicitly."""
    generate = create_autospec(vasp_mod.generate_potcar, return_value=None)
    monkeypatch.setattr(vasp_mod, "generate_potcar", generate)
    vasp_mod._regenerate_potcar(str(tmp_path))
    generate.assert_called_once_with(str(tmp_path), keep_existing=False)


# --- prepare_inputs: keep=None, force, incar_mods, NEB ---------------------


def test_prepare_inputs_keep_none(monkeypatch, tmp_path: Path) -> None:
    """keep=None is normalised to an empty set before _prepare_single."""
    (tmp_path / "INCAR").touch()
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: False)
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: False)
    prepared = []
    monkeypatch.setattr(
        vasp_mod.VaspAdapter,
        "_prepare_single",
        lambda self, path, keep: prepared.append(keep),
    )

    _adapter().prepare_inputs(tmp_path, keep=None)
    assert prepared == [set()]


def test_prepare_inputs_force_skips_incar_chain(
    monkeypatch, tmp_path: Path
) -> None:
    """force=True skips INCAR chain rotation."""
    (tmp_path / "INCAR").touch()
    rotate = create_autospec(vasp_mod._rotate_incar_chain, return_value=None)
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: False)
    monkeypatch.setattr(vasp_mod, "_rotate_incar_chain", rotate)
    monkeypatch.setattr(
        vasp_mod.VaspAdapter, "_prepare_single", lambda self, path, keep: None
    )

    _adapter().prepare_inputs(tmp_path, force=True)
    rotate.assert_not_called()


def test_prepare_inputs_applies_incar_mods_before_prepare(
    monkeypatch, tmp_path: Path
) -> None:
    """incar_mods run through _modify_incar before _prepare_single."""
    (tmp_path / "INCAR").touch()
    mods = {"ALGO": "Fast"}
    calls = []
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: False)
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: False)
    monkeypatch.setattr(
        vasp_mod,
        "_modify_incar",
        lambda path, m: calls.append(("modify", path, m)),
    )
    monkeypatch.setattr(
        vasp_mod.VaspAdapter,
        "_prepare_single",
        lambda self, path, keep: calls.append(("prepare", path)),
    )

    _adapter(incar_mods=mods).prepare_inputs(tmp_path)
    assert calls == [
        ("modify", tmp_path, mods),
        ("prepare", tmp_path),
    ]


@pytest.mark.parametrize(
    "no_clean, expected_cleared",
    [(False, ["0", "1"]), (True, [])],
    ids=["clean", "no_clean"],
)
def test_prepare_inputs_neb_subdirs(
    monkeypatch, tmp_path: Path, no_clean: bool, expected_cleared: list[str]
) -> None:
    """NEB runs prepare numbered subdirs; --no-clean skips their logs."""
    (tmp_path / "INCAR").touch()
    (tmp_path / "0").mkdir()
    (tmp_path / "1").mkdir()
    prepared = []
    cleared = []
    monkeypatch.setattr(vasp_mod, "mdp", lambda path: False)
    monkeypatch.setattr(vasp_mod, "directory_converged_p", lambda path: False)
    monkeypatch.setattr(vasp_mod, "nebp", lambda path: True)
    monkeypatch.setattr(
        vasp_mod.VaspAdapter,
        "_prepare_single",
        lambda self, path, keep: prepared.append(path),
    )
    monkeypatch.setattr(
        vasp_mod,
        "clear_slurm_logs",
        lambda path, extra_logs=None: cleared.append(path),
    )

    _adapter(no_clean=no_clean).prepare_inputs(tmp_path)
    assert prepared == [tmp_path, tmp_path / "0", tmp_path / "1"]
    assert cleared == [str(tmp_path / sub) for sub in expected_cleared]


# --- _prepare_single: POSCAR keep branches ----------------------------------


@pytest.mark.parametrize("poscar_exists", [True, False], ids=["exists", "missing"])
def test_prepare_single_keep_poscar(
    monkeypatch, vasp_input_dir: Path, capsys, poscar_exists: bool
) -> None:
    """--keep POSCAR keeps an existing POSCAR or copies CONTCAR when absent."""
    if not poscar_exists:
        (vasp_input_dir / "POSCAR").unlink()

    monkeypatch.setattr(
        vasp_mod, "maybe_regenerate", lambda *args, **kwargs: None
    )
    copy_contcar = create_autospec(vasp_mod.contcar_to_poscar, return_value=None)
    monkeypatch.setattr(vasp_mod, "contcar_to_poscar", copy_contcar)

    _adapter()._prepare_single(vasp_input_dir, keep={"POSCAR"})
    out = capsys.readouterr().out
    if poscar_exists:
        assert "Keeping existing POSCAR" in out
        copy_contcar.assert_not_called()
    else:
        assert "POSCAR does not exist" in out
        copy_contcar.assert_called_once_with(str(vasp_input_dir))
