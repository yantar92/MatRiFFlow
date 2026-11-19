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


"""Tests for matriflow.gorun.adapters.gpu."""

from __future__ import annotations

from pathlib import Path

from matrifflow.gorun.adapters.gpu import GpuAdapter, _wrap_command


def test_wrap_command_exact_output() -> None:
    """A simple command is wrapped in the heredoc template verbatim."""
    assert _wrap_command("echo hi") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "echo hi\n"
        "EOF"
    )


def test_wrap_command_multiline_command() -> None:
    """A multi-line command is placed between the setup line and EOF."""
    assert _wrap_command("line1\nline2") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "line1\nline2\n"
        "EOF"
    )


def test_wrap_command_empty_command() -> None:
    """An empty command still emits the wrapper and setup lines."""
    assert _wrap_command("") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "\n"
        "EOF"
    )


def _touch(path: Path, name: str) -> None:
    (path / name).write_text("", encoding="utf-8")


def test_no_previous_output_for_bare_runfile_py(tmp_path: Path) -> None:
    """A RUNFILE.py alone is input, not a previous run's artifact."""
    _touch(tmp_path, "RUNFILE.py")
    assert GpuAdapter().has_previous_output(tmp_path) is False


def test_no_previous_output_for_bare_runfile_sh(tmp_path: Path) -> None:
    """A RUNFILE.sh alone is input, not a previous run's artifact."""
    _touch(tmp_path, "RUNFILE.sh")
    assert GpuAdapter().has_previous_output(tmp_path) is False


def test_no_previous_output_when_both_runfiles_present(tmp_path: Path) -> None:
    """Both default input names are excluded from the output check."""
    _touch(tmp_path, "RUNFILE.sh")
    _touch(tmp_path, "RUNFILE.py")
    assert GpuAdapter().has_previous_output(tmp_path) is False


def test_previous_output_for_extra_file(tmp_path: Path) -> None:
    """A file next to the input script marks the directory as having output."""
    _touch(tmp_path, "RUNFILE.sh")
    _touch(tmp_path, "out.log")
    assert GpuAdapter().has_previous_output(tmp_path) is True


def test_previous_output_for_subdirectory(tmp_path: Path) -> None:
    """Directories count as previous output, not only files."""
    _touch(tmp_path, "RUNFILE.py")
    (tmp_path / "sub").mkdir()
    assert GpuAdapter().has_previous_output(tmp_path) is True


def test_no_previous_output_for_custom_script_only(tmp_path: Path) -> None:
    """With --script-name, the custom script is the only input name."""
    _touch(tmp_path, "train.py")
    assert GpuAdapter(script_name="train.py").has_previous_output(tmp_path) is False


def test_previous_output_for_unused_runfile_with_custom_script(
    tmp_path: Path,
) -> None:
    """A RUNFILE is an extra entry when --script-name selects another script."""
    _touch(tmp_path, "train.py")
    _touch(tmp_path, "RUNFILE.sh")
    assert GpuAdapter(script_name="train.py").has_previous_output(tmp_path) is True


def test_no_previous_output_for_empty_directory(tmp_path: Path) -> None:
    """An empty directory contains nothing from a previous run."""
    assert GpuAdapter().has_previous_output(tmp_path) is False
