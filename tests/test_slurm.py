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


"""Tests for matrifflow.gorun.slurm."""

from __future__ import annotations

import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

import matrifflow.gorun.slurm as slurm_mod

from matrifflow.gorun.slurm import (
    _executable_find,
    barf_if_no_cmd,
    clear_slurm_logs,
    directory_queued_p,
    get_best_script,
    sbatch_estimate_start,
    sbatch_script,
    user_job_count,
)


# --- sbatch_script ----------------------------------------------------------


def test_sbatch_script_empty_args() -> None:
    """With no args, the script carries only the fixed signal line."""
    assert sbatch_script("#!/usr/bin/env bash", {}, "echo hi") == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        "\n"
        "echo hi\n"
    )


def test_sbatch_script_single_arg() -> None:
    """A single sbatch arg is emitted with its key and quoted value."""
    assert sbatch_script(
        "#!/usr/bin/env bash", {"partition": "cpu"}, "echo hi"
    ) == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        '#SBATCH --partition="cpu"\n'
        "echo hi\n"
    )


def test_sbatch_script_multiple_args_preserve_order() -> None:
    """Args are emitted in dict insertion order, before the script."""
    assert sbatch_script(
        "#!/usr/bin/env bash",
        {"partition": "cpu", "time": "01:00:00"},
        "run.sh",
    ) == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --signal=B:USR1@300\n"
        '#SBATCH --partition="cpu"\n'
        '#SBATCH --time="01:00:00"\n'
        "run.sh\n"
    )


# --- _executable_find -------------------------------------------------------


def test_executable_find_present(monkeypatch) -> None:
    """A command found by shutil.which reports as present."""
    monkeypatch.setattr(slurm_mod.shutil, "which", lambda cmd: "/usr/bin/x")
    assert _executable_find("x") is True


def test_executable_find_absent(monkeypatch) -> None:
    """A command not found by shutil.which reports as absent."""
    monkeypatch.setattr(slurm_mod.shutil, "which", lambda cmd: None)
    assert _executable_find("x") is False


# --- barf_if_no_cmd / directory_queued_p / user_job_count -------------------


def test_barf_if_no_cmd_raises(monkeypatch) -> None:
    """A missing command raises FileNotFoundError."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: False)
    with pytest.raises(FileNotFoundError):
        barf_if_no_cmd("no-such-command")


def test_directory_queued_p_no_squeue(monkeypatch) -> None:
    """Without squeue, nothing is reported queued."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: False)
    assert directory_queued_p("/some/path") is False


def test_directory_queued_p_squeue(monkeypatch) -> None:
    """With squeue, queued-p reflects whether the path is in the output."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: True)
    monkeypatch.setattr(
        slurm_mod.subprocess,
        "check_output",
        lambda cmd, shell=True: b"/queued/path another\n",
    )
    assert directory_queued_p("/queued/path") is True
    assert directory_queued_p("/other/path") is False


def test_user_job_count_no_squeue(monkeypatch) -> None:
    """Without squeue, the job count is zero."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: False)
    assert user_job_count() == 0


def test_user_job_count_squeue(monkeypatch) -> None:
    """The squeue line count minus the header gives the job count."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: True)
    monkeypatch.setattr(
        slurm_mod.subprocess, "check_output", lambda cmd, shell=True: b"3\n"
    )
    assert user_job_count() == 2


# --- clear_slurm_logs -------------------------------------------------------


def test_clear_slurm_logs_removes_slurm_and_extra(tmp_path: Path) -> None:
    """slurm-*.out and listed extra logs are removed; unrelated files stay."""
    (tmp_path / "slurm-1.out").touch()
    (tmp_path / "slurm-2.out").touch()
    (tmp_path / "vasp.out").touch()
    (tmp_path / "keep.log").touch()

    clear_slurm_logs(str(tmp_path), extra_logs=["vasp.out"])

    assert not (tmp_path / "slurm-1.out").exists()
    assert not (tmp_path / "slurm-2.out").exists()
    assert not (tmp_path / "vasp.out").exists()
    assert (tmp_path / "keep.log").exists()


def test_clear_slurm_logs_no_extra(tmp_path: Path) -> None:
    """Without extra_logs only slurm-*.out files are removed."""
    (tmp_path / "slurm-1.out").touch()
    (tmp_path / "other.out").touch()

    clear_slurm_logs(str(tmp_path))

    assert not (tmp_path / "slurm-1.out").exists()
    assert (tmp_path / "other.out").exists()


def test_clear_slurm_logs_oserror_is_caught(tmp_path: Path) -> None:
    """A failed removal is reported, not raised.

    The matched name is a directory, so removal raises
    IsADirectoryError.
    """
    (tmp_path / "slurm-1.out").mkdir()
    clear_slurm_logs(str(tmp_path))


# --- sbatch_estimate_start --------------------------------------------------


def test_sbatch_estimate_start_success(monkeypatch, tmp_path: Path) -> None:
    """A successful --test-only output yields (wait_time, ncpus)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        if cmd.startswith("sbatch --test-only"):
            return (
                b"sbatch: Job 123 to start at 2026-08-25T14:00:00 "
                b"using 8 processors on nodes n001 in partition cpu"
            )
        if cmd.startswith("date"):
            return b"2026-08-25T13:00:00"
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    wait, cpus = sbatch_estimate_start("#!/bin/bash\necho hi\n")
    assert cpus == 8
    assert wait == timedelta(hours=1)


def test_sbatch_estimate_start_timezone_suffix(monkeypatch, tmp_path: Path) -> None:
    """A trailing token after the timestamp (e.g. timezone) is tolerated."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        if cmd.startswith("sbatch --test-only"):
            return (
                b"sbatch: Job 123 to start at 2026-09-30T13:14:31 a "
                b"using 8 processors on nodes nid005076 in partition small-g"
            )
        if cmd.startswith("date"):
            return b"2026-09-30T13:14:31"
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    wait, cpus = sbatch_estimate_start("#!/bin/bash\necho hi\n")
    assert cpus == 8
    assert wait == timedelta(0)


def test_sbatch_estimate_start_bad_account(monkeypatch, tmp_path: Path) -> None:
    """An invalid account error returns None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, output=b"Invalid account or account/partition combination specified"
        )

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    assert sbatch_estimate_start("x") is None


def test_sbatch_estimate_start_inside_node(monkeypatch, tmp_path: Path) -> None:
    """An inside-node access error yields a zero-wait, single-cpu estimate."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        if cmd.startswith("sbatch --test-only"):
            raise subprocess.CalledProcessError(
                1, cmd, output=b"allocation failure: Access/permission denied"
            )
        if cmd.startswith("date"):
            return b"2026-08-25T13:00:00"
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    wait, cpus = sbatch_estimate_start("x")
    assert cpus == 1
    assert wait == timedelta(0)


def test_sbatch_estimate_start_unavailable_node_config(monkeypatch, tmp_path: Path) -> None:
    """An unavailable node-config error returns None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, output=b"Requested node configuration is not available"
        )

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    assert sbatch_estimate_start("x") is None


def test_sbatch_estimate_start_unexpected_output_raises(monkeypatch, tmp_path: Path) -> None:
    """Unrecognized --test-only output raises RuntimeError instead of leaking it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        if cmd.startswith("sbatch --test-only"):
            return b"some unexpected output"
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    with pytest.raises(RuntimeError):
        sbatch_estimate_start("x")


def test_sbatch_estimate_start_other_error_reraises(
    monkeypatch, tmp_path: Path
) -> None:
    """An unrecognized CalledProcessError is re-raised after a diagnostic print."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "barf_if_no_cmd", lambda cmd: None)

    def fake(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output=b"some unrelated failure")

    monkeypatch.setattr(slurm_mod.subprocess, "check_output", fake)
    with pytest.raises(subprocess.CalledProcessError):
        sbatch_estimate_start("x")


# --- get_best_script --------------------------------------------------------


def test_get_best_script_picks_earliest_finish(monkeypatch) -> None:
    """The script whose estimated finish time is earliest wins."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: True)
    script = "echo hi"
    shebang = "#!/usr/bin/env bash"
    alt_args = [
        {"partition": "a", "time": "01:00:00"},
        {"partition": "b", "time": "00:30:00"},
    ]
    monkeypatch.setattr(
        slurm_mod,
        "sbatch_estimate_start",
        lambda s: {
            sbatch_script(shebang, alt_args[0], script): (timedelta(hours=1), 4),
            sbatch_script(shebang, alt_args[1], script): (timedelta(0), 8),
        }[s],
    )
    best = get_best_script(alt_args, script, shebang)
    assert best == sbatch_script(shebang, alt_args[1], script)


def test_get_best_script_all_none_raises(monkeypatch) -> None:
    """When every queue is unavailable, OSError is raised."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: True)
    monkeypatch.setattr(slurm_mod, "sbatch_estimate_start", lambda s: None)
    alt_args = [
        {"partition": "a", "time": "01:00:00"},
        {"partition": "b", "time": "00:30:00"},
    ]
    with pytest.raises(OSError):
        get_best_script(alt_args, "echo hi", "#!/usr/bin/env bash")


def test_get_best_script_no_sbatch_uses_first(monkeypatch) -> None:
    """Without sbatch, the first queue is used without estimating."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: False)
    alt_args = [
        {"partition": "a", "time": "01:00:00"},
        {"partition": "b", "time": "00:30:00"},
    ]
    script = "echo hi"
    shebang = "#!/usr/bin/env bash"
    best = get_best_script(alt_args, script, shebang)
    assert best == sbatch_script(shebang, alt_args[0], script)


def test_get_best_script_no_sbatch_single_queue(monkeypatch) -> None:
    """Without sbatch and a single queue, no warning is printed."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: False)
    alt_args = [{"partition": "a", "time": "01:00:00"}]
    script = "echo hi"
    shebang = "#!/usr/bin/env bash"
    best = get_best_script(alt_args, script, shebang)
    assert best == sbatch_script(shebang, alt_args[0], script)


def test_get_best_script_skips_none_estimates(monkeypatch) -> None:
    """Queues whose estimate is None are skipped in favour of a valid one."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: True)
    script = "echo hi"
    shebang = "#!/usr/bin/env bash"
    alt_args = [
        {"partition": "a", "time": "01:00:00"},
        {"partition": "b", "time": "00:30:00"},
    ]
    monkeypatch.setattr(
        slurm_mod,
        "sbatch_estimate_start",
        lambda s: {
            sbatch_script(shebang, alt_args[0], script): None,
            sbatch_script(shebang, alt_args[1], script): (timedelta(0), 8),
        }[s],
    )
    best = get_best_script(alt_args, script, shebang)
    assert best == sbatch_script(shebang, alt_args[1], script)


def test_get_best_script_later_candidate_does_not_win(monkeypatch) -> None:
    """A candidate finishing later than the current best is not selected."""
    monkeypatch.setattr(slurm_mod, "_executable_find", lambda cmd: True)
    script = "echo hi"
    shebang = "#!/usr/bin/env bash"
    alt_args = [
        {"partition": "a", "time": "00:30:00"},
        {"partition": "b", "time": "10:00:00"},
    ]
    monkeypatch.setattr(
        slurm_mod,
        "sbatch_estimate_start",
        lambda s: {
            sbatch_script(shebang, alt_args[0], script): (timedelta(0), 8),
            sbatch_script(shebang, alt_args[1], script): (timedelta(hours=1), 8),
        }[s],
    )
    best = get_best_script(alt_args, script, shebang)
    assert best == sbatch_script(shebang, alt_args[0], script)
