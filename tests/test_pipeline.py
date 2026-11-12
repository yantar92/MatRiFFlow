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


"""Tests for matrifflow.gorun.core.pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import matrifflow.gorun.core.pipeline as pipeline_mod

from matrifflow.gorun.core.pipeline import (
    _backup_current_dir,
    _last_run_number,
    _next_run_folder,
    _rsync_exclude_args,
    dispatch_run,
)


def _args(**overrides) -> argparse.Namespace:
    """Build a dispatch args Namespace with all attributes present."""
    ns = argparse.Namespace(
        force=False,
        local=False,
        mark=False,
        max_slurm_jobs=0,
        keep=None,
        queue=None,
        number_of_nodes=None,
        time_limit=None,
        config=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _setup_pipeline(monkeypatch, config=None, server="testserver"):
    """Mock every subprocess/config boundary used by dispatch_run."""
    if config is None:
        config = {
            "testserver": {
                "shebang": "#!/usr/bin/env bash",
                "queues": ["cpu"],
                "stub": {},
            }
        }
    monkeypatch.setattr(pipeline_mod, "load_config", lambda args: config)
    monkeypatch.setattr(pipeline_mod, "resolve_server", lambda config: server)
    monkeypatch.setattr(pipeline_mod, "directory_queued_p", lambda path: False)
    monkeypatch.setattr(pipeline_mod, "user_job_count", lambda: 0)
    monkeypatch.setattr(
        pipeline_mod, "clear_slurm_logs", lambda path, extra_logs=None: None
    )
    monkeypatch.setattr(
        pipeline_mod, "get_sbatch_args", lambda *a, **k: {"partition": "cpu"}
    )
    monkeypatch.setattr(
        pipeline_mod, "get_best_script", lambda *a, **k: "FULL_SCRIPT"
    )
    return config


# --- _rsync_exclude_args ----------------------------------------------------


def test_rsync_exclude_args_default() -> None:
    """With no adapter excludes, only gorun_* is excluded."""
    assert _rsync_exclude_args([]) == "--exclude 'gorun_*'"


def test_rsync_exclude_args_with_adapter() -> None:
    """Adapter-specific excludes are appended after gorun_*."""
    assert (
        _rsync_exclude_args(["WAVECAR"])
        == "--exclude 'gorun_*' --exclude 'WAVECAR'"
    )


# --- _last_run_number / _next_run_folder ------------------------------------


def test_last_run_number_empty(tmp_path: Path) -> None:
    """No gorun_N folders yields None."""
    assert _last_run_number(tmp_path) is None


def test_last_run_number_max(tmp_path: Path) -> None:
    """The highest numbered folder is returned."""
    (tmp_path / "gorun_1").mkdir()
    (tmp_path / "gorun_3").mkdir()
    assert _last_run_number(tmp_path) == 3


def test_last_run_number_counts_tarball(tmp_path: Path) -> None:
    """A gorun_N.tar.gz also contributes its number."""
    (tmp_path / "gorun_2.tar.gz").touch()
    assert _last_run_number(tmp_path) == 2


def test_next_run_folder_empty(tmp_path: Path) -> None:
    """The first folder is gorun_1."""
    assert _next_run_folder(tmp_path) == "gorun_1"


def test_next_run_folder_after_existing(tmp_path: Path) -> None:
    """The next folder after gorun_3 is gorun_4."""
    (tmp_path / "gorun_3").mkdir()
    assert _next_run_folder(tmp_path) == "gorun_4"


# --- _backup_current_dir ----------------------------------------------------


def test_backup_current_dir(monkeypatch, tmp_path: Path) -> None:
    """Backup compresses the previous dir, drops gorun_ready, and rsyncs."""
    (tmp_path / "gorun_1").mkdir()
    (tmp_path / "gorun_2").mkdir()
    (tmp_path / "gorun_ready").touch()

    run = create_autospec(pipeline_mod.subprocess.check_call, return_value=0)
    monkeypatch.setattr(pipeline_mod, "barf_if_no_cmd", lambda cmd: None)
    monkeypatch.setattr(pipeline_mod.subprocess, "check_call", run)

    _backup_current_dir(tmp_path, excludes=["WAVECAR"])

    run.assert_called_once()
    command = run.call_args.args[0]
    assert "--exclude 'gorun_*' --exclude 'WAVECAR'" in command
    assert "gorun_3" in command
    assert not (tmp_path / "gorun_ready").exists()
    assert (tmp_path / "gorun_2.tar.gz").exists()
    assert not (tmp_path / "gorun_2").exists()


def test_backup_current_dir_does_not_compress_first(
    monkeypatch, tmp_path: Path
) -> None:
    """gorun_1 (the first backup) is never compressed to a tarball."""
    (tmp_path / "gorun_1").mkdir()
    monkeypatch.setattr(pipeline_mod, "barf_if_no_cmd", lambda cmd: None)
    monkeypatch.setattr(
        pipeline_mod.subprocess, "check_call", lambda cmd, shell=True: None
    )

    _backup_current_dir(tmp_path, excludes=[])

    assert (tmp_path / "gorun_1").is_dir()
    assert not (tmp_path / "gorun_1.tar.gz").exists()


# --- dispatch_run guards ----------------------------------------------------


def test_dispatch_run_running_guard(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A RUNNING file exits 0 before any other work."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    (tmp_path / "RUNNING").touch()
    assert dispatch_run(_args(), stub_adapter) == 0


def test_dispatch_run_queued_guard(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A queued directory exits 0."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "directory_queued_p", lambda path: True)
    assert dispatch_run(_args(), stub_adapter) == 0


def test_dispatch_run_gorun_ready_guard(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A gorun_ready file exits 0."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    (tmp_path / "gorun_ready").touch()
    assert dispatch_run(_args(), stub_adapter) == 0


def test_dispatch_run_no_valid_input(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """No valid input exits 1."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    stub_adapter.is_valid_input.return_value = False
    assert dispatch_run(_args(), stub_adapter) == 1


def test_dispatch_run_converged_guard(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A converged run exits 1 when not forced."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    stub_adapter.is_converged.return_value = True
    assert dispatch_run(_args(), stub_adapter) == 1


def test_dispatch_run_unknown_server(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """An unknown server exits 1."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch, server=None)
    assert dispatch_run(_args(), stub_adapter) == 1


def test_dispatch_run_no_config_exits(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """Missing config and not --local exits 1 before resolving a server."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline_mod, "load_config", lambda args: None)
    assert dispatch_run(_args(), stub_adapter) == 1


def test_dispatch_run_no_queues(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """A server with no queues configured exits 1."""
    monkeypatch.chdir(tmp_path)
    config = {"testserver": {"shebang": "#!/usr/bin/env bash", "stub": {}}}
    _setup_pipeline(monkeypatch, config=config)
    assert dispatch_run(_args(), stub_adapter) == 1


# --- dispatch_run mark path -------------------------------------------------


def test_dispatch_run_mark_path(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """The --mark path writes sub and creates gorun_ready."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    assert dispatch_run(_args(mark=True), stub_adapter) == 0
    assert (tmp_path / "sub").read_text() == "FULL_SCRIPT"
    assert (tmp_path / "gorun_ready").exists()


def test_dispatch_run_throttle_loop(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """max_slurm_jobs throttles until the running job count drops below it."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    counts = iter([3, 0])
    monkeypatch.setattr(pipeline_mod, "user_job_count", lambda: next(counts))
    slept: list[int] = []
    monkeypatch.setattr(pipeline_mod.time, "sleep", slept.append)

    assert dispatch_run(_args(max_slurm_jobs=2, mark=True), stub_adapter) == 0
    assert slept == [10]


def test_dispatch_run_skip_setup_env(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """GORUN_SKIP_SETUP=1 skips the adapter setup_commands call."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    monkeypatch.setenv("GORUN_SKIP_SETUP", "1")

    assert dispatch_run(_args(mark=True), stub_adapter) == 0
    stub_adapter.setup_commands.assert_not_called()


def test_dispatch_run_sbatch_submit_success(
    monkeypatch, tmp_path: Path, stub_adapter
) -> None:
    """The non-mark, non-local path submits via sbatch and returns 0."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    monkeypatch.setattr(
        pipeline_mod.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0),
    )

    assert dispatch_run(_args(), stub_adapter) == 0
    assert (tmp_path / "sub").read_text() == "FULL_SCRIPT"


def test_dispatch_run_sbatch_submit_failure(
    monkeypatch, tmp_path: Path, stub_adapter
) -> None:
    """A non-zero sbatch exit status is reported and returns 1."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    monkeypatch.setattr(
        pipeline_mod.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1),
    )

    assert dispatch_run(_args(), stub_adapter) == 1


def test_dispatch_run_force_bypasses_running(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """--force bypasses the RUNNING guard and reaches the mark path."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    (tmp_path / "RUNNING").touch()
    assert dispatch_run(_args(force=True, mark=True), stub_adapter) == 0
    assert (tmp_path / "gorun_ready").exists()


# --- dispatch_run local (config-free fallback) ------------------------------


def test_dispatch_run_local_no_config_falls_back(monkeypatch, tmp_path: Path, stub_adapter) -> None:
    """--local with no config file runs without resolving a server."""
    monkeypatch.chdir(tmp_path)
    load_config = create_autospec(pipeline_mod.load_config, return_value=None)
    resolve_server = create_autospec(pipeline_mod.resolve_server, return_value="testserver")
    monkeypatch.setattr(pipeline_mod, "load_config", load_config)
    monkeypatch.setattr(pipeline_mod, "resolve_server", resolve_server)
    monkeypatch.setattr(pipeline_mod, "directory_queued_p", lambda path: False)
    monkeypatch.setattr(pipeline_mod, "user_job_count", lambda: 0)
    monkeypatch.setattr(
        pipeline_mod, "clear_slurm_logs", lambda path, extra_logs=None: None
    )
    monkeypatch.setattr(
        pipeline_mod, "get_sbatch_args", lambda *a, **k: {"partition": "cpu"}
    )
    monkeypatch.setattr(
        pipeline_mod, "get_best_script", lambda *a, **k: "FULL_SCRIPT"
    )

    assert dispatch_run(_args(local=True, mark=True), stub_adapter) == 0
    # Config is read, but with no result the server is never resolved.
    load_config.assert_called_once()
    resolve_server.assert_not_called()
    assert (tmp_path / "gorun_ready").exists()
    sub = (tmp_path / "sub").read_text()
    # No sbatch headers for a local run.
    assert "#SBATCH" not in sub
    assert "#!/usr/bin/env bash" in sub


def test_dispatch_run_local_honors_matching_server_section(
    monkeypatch, tmp_path: Path, stub_adapter
) -> None:
    """--local with a resolvable config uses its server section."""
    monkeypatch.chdir(tmp_path)
    config = {
        "testserver": {"shebang": "#!/bin/bash -l", "mpiexec": "srun", "stub": {}}
    }
    _setup_pipeline(monkeypatch, config=config, server="testserver")
    assert dispatch_run(_args(local=True, mark=True), stub_adapter) == 0
    stub_adapter.setup_commands.assert_called_once_with(config["testserver"])


def test_dispatch_run_local_unmatched_server_falls_back(
    monkeypatch, tmp_path: Path, stub_adapter
) -> None:
    """--local with an unmatched host falls back to an empty section."""
    monkeypatch.chdir(tmp_path)
    config = {"some_other_server": {"shebang": "#!/bin/bash -l"}}
    monkeypatch.setattr(pipeline_mod, "load_config", lambda args: config)
    monkeypatch.setattr(pipeline_mod, "resolve_server", lambda config: None)
    monkeypatch.setattr(pipeline_mod, "directory_queued_p", lambda path: False)
    monkeypatch.setattr(pipeline_mod, "user_job_count", lambda: 0)
    monkeypatch.setattr(
        pipeline_mod, "clear_slurm_logs", lambda path, extra_logs=None: None
    )
    monkeypatch.setattr(
        pipeline_mod, "get_sbatch_args", lambda *a, **k: {"partition": "cpu"}
    )
    monkeypatch.setattr(
        pipeline_mod, "get_best_script", lambda *a, **k: "FULL_SCRIPT"
    )

    assert dispatch_run(_args(local=True, mark=True), stub_adapter) == 0
    stub_adapter.setup_commands.assert_called_once_with({})


def test_dispatch_run_unknown_server_exits_without_local(
    monkeypatch, tmp_path: Path, stub_adapter
) -> None:
    """An unresolvable server exits 1 when --local is not passed."""
    monkeypatch.chdir(tmp_path)
    config = {"some_other_server": {"shebang": "#!/bin/bash -l"}}
    monkeypatch.setattr(pipeline_mod, "load_config", lambda args: config)
    monkeypatch.setattr(pipeline_mod, "resolve_server", lambda config: None)
    monkeypatch.setattr(pipeline_mod, "directory_queued_p", lambda path: False)
    monkeypatch.setattr(pipeline_mod, "user_job_count", lambda: 0)
    monkeypatch.setattr(
        pipeline_mod, "clear_slurm_logs", lambda path, extra_logs=None: None
    )
    monkeypatch.setattr(
        pipeline_mod, "get_sbatch_args", lambda *a, **k: {"partition": "cpu"}
    )
    monkeypatch.setattr(
        pipeline_mod, "get_best_script", lambda *a, **k: "FULL_SCRIPT"
    )

    assert dispatch_run(_args(local=False, mark=True), stub_adapter) == 1


def test_dispatch_run_local_script_propagates_status(
    monkeypatch, tmp_path: Path, stub_adapter
) -> None:
    """The local run script exits with the computation's status, not 0."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    assert dispatch_run(_args(local=True, mark=True), stub_adapter) == 0
    sub = (tmp_path / "sub").read_text()
    assert "_gorun_status=$?" in sub
    assert "exit $_gorun_status" in sub


def test_dispatch_run_local_script_traps_signals(
    monkeypatch, tmp_path: Path, stub_adapter
) -> None:
    """The run script traps USR1/TERM and awaits a backgrounded body."""
    monkeypatch.chdir(tmp_path)
    _setup_pipeline(monkeypatch)
    assert dispatch_run(_args(local=True, mark=True), stub_adapter) == 0
    sub = (tmp_path / "sub").read_text()
    assert "trap '_signal_exit 138' USR1" in sub
    assert "trap '_signal_exit 143' TERM" in sub
    assert "} &" in sub
    assert "wait $_gorun_pid" in sub
