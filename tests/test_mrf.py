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


"""Tests for matrifflow.pymatgen.cli.mrf (master CLI dispatch)."""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

import matrifflow.pymatgen.cli.mrf as mrf


def _patch_create_module(monkeypatch, fake_module) -> None:
    """Patch importlib so only mrf_create resolves to *fake_module*.

    main() imports the requested subcommand module lazily; every other
    import must keep resolving normally (for example
    pymatgen.ext.matproj on the error path).
    """
    real_import = importlib.import_module

    def fake_import(name):
        if name == 'matrifflow.pymatgen.cli.mrf_create':
            return fake_module
        return real_import(name)

    monkeypatch.setattr(importlib, 'import_module', fake_import)


## _showwarning


def test_showwarning_prints_category_and_message(capsys) -> None:
    """The coloured warning printer includes the category name."""
    mrf._showwarning('boom', UserWarning, 'f.py', 12)
    captured = capsys.readouterr()
    assert 'UserWarning:' in captured.err
    assert 'boom' in captured.err


def test_showwarning_honours_file_argument() -> None:
    """An explicit file redirects the printed warning to that file."""
    buf = io.StringIO()
    mrf._showwarning('msg', RuntimeWarning, 'f.py', 1, file=buf)
    assert 'RuntimeWarning:' in buf.getvalue()
    assert 'msg' in buf.getvalue()


## setup_logger


def test_setup_logger_verbosity_levels(monkeypatch) -> None:
    """Verbosity selects the log level and number of handlers."""
    configure = create_autospec(logging.basicConfig, return_value=None)
    monkeypatch.setattr(logging, 'basicConfig', configure)
    monkeypatch.setattr(logging, 'FileHandler', lambda *a, **k: object())
    monkeypatch.setattr(logging, 'StreamHandler', lambda *a, **k: object())

    mrf.setup_logger(argparse.Namespace(verbosity=0))
    mrf.setup_logger(argparse.Namespace(verbosity=1))
    mrf.setup_logger(argparse.Namespace(verbosity=2))

    assert configure.call_count == 3
    calls = [entry.kwargs for entry in configure.call_args_list]
    assert calls[0]['level'] == logging.INFO
    assert len(calls[0]['handlers']) == 1  # file only
    assert calls[1]['level'] == logging.INFO
    assert len(calls[1]['handlers']) == 2  # file + stdout
    assert calls[2]['level'] == logging.DEBUG
    assert len(calls[2]['handlers']) == 2


## main


def test_main_run_intercepts_to_gorun(monkeypatch) -> None:
    """``mrf run ...`` is forwarded to gorun before argparse runs."""
    from matrifflow.gorun import gorun

    main = create_autospec(gorun.main, return_value=0)
    monkeypatch.setattr(sys, 'argv', ['mrf', 'run', 'vasp', '--dir', 'x'])
    monkeypatch.setattr(gorun, 'main', main)
    assert mrf.main() == 0
    main.assert_called_once_with(['vasp', '--dir', 'x'], prog='mrf run')


def test_main_dispatches_to_subcommand(monkeypatch) -> None:
    """A subcommand module is imported and its func run, returning its code."""
    monkeypatch.setattr(sys, 'argv', ['mrf', 'create'])
    monkeypatch.setattr(mrf, 'setup_logger', lambda args: None)

    fake = SimpleNamespace(add_args=Mock(return_value=None), create=Mock(return_value=7))
    _patch_create_module(monkeypatch, fake)
    assert mrf.main() == 7
    fake.add_args.assert_called_once()
    fake.create.assert_called_once()


def test_main_value_error_is_reported(monkeypatch, capsys) -> None:
    """A ValueError from the subcommand is reported as a one-line error."""
    monkeypatch.setattr(sys, 'argv', ['mrf', 'create'])
    monkeypatch.setattr(mrf, 'setup_logger', lambda args: None)

    fake = SimpleNamespace(
        add_args=Mock(return_value=None), create=Mock(side_effect=ValueError('nope'))
    )
    _patch_create_module(monkeypatch, fake)
    assert mrf.main() == 1
    assert 'nope' in capsys.readouterr().err


def test_main_value_error_reraised_with_verbose(monkeypatch) -> None:
    """With -vv, a user error re-raises so the full traceback is shown."""
    monkeypatch.setattr(sys, 'argv', ['mrf', '-vv', 'create'])
    monkeypatch.setattr(mrf, 'setup_logger', lambda args: None)

    fake = SimpleNamespace(
        add_args=Mock(return_value=None), create=Mock(side_effect=ValueError('nope'))
    )
    _patch_create_module(monkeypatch, fake)
    with pytest.raises(ValueError, match='nope'):
        mrf.main()


def test_main_no_subcommand_exits(monkeypatch) -> None:
    """Running with no subcommand exits via argparse."""
    monkeypatch.setattr(sys, 'argv', ['mrf'])
    with pytest.raises(SystemExit) as exc:
        mrf.main()
    assert exc.value.code == 2
