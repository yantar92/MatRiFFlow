# MIT License
# Copyright (c) 2026 Inverse Materials Design Group
# See LICENSE in the project root for the full license text.

"""Compatibility and incomplete-output tests for the VASP output wrappers.

Synthetic cache state isolates MatRiFFlow's compatibility decisions from
pymatgen's parser. Disk fallbacks use small, plain or compressed log files.
"""

import os
from unittest.mock import create_autospec

import pytest
from monty.io import zopen

from matrifflow.pymatgen.io.vasp.outputs import Outcar, RunTimingStats, Vasplog


TIMING_LINES = [
    'Ionic step 1\n',
    'LOOP: cpu time 1.0: real time 2.0\n',
    'LOOP: cpu time 3.0: real time 4.0\n',
]


def _assert_timing_stats(stats: RunTimingStats) -> None:
    """Check wall-clock statistics for TIMING_LINES."""
    assert stats.scf.n == 2
    assert stats.scf.mean == pytest.approx(3.0)
    assert stats.scf.std == pytest.approx(2 ** 0.5)
    assert stats.ionic.n == 1
    assert stats.ionic.mean == pytest.approx(6.0)


@pytest.mark.parametrize('cache_kind', ['lines', 'text', 'empty_lines'])
def test_outcar_cached_data_needs_no_file(tmp_path, cache_kind) -> None:
    """Both cache formats work without disk access; even an empty cache wins."""
    outcar = Outcar.__new__(Outcar)
    outcar.filename = str(tmp_path / 'nonexistent_OUTCAR')
    if cache_kind == 'text':
        outcar._text = ''.join(TIMING_LINES)
        expected_lines = [line.rstrip('\n') for line in TIMING_LINES]
    else:
        outcar._lines = [] if cache_kind == 'empty_lines' else TIMING_LINES.copy()
        # _lines must take precedence over the older text cache.
        outcar._text = 'stale cached text'
        expected_lines = outcar._lines

    assert outcar._raw_log_lines() == expected_lines
    if cache_kind == 'empty_lines':
        assert outcar.timing_stats.scf.n == 0
        assert outcar.timing_stats.ionic.n == 0
    else:
        _assert_timing_stats(outcar.timing_stats)
    assert not outcar.file.exists()


@pytest.mark.parametrize('filename', ['OUTCAR', 'OUTCAR.gz'])
def test_outcar_without_cache_reads_disk(tmp_path, filename) -> None:
    """Legacy objects without caches read plain and compressed OUTCAR files."""
    path = tmp_path / filename
    with zopen(path, mode='wt', encoding='utf-8') as stream:
        stream.writelines(TIMING_LINES)
    outcar = Outcar.__new__(Outcar)
    outcar.filename = str(path)

    assert outcar._in_memory_lines() is None
    assert outcar._raw_log_lines() == TIMING_LINES
    _assert_timing_stats(outcar.timing_stats)
    assert outcar.file == path


@pytest.mark.parametrize('error_type', [IndexError, OSError, ValueError])
def test_outcar_final_forces_unreadable_table(monkeypatch, error_type) -> None:
    """Expected table-reading failures leave final forces unknown."""
    read_table = create_autospec(
        Outcar.read_table_pattern, side_effect=error_type('unreadable force table'))
    monkeypatch.setattr(Outcar, 'read_table_pattern', read_table)
    outcar = Outcar.__new__(Outcar)

    assert outcar.final_forces is None
    read_table.assert_called_once()
    assert read_table.call_args.kwargs['last_one_only'] is True


@pytest.mark.parametrize('table', [None, []])
def test_outcar_final_forces_absent_table(monkeypatch, table) -> None:
    """No parsed rows means unknown forces, not an empty force array."""
    read_table = create_autospec(Outcar.read_table_pattern, return_value=table)
    monkeypatch.setattr(Outcar, 'read_table_pattern', read_table)

    assert Outcar.__new__(Outcar).final_forces is None
    read_table.assert_called_once()


@pytest.mark.parametrize('path_kind', ['missing', 'file', 'empty_directory'])
def test_vasplog_from_dir_without_logs(tmp_path, path_kind) -> None:
    """Missing paths, regular files and empty directories produce no parsers."""
    path = tmp_path / 'source'
    if path_kind == 'file':
        path.write_text('not a directory\n')
    elif path_kind == 'empty_directory':
        path.mkdir()

    expected_files = [] if path_kind == 'empty_directory' else None
    assert Vasplog.vasp_log_files(path) == expected_files
    assert Vasplog.from_dir(path) == []


def test_vasplog_from_dir_orders_logs_by_mtime(tmp_path) -> None:
    """Directory parsing orders logs by age, ignoring OUTCAR and directories."""
    older = tmp_path / 'stdout'
    newer = tmp_path / 'slurm-1.out'
    older.write_text('BRMIX: very serious problems\n')
    newer.write_text('DAV: 1 -0.47E+03\n')
    (tmp_path / 'OUTCAR').write_text('ignored large output\n')
    (tmp_path / 'vasp.out').mkdir()
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    logs = Vasplog.from_dir(tmp_path)

    assert [log.file for log in logs] == [older, newer]
    assert logs[0].warnings.has('brmix')
    assert logs[1].progress.has('00SCF')
    assert logs[1].progress is logs[1].progress


def test_vasplog_oversized_file_warns_and_reads_prefix(tmp_path, monkeypatch) -> None:
    """Oversized files warn and leave records beyond the read limit unparsed."""
    first_line = 'BRMIX: very serious problems\n'
    path = tmp_path / 'stdout'
    path.write_text(first_line + 'padding\n' + 'ZBRENT: fatal error in bracketing\n')
    # readlines(sizehint) may consume the line crossing the limit. Place
    # padding after the limit to keep the final diagnostic outside it.
    monkeypatch.setattr(Vasplog, 'MAX_SIZE', len(first_line) - 1)

    with pytest.warns(ResourceWarning, match='too large.*Reading partially'):
        log = Vasplog(path)

    assert log.warnings.has('brmix')
    assert not log.warnings.has('zbrent')
    assert path.read_text().endswith('ZBRENT: fatal error in bracketing\n')
