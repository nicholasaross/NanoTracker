"""Sanity tests: the package imports cleanly and exposes a version."""

from __future__ import annotations


def test_package_imports() -> None:
    import streettracker
    assert isinstance(streettracker.__version__, str)
    assert streettracker.__version__


def test_common_subpackage_imports() -> None:
    from streettracker.common import color, hourly, output, schema, summary  # noqa: F401


def test_cli_help_runs() -> None:
    from streettracker.cli import main
    assert main([]) == 0
    assert main(["--version"]) == 0
