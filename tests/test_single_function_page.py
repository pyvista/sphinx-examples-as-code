"""Regression test for a real numpydoc "hoisted section" page structure.

Builds a real sphinx site (not a hand-built doctree) using the same
rubric-to-heading + section-hoisting setup pyvista's actual docs build uses
(see ``single_function_fixture/conf.py``), confirming
``_qualified_name_for`` names the download after the documented function
rather than falling back to a nonsensical ``<docname>-example-1`` -- the
exact bug seen in a real pyvista build for
``pyvista.examples.downloads.download_bunny``.

Complements the hand-built-doctree unit tests for ``_qualified_name_for``
in ``test_sphinx_examples_as_code.py``.
"""

from __future__ import annotations

from pathlib import Path

from test_tinypages import _run_sphinx_build
from test_tinypages import _sphinx_build_cmd

FIXTURE_DIR = Path(__file__).parent / 'single_function_fixture'


def test_download_named_after_the_function_not_the_docname(tmp_path: Path):
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(_sphinx_build_cmd(FIXTURE_DIR, html_dir, doctree_dir))
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    downloads_dir = html_dir / '_downloads'
    py_files = list(downloads_dir.rglob('*.py'))
    assert len(py_files) == 1, f'expected exactly one generated .py, got {py_files}'
    py_path = py_files[0]

    # not "index-example-1" (the docname-derived fallback name)
    assert py_path.stem == 'mymodule_download_bunny'

    lines = py_path.read_text(encoding='utf-8').splitlines()
    assert lines[0] == '# Examples from mymodule.download_bunny'
