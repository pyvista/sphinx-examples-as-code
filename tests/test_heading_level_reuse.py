"""Real-build regression test for gallery sibling-section heading levels.

Builds a real sphinx-gallery site (not a hand-built doctree) with two
sibling-section headings: one nested under the page's own title (the
common case), one reusing the page title's own underline character (a
real RST heading, one level up from the first). Complements the
hand-built-doctree unit tests for ``_heading_level`` in
``test_sphinx_examples_as_code.py``.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from test_tinypages import _run_sphinx_build
from test_tinypages import _sphinx_build_cmd

FIXTURE_DIR = Path(__file__).parent / 'heading_level_fixture'


def test_reused_underline_char_renders_as_level_1(tmp_path: Path):
    # Copy the fixture rather than building from it in place: sphinx-gallery
    # writes its generated .rst files into the source tree itself as a side
    # effect of building (see test_gallery_downloads.py's own _build()).
    src_dir = tmp_path / 'src'
    shutil.copytree(FIXTURE_DIR, src_dir)

    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(_sphinx_build_cmd(src_dir, html_dir, doctree_dir))
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    py_path = next((html_dir / '_downloads').rglob('plot_reused_level.py'))
    lines = py_path.read_text(encoding='utf-8').splitlines()

    nested_idx = lines.index('# Nested Subsection')
    assert lines[nested_idx + 1] == '# ' + '-' * len('Nested Subsection')

    reused_idx = lines.index('# Reused Top Level')
    assert lines[reused_idx + 1] == '# ' + '=' * len('Reused Top Level')
