"""Real-build regression tests for gallery sibling-section heading levels.

Builds a real sphinx-gallery site (not a hand-built doctree) covering:
one sibling-section heading nested under the page's own title (the common
case), one reusing the page title's own underline character (a real RST
heading, one level up from the first), and three genuinely nested levels
(page title, section, subsection). Complements the hand-built-doctree unit
tests for ``_heading_level``/``_title_underline_segment`` in
``test_sphinx_examples_as_code.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
from test_gallery_downloads import _is_ours
from test_tinypages import _run_sphinx_build
from test_tinypages import _sphinx_build_cmd

FIXTURE_DIR = Path(__file__).parent / 'heading_level_fixture'


@pytest.fixture(scope='module')
def built(tmp_path_factory) -> Path:
    # Copy the fixture rather than building from it in place: sphinx-gallery
    # writes its generated .rst files into the source tree itself as a side
    # effect of building (see test_gallery_downloads.py's own _build()).
    tmp_path = tmp_path_factory.mktemp('heading_level_build')
    src_dir = tmp_path / 'src'
    shutil.copytree(FIXTURE_DIR, src_dir)

    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(_sphinx_build_cmd(src_dir, html_dir, doctree_dir))
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'
    return html_dir


def _generated(html_dir: Path, name: str) -> Path:
    # gallery_downloads=True leaves sphinx-gallery's own same-named download
    # in place too (see test_gallery_downloads.py) -- _is_ours picks ours out.
    matches = [p for p in (html_dir / '_downloads').rglob(name) if _is_ours(p)]
    assert len(matches) == 1, f'expected exactly one generated {name}, got {matches}'
    return matches[0]


def test_reused_underline_char_renders_as_level_1(built: Path):
    py_path = _generated(built, 'plot_reused_level.py')
    lines = py_path.read_text(encoding='utf-8').splitlines()

    nested_idx = lines.index('# Nested Subsection')
    assert lines[nested_idx + 1] == '# ' + '-' * len('Nested Subsection')

    reused_idx = lines.index('# Reused Top Level')
    assert lines[reused_idx + 1] == '# ' + '=' * len('Reused Top Level')


def test_reused_underline_char_renders_as_level_1_in_ipynb(built: Path):
    nb_path = _generated(built, 'plot_reused_level.ipynb')
    notebook = json.loads(nb_path.read_text(encoding='utf-8'))
    all_lines = [line.rstrip() for cell in notebook['cells'] for line in cell['source']]
    assert '## Nested Subsection' in all_lines
    assert '# Reused Top Level' in all_lines


def test_three_genuine_levels_py_uses_tilde_for_the_third(built: Path):
    """A subsection nested two sections deep gets a ``~`` underline, not another dash line.

    Otherwise it would render identically to its own parent section's
    heading -- indistinguishable from a sibling instead of a subsection.
    """
    py_path = _generated(built, 'plot_three_levels.py')
    lines = py_path.read_text(encoding='utf-8').splitlines()

    section_idx = lines.index('# Section')
    assert lines[section_idx + 1] == '# ' + '-' * len('Section')

    subsection_idx = lines.index('# Subsection')
    assert lines[subsection_idx + 1] == '# ' + '~' * len('Subsection')


def test_three_genuine_levels_atx_renders_as_h3_in_ipynb(built: Path):
    """In .ipynb, every level uses ATX -- the third level is a plain ``###`` line."""
    nb_path = _generated(built, 'plot_three_levels.ipynb')
    notebook = json.loads(nb_path.read_text(encoding='utf-8'))
    all_source = ''.join(''.join(cell['source']) for cell in notebook['cells'])
    assert '### Subsection' in all_source
