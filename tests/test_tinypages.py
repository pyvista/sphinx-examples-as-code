"""Tests for the ``sphinx_examples_as_code`` Sphinx extension."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from subprocess import PIPE
from subprocess import Popen
import sys

import nbformat
import pytest

import sphinx_examples_as_code as seac

SRCDIR = Path(__file__).parent / 'tinypages'


def _sphinx_build_cmd(
    source_dir: Path,
    html_dir: Path,
    doctree_dir: Path | None = None,
    sphinx_args: tuple[str, ...] = (),
) -> list[str]:
    """Create a sphinx-build command."""
    cmd = [
        sys.executable,
        '-msphinx',
        '-W',
        '-b',
        'html',
        *sphinx_args,
    ]

    if doctree_dir is not None:
        cmd.extend(['-d', str(doctree_dir)])

    cmd.extend([str(source_dir), str(html_dir)])

    if sys.platform == 'win32':
        # matplotlib's plot directive fails to render images on Windows CI, resulting in Sphinx
        # warnings that are fatal errors under ``-W``. Seems like a genuine bug with matplotlib
        # and not PyVista, so we ignore the warnings.
        cmd += ['-D', 'suppress_warnings=image.not_readable,download.not_readable']

    return cmd


def _run_sphinx_build(cmd: list[str]) -> tuple[int, str, str]:
    """Run sphinx-build and return returncode, stdout, and stderr."""
    proc = Popen(
        cmd,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        env={**os.environ, 'MPLBACKEND': ''},
        encoding='utf8',
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


def _read(paths: list[Path], name_contains: str) -> str:
    """Find the one generated file whose name contains ``name_contains`` and return its text."""
    matches = [p for p in paths if name_contains in p.name]
    assert matches, f'No generated file matching {name_contains!r} in {[p.name for p in paths]}'
    assert len(matches) == 1, f'Expected exactly one match for {name_contains!r}, got {matches}'
    return matches[0].read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def built(tmp_path_factory) -> tuple[Path, list[Path]]:
    """Build the fixture once and share it (and its generated files) across this module."""
    tmp_path = tmp_path_factory.mktemp('sphinx_examples_as_code_build')
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'

    returncode, out, err = _run_sphinx_build(
        _sphinx_build_cmd(
            SRCDIR,
            html_dir,
            doctree_dir,
            # The default footer is exercised by its own dedicated tests
            # below; disabled here so the many unrelated "absent from the
            # whole file" assertions elsewhere in this module aren't
            # incidentally tripped by its (unrelated) text/backticks/URL.
            ('-D', 'sphinx_examples_as_code_conf.footer='),
        ),
    )
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    downloads_dir = html_dir / '_downloads'
    examples = sorted(downloads_dir.rglob('*.py')) if downloads_dir.is_dir() else []
    return html_dir, examples


@pytest.fixture(scope='module')
def built_notebooks(built: tuple[Path, list[Path]]) -> list[Path]:
    """Reuse the same build as ``built``, listing its generated .ipynb files instead."""
    downloads_dir = built[0] / '_downloads'
    return sorted(downloads_dir.rglob('*.ipynb')) if downloads_dir.is_dir() else []


def test_sphinx_examples_as_code_execute(built: tuple[Path, list[Path]]):
    """Every generated example script should run standalone without error.

    This is the key correctness check: a script that merely *looks*
    plausible would still fail here if, say, it were missing an import
    that got silently dropped during conversion.
    """
    _html_dir, examples = built
    assert examples, 'expected at least one generated example script'

    env = {**os.environ, 'PYVISTA_OFF_SCREEN': 'true'}
    failures = []
    for path in examples:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=path.parent,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f'{path.name}:\n{result.stdout}\n{result.stderr}')

    assert not failures, 'Some generated examples failed to execute:\n\n' + '\n\n'.join(failures)


def test_no_examples_section_produces_no_download(built: tuple[Path, list[Path]]):
    """A docstring with no "Examples" heading at all should be untouched."""
    _html_dir, examples = built
    assert not any('case_no_examples' in p.name for p in examples)


def test_prose_only_examples_produces_no_download(built: tuple[Path, list[Path]]):
    """An Examples section with no real code should not produce a download."""
    _html_dir, examples = built
    assert not any('case_prose_only' in p.name for p in examples)


def test_pyvista_plot_wrapped_examples_still_convert(built: tuple[Path, list[Path]]):
    """A pyvista-importing Examples section gets auto-wrapped in ``.. pyvista-plot::``.

    That directive wraps its generated source in a
    ``.. container:: pyvista-plot-source`` node (see plot_directive.py's
    TEMPLATE) -- this confirms the Examples-heading scan isn't thrown off
    by that extra container, still extracts the real code from inside it,
    and drops the rendered figure rather than turning it into a comment.
    """
    _html_dir, examples = built
    src = _read(examples, 'case_pyvista_plot_wrapped')
    assert 'import pyvista as pv' in src
    assert 'pv.Sphere().plot()' in src
    # nothing figure/image-related should have leaked in as a comment
    assert 'image' not in src.lower()
    assert '.png' not in src


def test_docstring_examples_conversion(built: tuple[Path, list[Path]]):
    """Spot-check the prose/code conversion rules on a few representative cases."""
    _html_dir, examples = built

    dropdown_src = _read(examples, 'case_dropdown')
    assert 'Click me' not in dropdown_src
    assert 'hidden content' not in dropdown_src.lower()
    assert 'import sys' in dropdown_src

    tabset_src = _read(examples, 'case_tabset')
    assert 'Static Scene' not in tabset_src
    assert 'Interactive Scene' not in tabset_src
    assert 'iframe' not in tabset_src

    note_src = _read(examples, 'case_note')
    assert '# NOTE:' in note_src
    assert '# This is a note' in note_src

    warning_src = _read(examples, 'case_warning')
    assert '# WARNING:' in warning_src

    multi_note_src = _read(examples, 'case_multi_paragraph_note')
    assert '# First paragraph of the note.' in multi_note_src
    assert '# Second paragraph of the note.' in multi_note_src

    admonition_src = _read(examples, 'case_generic_admonition')
    assert '# Custom Title:' in admonition_src

    xref_src = _read(examples, 'case_xref_plain')
    assert '`docstring_cases.Sample`' in xref_src
    assert '`docstring_cases.Sample.show()`' in xref_src

    xref_title_src = _read(examples, 'case_xref_explicit_title')
    assert '`Sample class`' in xref_title_src

    ref_src = _read(examples, 'case_ref_plain')
    # a plain :ref: keeps its resolved display text with no backticks
    assert '`' not in ref_src.split('import sys')[0]

    inline_literal_src = _read(examples, 'case_inline_literal')
    assert '`some_variable = True`' in inline_literal_src

    combined_src = _read(examples, 'case_combined')
    assert '`docstring_cases.Sample`' in combined_src
    assert '`some_variable = True`' in combined_src
    assert '# NOTE:' in combined_src
    assert 'More details' not in combined_src  # dropped dropdown
    assert '3' not in combined_src  # doctest output line dropped entirely, not commented


def test_no_doctest_output_included(built: tuple[Path, list[Path]]):
    """Doctest output lines should never appear anywhere -- only the input code."""
    src = _read(built[1], 'case_doctest_with_output')
    assert '6 * 7' in src
    assert "'hello ' + 'world'" in src
    # the expected results of running that code should not appear at all,
    # in any form (not as code, not as a comment)
    assert '42' not in src
    assert 'hello world' not in src


def test_download_uses_32_char_hash(built: tuple[Path, list[Path]]):
    """The digest directory under ``_downloads/`` should be a 32-character hash.

    Matches the digest length Sphinx's own native download-file handling
    uses (e.g. matplotlib's built-in plot-directive download links), rather
    than some other arbitrary truncation.
    """
    _html_dir, examples = built
    assert examples
    for path in examples:
        digest_dir = path.parent.name
        assert len(digest_dir) == 32, f'{path}: expected a 32-char digest dir, got {digest_dir!r}'
        assert all(c in '0123456789abcdef' for c in digest_dir)


def test_header_format(built: tuple[Path, list[Path]]):
    """Every generated file should start with a title header and matching underline."""
    src = _read(built[1], 'case_note')
    lines = src.splitlines()
    assert lines[0] == '# Examples from docstring_cases.case_note'
    title = lines[0].removeprefix('# ')
    assert lines[1] == '# ' + '-' * len(title)
    assert len(lines[0]) == len(lines[1])
    # blank line between the header and the rest of the content
    assert lines[2] == ''


def test_whitespace_conventions(built: tuple[Path, list[Path]]):
    """Check the spacing rules.

    text-directly-above-code, blank-after-code, directives get blank lines on both sides,
    and the file ends with a trailing blank line.
    """
    # case_combined: prose -> note (directive) -> code -> dropdown (dropped)
    combined_src = _read(built[1], 'case_combined')
    lines = combined_src.splitlines()

    # header (directive) is followed by a blank line
    assert lines[2] == ''
    # the note block (directive) has a blank line before AND after it
    note_idx = next(i for i, line in enumerate(lines) if line == '# NOTE:')
    assert lines[note_idx - 1] == ''
    note_end = next(
        i for i in range(note_idx + 1, len(lines)) if not lines[i].startswith('#') and lines[i]
    )
    assert lines[note_end - 1] == ''  # blank line right after the note, before code

    # a code block is always followed by a blank line
    code_idx = next(i for i, line in enumerate(lines) if line == 'import sys')
    # find where this run of code ends
    end_of_code = code_idx
    while end_of_code + 1 < len(lines) and lines[end_of_code + 1].strip():
        end_of_code += 1
    assert lines[end_of_code + 1] == ''

    # file ends with exactly one trailing blank line
    assert combined_src.endswith('\n\n')
    assert not combined_src.endswith('\n\n\n')

    # case_xref_plain: prose sits directly above its doctest code, no blank
    # line in between, since the source docstring has none either
    xref_src = _read(built[1], 'case_xref_plain')
    xref_lines = xref_src.splitlines()
    code_start = next(i for i, line in enumerate(xref_lines) if line == 'import sys')
    assert xref_lines[code_start - 1] != ''
    assert xref_lines[code_start - 1].startswith('#')


def test_seealso_admonition(built: tuple[Path, list[Path]]):
    """A ``.. seealso::`` block's separate paragraphs must not run together.

    Mirrors pyvista's dataset downloader docstrings, which follow their
    doctest with a ``.. seealso::`` linking to the Dataset Gallery.
    """
    src = _read(built[1], 'case_seealso')
    assert '# SEE ALSO:' in src
    assert '# Some Target' in src
    assert '# See this in the gallery for more info.' in src
    # each paragraph must be its own line -- not concatenated together
    assert 'Some Target\n# See' in src or 'Some TargetSee' not in src


def test_stray_markup_in_doctest_comment_cleaned(built: tuple[Path, list[Path]]):
    """RST written inside a doctest comment is never resolved by docutils.

    Since it's preformatted, that raw text would otherwise leak through
    verbatim (backticks, role prefixes and all) instead of being cleaned up
    the way the same syntax in ordinary prose already is.
    """
    hyperlink_src = _read(built[1], 'case_stray_hyperlink_in_doctest_comment')
    assert '# See Extension <https://example.com/ext>.' in hyperlink_src
    assert '# See some_target for more.' in hyperlink_src
    assert '`' not in hyperlink_src
    assert '`_' not in hyperlink_src

    xref_src = _read(built[1], 'case_stray_xref_in_doctest_comment')
    assert '# Uses cell_centers.' in xref_src
    assert '# Uses pyvista.read.' in xref_src
    assert ':func:' not in xref_src
    assert '`' not in xref_src


def test_link_position_config(tmp_path: Path):
    """``sphinx_examples_as_code_conf['link_position']`` moves the link within its section.

    Default is ``'top'`` (already covered by every other test in this
    module, where the link always precedes the code); this checks ``'bottom'``.
    """
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(
        _sphinx_build_cmd(
            SRCDIR,
            html_dir,
            doctree_dir,
            ('-D', 'sphinx_examples_as_code_conf.link_position=bottom'),
        ),
    )
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    html = (html_dir / 'docstring_cases.html').read_text(encoding='utf-8')
    section_start = html.find('id="docstring_cases.case_note"')
    assert section_start != -1
    link_pos = html.find('reference download', section_start)
    doctest_pos = html.find('highlight-default', section_start)
    note_pos = html.find('admonition note', section_start)
    assert link_pos != -1
    assert link_pos > doctest_pos
    assert link_pos > note_pos


# ---------------------------------------------------------------------------
# .ipynb notebook generation and the sphinx_examples_as_code_conf['formats'] option
# ---------------------------------------------------------------------------


def test_notebook_generated_alongside_py(
    built: tuple[Path, list[Path]], built_notebooks: list[Path]
):
    """By default, every Examples section gets both a .py and a .ipynb download."""
    py_names = {p.stem for p in built[1]}
    ipynb_names = {p.stem for p in built_notebooks}
    assert py_names
    assert py_names == ipynb_names


def test_notebook_and_py_links_placed_together(built: tuple[Path, list[Path]]):
    """The two download links render in the same paragraph, next to each other."""
    html = (built[0] / 'docstring_cases.html').read_text(encoding='utf-8')
    section_start = html.find('id="docstring_cases.case_note"')
    assert section_start != -1
    snippet = html[section_start : section_start + 800]
    py_pos = snippet.find('Download Python source code')
    ipynb_pos = snippet.find('Download Jupyter notebook')
    assert py_pos != -1
    assert ipynb_pos != -1
    between = snippet[py_pos:ipynb_pos]
    assert '</p>' not in between  # both links are in the same paragraph
    assert ' | ' in between


def test_notebook_is_valid_nbformat(built_notebooks: list[Path]):
    """Every generated .ipynb should be a well-formed, valid notebook."""
    assert built_notebooks
    for path in built_notebooks:
        node = nbformat.read(path, as_version=4)
        nbformat.validate(node)


def test_notebook_code_cells_match_py_code(
    built: tuple[Path, list[Path]], built_notebooks: list[Path]
):
    """A notebook's code cells, concatenated, should match the .py file's code lines.

    Confirms the cell-splitting doesn't drop or duplicate any code -- only
    reorganizes prose/directives into separate markdown cells around it.
    """
    py_src = _read(built[1], 'case_combined')
    py_code_lines = [line for line in py_src.splitlines() if line and not line.startswith('#')]

    notebook_path = next(p for p in built_notebooks if p.stem == 'docstring_cases_case_combined')
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    notebook_code_lines = [
        line.rstrip('\n') for cell in code_cells for line in cell['source'] if line.strip()
    ]

    assert notebook_code_lines == py_code_lines


@pytest.mark.parametrize(
    ('formats', 'expect_py', 'expect_ipynb'),
    [
        ('py', True, False),
        ('ipynb', False, True),
        ('py,ipynb', True, True),
    ],
)
def test_formats_config_selection(
    tmp_path: Path,
    formats: str,
    expect_py: bool,
    expect_ipynb: bool,
):
    """``sphinx_examples_as_code_conf['formats']`` controls which download(s) get generated."""
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(
        _sphinx_build_cmd(
            SRCDIR,
            html_dir,
            doctree_dir,
            ('-D', f'sphinx_examples_as_code_conf.formats={formats}'),
        ),
    )
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    downloads_dir = html_dir / '_downloads'
    py_files = list(downloads_dir.rglob('*.py'))
    ipynb_files = list(downloads_dir.rglob('*.ipynb'))

    assert bool(py_files) == expect_py
    assert bool(ipynb_files) == expect_ipynb

    html = (html_dir / 'docstring_cases.html').read_text(encoding='utf-8')
    assert ('Download Python source code' in html) == expect_py
    assert ('Download Jupyter notebook' in html) == expect_ipynb


# ---------------------------------------------------------------------------
# html_baseurl and See Also link resolution
# ---------------------------------------------------------------------------


def test_base_url_unset_no_links_anywhere(
    built: tuple[Path, list[Path]], built_notebooks: list[Path]
):
    """Default (``html_baseurl`` unset): no links anywhere, in either format -- the status quo."""
    for name in (
        'case_regular_xref_for_base_url',
        'case_see_also_directive_for_base_url',
        'case_see_also_bare_rubric_for_base_url',
        'case_see_also_underline_heading_for_base_url',
    ):
        py_src = _read(built[1], name)
        assert 'http' not in py_src

        nb_path = next(p for p in built_notebooks if p.stem == f'docstring_cases_{name}')
        nb_text = nb_path.read_text(encoding='utf-8')
        assert 'http' not in nb_text
        assert '](' not in nb_text  # no markdown links


def test_base_url_set_resolves_links(tmp_path: Path):
    # With html_baseurl configured (the standard Sphinx setting, also used
    # for canonical links/sitemap generation -- not a setting of this
    # extension's own), See Also parts (in any form) get literal URL text in
    # .py and clickable markdown links in .ipynb; regular refs only get the
    # link in .ipynb, and .py stays link-free outside a See Also part.
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(
        _sphinx_build_cmd(
            SRCDIR,
            html_dir,
            doctree_dir,
            (
                '-D',
                'html_baseurl=https://docs.example.com/',
                # disabled -- this test's "no http anywhere" checks would
                # otherwise trip on the (unrelated) default footer's URL
                '-D',
                'sphinx_examples_as_code_conf.footer=',
            ),
        ),
    )
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    downloads_dir = html_dir / '_downloads'
    py_files = list(downloads_dir.rglob('*.py'))
    ipynb_files = list(downloads_dir.rglob('*.ipynb'))

    # a regular (non-See-Also) reference: link resolved in .ipynb, omitted in .py
    regular_py = _read(py_files, 'case_regular_xref_for_base_url')
    assert 'http' not in regular_py
    assert '`docstring_cases.Sample`' in regular_py

    regular_nb = next(
        p for p in ipynb_files if p.stem == 'docstring_cases_case_regular_xref_for_base_url'
    ).read_text(encoding='utf-8')
    assert '[`docstring_cases.Sample`](https://docs.example.com/' in regular_nb

    # all three See Also forms should behave identically
    for name in (
        'case_see_also_directive_for_base_url',
        'case_see_also_bare_rubric_for_base_url',
        'case_see_also_underline_heading_for_base_url',
    ):
        py_src = _read(py_files, name)
        assert '# SEE ALSO:' in py_src
        assert '# docstring_cases.Sample https://docs.example.com/' in py_src
        assert '`docstring_cases.Sample`' not in py_src  # no backticks in See Also url lines

        nb_path = next(p for p in ipynb_files if p.stem == f'docstring_cases_{name}')
        nb_text = nb_path.read_text(encoding='utf-8')
        assert '[`docstring_cases.Sample`](https://docs.example.com/' in nb_text


def test_markdown_cells_use_hard_line_breaks(built_notebooks: list[Path]):
    """Adjacent lines within one markdown cell need an explicit hard break.

    Without it, markdown treats a single newline as whitespace and runs
    everything in the cell together into one paragraph.
    """
    nb_path = next(p for p in built_notebooks if p.stem == 'docstring_cases_case_note')
    notebook = json.loads(nb_path.read_text(encoding='utf-8'))
    header_cell = next(c for c in notebook['cells'] if c['cell_type'] == 'markdown')
    assert header_cell['source'][0].endswith('  \n')


def test_html_baseurl_missing_scheme_disables_links_without_failing_build(tmp_path: Path):
    """A malformed ``html_baseurl`` (no scheme) degrades to "no links", not a build failure.

    Unlike this extension's own config keys, ``html_baseurl`` isn't
    something it owns or validates -- it's a standard Sphinx setting also
    used for canonical links/sitemap generation, which Sphinx itself
    doesn't validate the format of either. Failing the build over it would
    be presumptuous of a value that might be set (however unusually) for
    reasons of its own.
    """
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(
        _sphinx_build_cmd(
            SRCDIR,
            html_dir,
            doctree_dir,
            (
                '-D',
                'html_baseurl=docs.example.com/',
                # disabled -- its own (unrelated) link would trip the "no
                # markdown links" check below
                '-D',
                'sphinx_examples_as_code_conf.footer=',
            ),
        ),
    )
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    nb_path = next(
        (html_dir / '_downloads').rglob('docstring_cases_case_regular_xref_for_base_url.ipynb')
    )
    nb_text = nb_path.read_text(encoding='utf-8')
    assert '](' not in nb_text  # no markdown links -- same as html_baseurl unset entirely


def test_base_url_missing_trailing_slash_still_resolves_subpath(tmp_path: Path):
    """An ``html_baseurl`` with a subpath but no trailing slash should be normalized.

    Rather than silently dropping the last path segment.
    """
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(
        _sphinx_build_cmd(
            SRCDIR,
            html_dir,
            doctree_dir,
            ('-D', 'html_baseurl=https://docs.example.com/en/stable'),
        ),
    )
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    nb_path = next(
        (html_dir / '_downloads').rglob('docstring_cases_case_regular_xref_for_base_url.ipynb')
    )
    nb_text = nb_path.read_text(encoding='utf-8')
    assert 'https://docs.example.com/en/stable/docstring_cases.html' in nb_text


# ---------------------------------------------------------------------------
# sphinx_examples_as_code_conf['footer']
# ---------------------------------------------------------------------------

_DEFAULT_FOOTER_PY_LINES = (
    '# Generated by sphinx-examples-as-code https://github.com/pyvista/sphinx-examples-as-code',
)
_DEFAULT_FOOTER_IPYNB_LINK = (
    '[sphinx-examples-as-code](https://github.com/pyvista/sphinx-examples-as-code)'
)


def test_default_footer_appears_in_py(tmp_path: Path):
    """With nothing overridden, the default footer is appended -- this is a real default."""
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(_sphinx_build_cmd(SRCDIR, html_dir, doctree_dir))
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    py_path = next((html_dir / '_downloads').rglob('*case_note*.py'))
    lines = py_path.read_text(encoding='utf-8').splitlines()
    sep_line = f'# {seac._FOOTER_SEPARATOR}'
    sep_idx = lines.index(sep_line)
    assert lines[sep_idx - 1] == ''  # blank line before the footer's separator
    assert lines[sep_idx + 1 : sep_idx + 1 + len(_DEFAULT_FOOTER_PY_LINES)] == list(
        _DEFAULT_FOOTER_PY_LINES
    )


def test_default_footer_appears_in_ipynb_with_clickable_links(tmp_path: Path):
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(_sphinx_build_cmd(SRCDIR, html_dir, doctree_dir))
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    nb_path = next((html_dir / '_downloads').rglob('*case_note*.ipynb'))
    notebook = json.loads(nb_path.read_text(encoding='utf-8'))
    all_source = ''.join(''.join(cell['source']) for cell in notebook['cells'])
    assert _DEFAULT_FOOTER_IPYNB_LINK in all_source
    # the repo name reads as a real link, not inline code with no link at all
    assert '`sphinx-examples-as-code`' not in all_source
    # .py's plain "text url" form (no markdown link syntax) shouldn't appear here
    assert 'Generated by sphinx-examples-as-code https://' not in all_source


def test_default_footer_is_its_own_dedicated_ipynb_cell(tmp_path: Path):
    """The footer always gets its own markdown cell, never sharing one.

    ``case_note``'s Examples section ends with a ``.. note::`` -- markdown
    content directly preceding the footer, with no code in between -- which
    would otherwise land in the same cell (``_segments_to_cells`` only splits
    on code/markdown transitions). Confirms it's split out anyway; see
    ``test_gallery_downloads.py`` for the case where the footer already
    follows a code cell, so it would land in its own cell either way.

    Also confirms the separator renders as a real ``<hr>``, not more text:
    confirmed against an actual CommonMark parser (Python-Markdown) during
    development, not just assumed from the spec -- a lone line of 3+ dashes,
    preceded by a blank line, is a thematic break ('=' would not be, only
    valid for setext heading underlines, not thematic breaks).
    """
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(_sphinx_build_cmd(SRCDIR, html_dir, doctree_dir))
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    nb_path = next((html_dir / '_downloads').rglob('*case_note*.ipynb'))
    notebook = json.loads(nb_path.read_text(encoding='utf-8'))
    footer_cell = next(
        c for c in notebook['cells'] if _DEFAULT_FOOTER_IPYNB_LINK in ''.join(c['source'])
    )
    lines = [line.rstrip() for line in footer_cell['source']]
    assert lines[0] == seac._FOOTER_SEPARATOR
    # the note's own content stayed behind in a separate, earlier cell
    assert 'useful information' not in ''.join(footer_cell['source'])


def test_footer_disabled_by_empty_override(built: tuple[Path, list[Path]]):
    # the shared ``built`` fixture already disables the footer -- this just
    # makes that fact an explicit, named assertion rather than an implicit
    # side effect of every other test in this module passing
    src = _read(built[1], 'case_note')
    assert 'sphinx-examples-as-code' not in src
    assert 'generated by' not in src.lower()


def test_footer_custom_string(tmp_path: Path):
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    returncode, out, err = _run_sphinx_build(
        _sphinx_build_cmd(
            SRCDIR,
            html_dir,
            doctree_dir,
            ('-D', 'sphinx_examples_as_code_conf.footer=Custom footer text.'),
        ),
    )
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    py_path = next((html_dir / '_downloads').rglob('*case_note*.py'))
    content = py_path.read_text(encoding='utf-8')
    assert '# Custom footer text.' in content
    assert 'sphinx-examples-as-code' not in content
