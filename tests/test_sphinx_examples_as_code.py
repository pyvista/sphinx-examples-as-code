"""Unit tests for individual ``sphinx_examples_as_code.py`` functions.

Complements ``test_tinypages.py``'s full Sphinx-build
tests with fast, direct tests of branches that are impractical to reach
through a full build (mocked Sphinx app, hand-built doctree fragments).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import Mock

from docutils import nodes
from docutils.core import publish_doctree
import pytest
from sphinx import addnodes
from sphinx.errors import ConfigError

import sphinx_examples_as_code as seac

if TYPE_CHECKING:
    from pathlib import Path


def _parse(rst: str) -> nodes.document:
    return publish_doctree(rst, settings_overrides={'report_level': 5})


def _ctx(
    *,
    fmt: str = 'py',
    in_see_also: bool = False,
    in_footer: bool = False,
    base_url: str | None = None,
    docname: str = 'page',
):
    """Build a minimal ``_RenderContext`` with a mocked Sphinx app."""
    app = Mock()
    app.config.html_baseurl = base_url
    app.builder.get_target_uri.side_effect = lambda d: f'{d}.html'
    return seac._RenderContext(
        app=app, docname=docname, fmt=fmt, in_see_also=in_see_also, in_footer=in_footer
    )


# ---------------------------------------------------------------------------
# _has_class
# ---------------------------------------------------------------------------


def test_has_class_true():
    assert seac._has_class(nodes.container(classes=['sd-dropdown']), 'sd-dropdown')


def test_has_class_false():
    assert not seac._has_class(nodes.container(classes=['other']), 'sd-dropdown')


def test_has_class_node_without_get():
    assert not seac._has_class(nodes.Text('hi'), 'sd-dropdown')


# ---------------------------------------------------------------------------
# _is_examples_heading
# ---------------------------------------------------------------------------


def test_is_examples_heading_rubric():
    doctree = _parse('.. rubric:: Examples')
    assert seac._is_examples_heading(doctree[0])


def test_is_examples_heading_case_insensitive():
    doctree = _parse('.. rubric:: EXAMPLES')
    assert seac._is_examples_heading(doctree[0])


def test_is_examples_heading_wrong_text():
    doctree = _parse('.. rubric:: See Also')
    assert not seac._is_examples_heading(doctree[0])


def test_is_examples_heading_wrong_node_type():
    doctree = _parse('Examples')
    assert not seac._is_examples_heading(doctree[0])


def test_is_examples_heading_title():
    doctree = _parse('Examples\n========\n\nbody text')
    assert seac._is_examples_heading(doctree[0])


def test_add_comment_multiline_with_blank_line():
    lines: list[str] = []
    seac._add_comment(lines, 'first\n\nthird')
    assert lines == ['# first', '#', '# third']


# ---------------------------------------------------------------------------
# _render_inline
# ---------------------------------------------------------------------------


def test_render_inline_literal():
    doctree = _parse('``code``')
    assert seac._render_inline(doctree[0][0], _ctx()) == '`code`'


def test_render_inline_plain_text():
    doctree = _parse('hello world')
    assert seac._render_inline(doctree[0], _ctx()) == 'hello world'


def test_render_inline_image_returns_empty():
    doctree = _parse('.. image:: foo.png')
    assert seac._render_inline(doctree[0], _ctx()) == ''


def test_render_inline_childless_fallback_to_astext():
    assert seac._render_inline(nodes.transition(), _ctx()) == ''


# ---------------------------------------------------------------------------
# _is_see_also_heading
# ---------------------------------------------------------------------------


def test_is_see_also_heading_rubric():
    doctree = _parse('.. rubric:: See Also')
    assert seac._is_see_also_heading(doctree[0])


def test_is_see_also_heading_case_insensitive():
    doctree = _parse('.. rubric:: SEE ALSO')
    assert seac._is_see_also_heading(doctree[0])


def test_is_see_also_heading_wrong_text():
    doctree = _parse('.. rubric:: Notes')
    assert not seac._is_see_also_heading(doctree[0])


def test_is_see_also_heading_wrong_node_type():
    doctree = _parse('See Also is not this paragraph')
    assert not seac._is_see_also_heading(doctree[0])


# ---------------------------------------------------------------------------
# _resolve_link_url
# ---------------------------------------------------------------------------


def test_resolve_link_url_no_refuri_returns_none():
    ref = nodes.reference()
    assert seac._resolve_link_url(ref, _ctx(base_url='https://docs.example.com/')) is None


def test_resolve_link_url_no_base_url_configured_returns_none():
    ref = nodes.reference()
    ref['refuri'] = 'foo.html#bar'
    assert seac._resolve_link_url(ref, _ctx(base_url=None)) is None


def test_resolve_link_url_external_hyperlink_already_absolute():
    ref = nodes.reference()
    ref['refuri'] = 'https://example.com/page'
    # an external hyperlink's refuri is already a full URL -- it resolves
    # even with no base_url configured, since there's nothing to join
    assert seac._resolve_link_url(ref, _ctx(base_url=None)) == 'https://example.com/page'


def test_resolve_link_url_internal_reference_resolved_against_base_url():
    ref = nodes.reference()
    ref['refuri'] = '../plotting/_autosummary/pyvista.Plotter.html#pyvista.Plotter'
    ctx = _ctx(
        base_url='https://docs.pyvista.org/',
        docname='api/core/_autosummary/pyvista.PolyData',
    )
    url = seac._resolve_link_url(ref, ctx)
    assert (
        url == 'https://docs.pyvista.org/api/core/plotting/_autosummary/pyvista.Plotter.html'
        '#pyvista.Plotter'
    )


def test_resolve_link_url_same_page_reference_uses_refid():
    # a reference to a target on the *same* page has no refuri at all --
    # Sphinx uses refid (an in-page anchor) instead
    ref = nodes.reference()
    ref['refid'] = 'docstring_cases.Sample'
    ctx = _ctx(base_url='https://docs.pyvista.org/', docname='docstring_cases')
    url = seac._resolve_link_url(ref, ctx)
    assert url == 'https://docs.pyvista.org/docstring_cases.html#docstring_cases.Sample'


# ---------------------------------------------------------------------------
# _render_reference
# ---------------------------------------------------------------------------


def _reference_with_literal(display: str, refuri: str | None) -> nodes.reference:
    """Build a reference wrapping a literal, like a resolved :class:/:meth:/... xref."""
    ref = nodes.reference()
    if refuri:
        ref['refuri'] = refuri
    ref += nodes.literal('', display)
    return ref


def _reference_plain(display: str, refuri: str | None) -> nodes.reference:
    """Build a reference with no inner literal, like a :ref:/:doc: or external hyperlink."""
    ref = nodes.reference()
    if refuri:
        ref['refuri'] = refuri
    ref += nodes.Text(display)
    return ref


def test_render_reference_code_no_url_backtick_wrapped():
    ref = _reference_with_literal('pyvista.Plotter', None)
    assert seac._render_reference(ref, _ctx(fmt='py')) == '`pyvista.Plotter`'
    assert seac._render_reference(ref, _ctx(fmt='ipynb')) == '`pyvista.Plotter`'


def test_render_reference_plain_no_url_no_backticks():
    ref = _reference_plain('Some Section', None)
    assert seac._render_reference(ref, _ctx(fmt='py')) == 'Some Section'
    assert seac._render_reference(ref, _ctx(fmt='ipynb')) == 'Some Section'


def test_render_reference_py_non_see_also_omits_url_even_when_resolved():
    ref = _reference_with_literal('pyvista.Plotter', 'plotter.html')
    ctx = _ctx(fmt='py', in_see_also=False, base_url='https://docs.pyvista.org/')
    assert seac._render_reference(ref, ctx) == '`pyvista.Plotter`'


def test_render_reference_py_see_also_uses_name_and_url_on_own_line():
    ref = _reference_with_literal('pyvista.Plotter', 'plotter.html')
    ctx = _ctx(fmt='py', in_see_also=True, base_url='https://docs.pyvista.org/')
    result = seac._render_reference(ref, ctx)
    assert result == '\npyvista.Plotter https://docs.pyvista.org/plotter.html\n'
    assert '`' not in result


def test_render_reference_ipynb_code_resolved_becomes_markdown_link():
    ref = _reference_with_literal('pyvista.Plotter', 'plotter.html')
    ctx = _ctx(fmt='ipynb', base_url='https://docs.pyvista.org/')
    assert (
        seac._render_reference(ref, ctx)
        == '[`pyvista.Plotter`](https://docs.pyvista.org/plotter.html)'
    )


def test_render_reference_ipynb_plain_resolved_becomes_markdown_link():
    ref = _reference_plain('Some Section', 'other.html')
    ctx = _ctx(fmt='ipynb', base_url='https://docs.pyvista.org/')
    assert seac._render_reference(ref, ctx) == '[Some Section](https://docs.pyvista.org/other.html)'


def test_render_reference_ipynb_in_see_also_same_as_elsewhere():
    # per spec: ipynb treats See Also refs exactly like any other ref
    ref = _reference_with_literal('pyvista.Plotter', 'plotter.html')
    ctx_outside = _ctx(fmt='ipynb', in_see_also=False, base_url='https://docs.pyvista.org/')
    ctx_inside = _ctx(fmt='ipynb', in_see_also=True, base_url='https://docs.pyvista.org/')
    assert seac._render_reference(ref, ctx_outside) == seac._render_reference(ref, ctx_inside)


def test_render_reference_py_in_footer_uses_name_and_url_inline():
    # unlike in_see_also, no surrounding newlines: the footer is a short
    # sentence, not a list of links, so the link stays inline
    ref = _reference_plain('sphinx-examples-as-code', 'repo.html')
    ctx = _ctx(fmt='py', in_footer=True, base_url='https://docs.pyvista.org/')
    result = seac._render_reference(ref, ctx)
    assert result == 'sphinx-examples-as-code https://docs.pyvista.org/repo.html'


def test_render_reference_ipynb_in_footer_same_as_elsewhere():
    ref = _reference_plain('sphinx-examples-as-code', 'repo.html')
    ctx_outside = _ctx(fmt='ipynb', in_footer=False, base_url='https://docs.pyvista.org/')
    ctx_inside = _ctx(fmt='ipynb', in_footer=True, base_url='https://docs.pyvista.org/')
    assert seac._render_reference(ref, ctx_outside) == seac._render_reference(ref, ctx_inside)


# ---------------------------------------------------------------------------
# _join_segments
# ---------------------------------------------------------------------------


def test_join_segments_text_then_code_no_blank():
    result = seac._join_segments([('text', ['# a']), ('code', ['x = 1'])])
    assert result == ['# a', 'x = 1']


def test_join_segments_code_then_text_blank():
    result = seac._join_segments([('code', ['x = 1']), ('text', ['# a'])])
    assert result == ['x = 1', '', '# a']


def test_join_segments_directive_gets_blank_both_sides():
    result = seac._join_segments(
        [('text', ['# a']), ('directive', ['# NOTE:', '# n']), ('code', ['x = 1'])]
    )
    assert result == ['# a', '', '# NOTE:', '# n', '', 'x = 1']


def test_join_segments_first_segment_no_leading_blank():
    result = seac._join_segments([('directive', ['# H'])])
    assert result == ['# H']


def test_join_segments_skips_empty_segments():
    result = seac._join_segments([('text', []), ('code', ['x = 1'])])
    assert result == ['x = 1']


# ---------------------------------------------------------------------------
# _convert_doctest_block
# ---------------------------------------------------------------------------


def test_convert_doctest_block_strips_prompts():
    doctree = _parse('>>> x = 1\n>>> y = 2')
    segments = seac._convert_doctest_block(doctree[0])
    assert segments == [('code', ['x = 1', 'y = 2'])]


def test_convert_doctest_block_drops_output():
    doctree = _parse('>>> 1 + 1\n2')
    segments = seac._convert_doctest_block(doctree[0])
    assert segments == [('code', ['1 + 1'])]


def test_convert_doctest_block_no_code_returns_empty():
    # a doctest block always has at least one >>> line by construction, so
    # this is exercised indirectly through _convert_node's dispatch instead
    node = nodes.doctest_block()
    node += nodes.Text('not a real doctest line')
    assert seac._convert_doctest_block(node) == []


def test_convert_doctest_block_continuation_line():
    doctree = _parse('>>> def f():\n...     return 1')
    segments = seac._convert_doctest_block(doctree[0])
    assert segments == [('code', ['def f():', '    return 1'])]


def test_convert_doctest_block_internal_blank_line_kept():
    node = nodes.doctest_block('', '>>> x = 1\n\n>>> y = 2')
    segments = seac._convert_doctest_block(node)
    assert segments == [('code', ['x = 1', '', 'y = 2'])]


def test_convert_doctest_block_trailing_blank_trimmed():
    node = nodes.doctest_block('', '>>> x = 1\n\n')
    segments = seac._convert_doctest_block(node)
    assert segments == [('code', ['x = 1'])]


# ---------------------------------------------------------------------------
# _convert_literal_block
# ---------------------------------------------------------------------------


def test_convert_literal_block_python():
    # Sphinx's own code-block directive sets a ``language`` attribute
    # (unlike plain docutils', which uses CSS classes instead), so this
    # node is built directly to match what a real Sphinx build produces.
    node = nodes.literal_block('', 'x = 1')
    node['language'] = 'python'
    segments = seac._convert_literal_block(node)
    assert segments == [('code', ['x = 1'])]


def test_convert_literal_block_capitalized_python():
    # sphinx-gallery emits ``.. code-block:: Python`` (capitalized) -- the
    # language match must not be case-sensitive, or gallery code blocks
    # would silently turn into comments instead of staying real code.
    node = nodes.literal_block('', 'x = 1')
    node['language'] = 'Python'
    assert seac._convert_literal_block(node) == [('code', ['x = 1'])]


def test_convert_literal_block_non_python_becomes_comment():
    # 'directive' kind (not 'text'): set off with blank lines on both sides,
    # since it's a distinct preformatted block in its original source, not
    # ordinary prose that happens to sit next to it
    node = nodes.literal_block('', 'echo hi')
    node['language'] = 'bash'
    segments = seac._convert_literal_block(node)
    assert segments == [('directive', ['# echo hi'])]


def test_convert_literal_block_empty_non_python():
    node = nodes.literal_block('', '')
    node['language'] = 'bash'
    assert seac._convert_literal_block(node) == []


def test_convert_literal_block_python_whitespace_only():
    node = nodes.literal_block('', '   \n   ')
    node['language'] = 'python'
    assert seac._convert_literal_block(node) == []


# ---------------------------------------------------------------------------
# _clean_stray_rst_markup / _clean_code_comment
# ---------------------------------------------------------------------------


def test_clean_stray_hyperlink_with_url():
    text = '# See `Ext <https://example.com>`_.'
    assert seac._clean_stray_rst_markup(text) == '# See Ext <https://example.com>.'


def test_clean_stray_hyperlink_without_url():
    text = '# see `target`_ here'
    assert seac._clean_stray_rst_markup(text) == '# see target here'


def test_clean_stray_xref_explicit_title():
    text = '# :func:`short <pyvista.long.path>`.'
    assert seac._clean_stray_rst_markup(text) == '# short.'


def test_clean_stray_xref_plain():
    text = '# :class:`pyvista.Plotter`.'
    assert seac._clean_stray_rst_markup(text) == '# pyvista.Plotter.'


def test_clean_code_comment_ignores_real_code():
    line = "x = '`not a ref`_'"
    assert seac._clean_code_comment(line) == line


def test_clean_code_comment_cleans_comment_lines():
    line = '# see `target`_'
    assert seac._clean_code_comment(line) == '# see target'


@pytest.mark.parametrize(('unicode_char', 'ascii_char'), seac.ASCII_REPLACEMENTS.items())
def test_clean_code_comment_replaces_non_ascii_chars(unicode_char: int, ascii_char: str):
    line = f'# comment {chr(unicode_char)} text'
    assert seac._clean_code_comment(line) == f'# comment {ascii_char} text'


# ---------------------------------------------------------------------------
# _convert_admonition / _convert_node dispatch
# ---------------------------------------------------------------------------


def test_convert_node_note():
    doctree = _parse('.. note::\n\n   hello')
    assert seac._convert_node(doctree[0], _ctx()) == [('directive', ['# NOTE:', '# hello'])]


def test_convert_node_seealso():
    node = addnodes.seealso()
    p1 = nodes.paragraph()
    p1 += nodes.Text('See X')
    p2 = nodes.paragraph()
    p2 += nodes.Text('More info')
    node += p1
    node += p2
    assert seac._convert_node(node, _ctx()) == [
        ('directive', ['# SEE ALSO:', '# See X', '# More info'])
    ]


def test_convert_node_seealso_propagates_in_see_also_to_references():
    # a reference nested inside .. seealso:: should get the "See Also"
    # link treatment (name + url on its own line), not the regular
    # link-omitted treatment it'd get anywhere else in a .py file
    node = addnodes.seealso()
    p = nodes.paragraph()
    p += _reference_with_literal('pyvista.Plotter', 'plotter.html')
    node += p
    ctx = _ctx(fmt='py', base_url='https://docs.pyvista.org/')
    segments = seac._convert_node(node, ctx)
    assert segments == [
        ('directive', ['# SEE ALSO:', '# pyvista.Plotter https://docs.pyvista.org/plotter.html'])
    ]


def test_convert_node_see_also_section_treated_like_admonition():
    doctree = _parse('Intro\n-----\n\nintro\n\nSee Also\n--------\n\n>>> x = 1')
    see_also_section = doctree[1]
    assert seac._convert_node(see_also_section, _ctx()) == [('directive', ['# SEE ALSO:', 'x = 1'])]


def test_convert_node_generic_admonition_uses_title():
    doctree = _parse('.. admonition:: Custom Title\n\n   body text')
    assert seac._convert_node(doctree[0], _ctx()) == [
        ('directive', ['# Custom Title:', '# body text'])
    ]


def test_convert_node_generic_admonition_no_title_defaults_to_note():
    node = nodes.admonition()
    p = nodes.paragraph()
    p += nodes.Text('body')
    node += p
    assert seac._convert_node(node, _ctx()) == [('directive', ['# NOTE:', '# body'])]


def test_convert_node_admonition_with_no_body_keeps_label():
    node = nodes.note()
    assert seac._convert_node(node, _ctx()) == [('directive', ['# NOTE:'])]


def test_convert_node_skip_subtree_class():
    node = nodes.container(classes=['sd-dropdown'])
    p = nodes.paragraph()
    p += nodes.Text('hidden')
    node += p
    assert seac._convert_node(node, _ctx()) == []


def test_convert_node_drops_sphinx_tags_paragraph():
    # sphinx-tags' ".. tags:: load" compiles to a plain paragraph with
    # class "tags" -- site navigation, not documentation content
    node = nodes.paragraph(classes=['tags'])
    node += nodes.inline('', 'Tags: ')
    node += nodes.reference('', 'load', refuri='_tags/load.html')
    assert seac._convert_node(node, _ctx()) == []


def test_convert_node_ignored_type():
    doctree = _parse('.. image:: foo.png')
    assert seac._convert_node(doctree[0], _ctx()) == []


def test_convert_node_versionmodified():
    node = addnodes.versionmodified()
    p = nodes.paragraph()
    p += nodes.Text('Added in version 1.0.')
    node += p
    assert seac._convert_node(node, _ctx()) == [('text', ['# Added in version 1.0.'])]


def test_convert_node_container_recurses():
    doctree = _parse('- item one\n- item two')
    segments = seac._convert_node(doctree[0], _ctx())
    assert segments == [('text', ['# item one']), ('text', ['# item two'])]


def test_convert_node_empty_paragraph_returns_empty():
    assert seac._convert_node(nodes.paragraph(), _ctx()) == []


def test_convert_node_section_title_becomes_heading_and_children_recurse():
    # gallery mode: a "# %%" cell's own sibling section -- its title gets
    # the same title+underline treatment as the file's own header, and its
    # other children (prose, code, ...) get converted normally
    section = nodes.section()
    section += nodes.title('', 'A subsection')
    section += nodes.paragraph('', 'prose here')
    code_block = nodes.literal_block('', 'x = 1')
    code_block['language'] = 'Python'
    section += code_block

    assert seac._convert_node(section, _ctx()) == [
        ('directive', ['# A subsection', '# ------------']),
        ('text', ['# prose here']),
        ('code', ['x = 1']),
    ]


def test_convert_node_literal_block_dispatch():
    node = nodes.literal_block('', 'x = 1')
    node['language'] = 'python'

    assert seac._convert_node(node, _ctx()) == [('code', ['x = 1'])]


# ---------------------------------------------------------------------------
# _has_real_code
# ---------------------------------------------------------------------------


def test_has_real_code_true():
    assert seac._has_real_code('x = 1\n')


def test_has_real_code_syntax_error():
    assert not seac._has_real_code('def (:\n')


def test_has_real_code_only_constant_expression():
    assert not seac._has_real_code('"just a comment string"\n')


def test_has_real_code_empty():
    assert not seac._has_real_code('')


# ---------------------------------------------------------------------------
# _span_from / _examples_spans
# ---------------------------------------------------------------------------


def test_span_from_stops_at_boundary():
    doctree = _parse('.. rubric:: Examples\n\ntext\n\n.. rubric:: Notes\n\nmore')
    end = seac._span_from(doctree, 1)
    assert end == 2  # only the paragraph, not the second rubric or beyond


def test_span_from_does_not_stop_at_see_also_rubric():
    doctree = _parse('.. rubric:: Examples\n\ntext\n\n.. rubric:: See Also\n\nmore')
    end = seac._span_from(doctree, 1)
    assert end == len(doctree.children)


def test_span_from_does_not_stop_at_see_also_section():
    # a hand-written "See Also\n--------" heading nests as its own section
    # rather than staying a flat sibling -- still shouldn't truncate the span
    doctree = _parse('.. rubric:: Examples\n\ntext\n\nSee Also\n--------\n\nmore')
    end = seac._span_from(doctree, 1)
    assert end == len(doctree.children)


def test_is_see_also_section_true():
    doctree = _parse('Intro\n-----\n\nintro text\n\nSee Also\n--------\n\nsome text')
    see_also_section = doctree[1]
    assert seac._is_see_also_section(see_also_section)


def test_is_see_also_section_false_for_other_heading():
    doctree = _parse('Intro\n-----\n\nintro text\n\nNotes\n-----\n\nsome text')
    other_section = doctree[1]
    assert not seac._is_see_also_section(other_section)


def test_find_external_see_also_found_before_span():
    # numpydoc's own "See Also" field is canonically reordered before
    # "Examples", so it sits *before* the span rather than inside it
    doctree = _parse('.. rubric:: Examples\n\ntext')
    seealso_node = addnodes.seealso()
    seealso_node += nodes.paragraph('', 'related')
    doctree.insert(0, seealso_node)  # now: [seealso, rubric, text]

    start = 2  # right after the rubric, now at index 1
    end = seac._span_from(doctree, start)
    found = seac._find_external_see_also(doctree, start, end)
    assert found is seealso_node


def test_find_external_see_also_returns_none_when_absent():
    doctree = _parse('.. rubric:: Examples\n\ntext')
    found = seac._find_external_see_also(doctree, 1, len(doctree.children))
    assert found is None


def test_span_from_runs_to_end_of_parent():
    doctree = _parse('.. rubric:: Examples\n\ntext one\n\ntext two')
    end = seac._span_from(doctree, 1)
    assert end == len(doctree.children)


def test_examples_spans_finds_multiple_headings():
    doctree = _parse('.. rubric:: Examples\n\ncode here')
    spans = seac._examples_spans(doctree)
    assert len(spans) == 1
    parent, start, _, _ = spans[0]
    assert parent is doctree
    assert start == 1


# ---------------------------------------------------------------------------
# _qualified_name_for
# ---------------------------------------------------------------------------


def test_qualified_name_for_desc_ancestor():
    desc = addnodes.desc()
    sig = addnodes.desc_signature(ids=['pkg.mod.func'])
    desc += sig
    content = addnodes.desc_content()
    heading = nodes.rubric()
    content += heading
    desc += content
    assert seac._qualified_name_for(heading, 'page', 1) == 'pkg.mod.func'


def test_qualified_name_for_fallback_no_desc_ancestor():
    section = nodes.section()
    heading = nodes.rubric()
    section += heading
    assert seac._qualified_name_for(heading, 'mypage', 3) == 'mypage-example-3'


def test_qualified_name_for_desc_signature_without_ids_falls_back():
    desc = addnodes.desc()
    sig = addnodes.desc_signature()  # no ids set
    desc += sig
    content = addnodes.desc_content()
    heading = nodes.rubric()
    content += heading
    desc += content
    assert seac._qualified_name_for(heading, 'mypage', 2) == 'mypage-example-2'


def test_qualified_name_for_examples_section_sibling_of_desc():
    r"""A real ``Examples\n--------`` heading, not a bare rubric.

    Matches an actual numpydoc-rendered autosummary page: docutils can't
    nest a new section inside the non-sectioning ``desc_content`` a
    docstring's other fields render into, so it promotes "Examples" to a
    section of its own, sitting as a *sibling* of the object's ``desc`` --
    not an ancestor of the heading the way the plain-rubric case above is.
    Confirmed against a real pyvista build, where this exact structure
    produced a nonsensical ``<docname>-example-1`` header for a page that
    unambiguously documents one function.
    """
    page_section = nodes.section()
    page_section += nodes.title('', 'download_bunny')
    desc = addnodes.desc()
    desc += addnodes.desc_signature(ids=['pyvista.examples.downloads.download_bunny'])
    page_section += desc
    examples_section = nodes.section()
    heading = nodes.title('', 'Examples')
    examples_section += heading
    page_section += examples_section

    assert (
        seac._qualified_name_for(heading, 'page', 1) == 'pyvista.examples.downloads.download_bunny'
    )


def test_qualified_name_for_examples_section_sibling_no_desc_falls_back():
    # a promoted sibling section with nothing resembling a desc anywhere
    # nearby -- e.g. a plain prose page's own "Examples\n--------" heading
    page_section = nodes.section()
    page_section += nodes.title('', 'Some Guide Page')
    examples_section = nodes.section()
    heading = nodes.title('', 'Examples')
    examples_section += heading
    page_section += examples_section

    assert seac._qualified_name_for(heading, 'mypage', 4) == 'mypage-example-4'


# ---------------------------------------------------------------------------
# _header_segment
# ---------------------------------------------------------------------------


def test_header_segment_format():
    kind, lines = seac._header_segment('pyvista.read')
    assert kind == 'directive'
    assert lines[0] == '# Examples from pyvista.read'
    assert lines[1] == '# ' + '-' * len('Examples from pyvista.read')


# ---------------------------------------------------------------------------
# _title_underline_segment
# ---------------------------------------------------------------------------


def test_title_underline_segment_format():
    kind, lines = seac._title_underline_segment('A Title')
    assert kind == 'directive'
    assert lines == ['# A Title', '# -------']


def test_convert_node_title_uses_underline_treatment():
    # a standalone title, as reached via a gallery sibling section
    title = nodes.title('', 'A subsection')
    assert seac._convert_node(title, _ctx()) == [
        ('directive', ['# A subsection', '# ------------'])
    ]


def test_convert_node_empty_title_returns_empty():
    assert seac._convert_node(nodes.title(), _ctx()) == []


# ---------------------------------------------------------------------------
# _footer_segment
# ---------------------------------------------------------------------------


def _footer_app(*, base_url: str | None = None) -> Mock:
    """Build a minimal mocked Sphinx app, sufficient for ``_footer_segment``."""
    app = Mock()
    app.config.html_baseurl = base_url
    app.builder.get_target_uri.side_effect = lambda d: f'{d}.html'
    return app


def test_footer_segment_none_returns_empty():
    assert seac._footer_segment(None, 'py', _footer_app(), 'page') == []


def test_footer_segment_empty_string_returns_empty():
    assert seac._footer_segment('', 'py', _footer_app(), 'page') == []


_FOOTER_SEP_LINE = f'# {seac._FOOTER_SEPARATOR}'


def test_footer_segment_starts_with_the_separator():
    # marks the footer as trailing boilerplate, distinct from the example's
    # own comments -- and (dashes specifically) renders as a real <hr> in
    # .ipynb, confirmed against an actual CommonMark parser, not assumed
    segments = seac._footer_segment('Generated file.', 'py', _footer_app(), 'page')
    assert segments[0][1][0] == _FOOTER_SEP_LINE


def test_footer_segment_plain_text_single_line():
    # no RST markup at all -- the common case for a custom footer -- behaves
    # exactly as it always has otherwise: one comment line after the separator
    segments = seac._footer_segment('Generated file.', 'py', _footer_app(), 'page')
    assert segments == [('directive', [_FOOTER_SEP_LINE, '# Generated file.'])]


def test_footer_segment_plain_text_multiline_becomes_one_comment_per_line():
    # a single newline (no blank line) stays one RST paragraph, with the
    # line break preserved in its text -- verified against a real docutils
    # parse, not assumed: publish_doctree('line one\nline two') keeps them
    # as one <paragraph> whose text is 'line one\nline two', not folded
    # into 'line one line two'
    segments = seac._footer_segment('line one\nline two', 'py', _footer_app(), 'page')
    kind, lines = segments[0]
    assert kind == 'directive'
    assert lines == [_FOOTER_SEP_LINE, '# line one', '# line two']


def test_footer_segment_blank_line_separated_paragraphs_both_kept():
    segments = seac._footer_segment('first\n\nsecond', 'py', _footer_app(), 'page')
    kind, lines = segments[0]
    assert kind == 'directive'
    assert lines == [_FOOTER_SEP_LINE, '# first', '# second']


def test_footer_segment_bare_url_recognized_by_docutils_natively():
    # no custom regex needed for this: docutils' standard RST parser
    # already auto-recognizes a bare http(s) URL as a hyperlink on its own.
    # There's no custom link text here though, so it's still shown as both
    # the label and the target -- an RST hyperlink with its own text (the
    # tests above) reads better, which is why _DEFAULT_FOOTER uses one.
    footer = 'https://example.com/issues'
    _kind, lines = seac._footer_segment(footer, 'ipynb', _footer_app(), 'page')[0]
    assert lines == [
        _FOOTER_SEP_LINE,
        '# [https://example.com/issues](https://example.com/issues)',
    ]


def test_footer_segment_link_mid_sentence_in_py_stays_inline():
    # unlike a "See Also" reference, a footer link doesn't force its own
    # line (in_footer=True renders inline, not in_see_also's newline-wrap)
    # -- a link sitting mid-sentence keeps reading as one line
    footer = 'See https://example.com/issues for details.'
    _kind, lines = seac._footer_segment(footer, 'py', _footer_app(), 'page')[0]
    assert lines == [
        _FOOTER_SEP_LINE,
        '# See https://example.com/issues https://example.com/issues for details.',
    ]


def test_footer_segment_rst_hyperlink_py_shows_text_and_url():
    # rendered with in_footer=True -- reuses the same resolve/render
    # machinery as "See Also", just without its own-line wrapping
    footer = '`Report issues <https://example.com/issues>`_'
    _kind, lines = seac._footer_segment(footer, 'py', _footer_app(), 'page')[0]
    assert lines == [_FOOTER_SEP_LINE, '# Report issues https://example.com/issues']


def test_footer_segment_rst_hyperlink_ipynb_becomes_clickable_link():
    footer = '`Report issues <https://example.com/issues>`_'
    _kind, lines = seac._footer_segment(footer, 'ipynb', _footer_app(), 'page')[0]
    assert lines == [_FOOTER_SEP_LINE, '# [Report issues](https://example.com/issues)']


def test_footer_segment_default_footer_py():
    lines = seac._footer_segment(seac._DEFAULT_FOOTER, 'py', _footer_app(), 'page')[0][1]
    assert lines == [
        _FOOTER_SEP_LINE,
        '# Generated by sphinx-examples-as-code https://github.com/pyvista/sphinx-examples-as-code',
    ]


def test_footer_segment_default_footer_ipynb_links_clickable():
    lines = seac._footer_segment(seac._DEFAULT_FOOTER, 'ipynb', _footer_app(), 'page')[0][1]
    line0 = '# Generated by [sphinx-examples-as-code](https://github.com/pyvista/sphinx-examples-as-code)'
    assert lines == [_FOOTER_SEP_LINE, line0]
    # the repo name is a real link, not just inline code with no link at all
    assert '`sphinx-examples-as-code`' not in ''.join(lines)


# ---------------------------------------------------------------------------
# _write_source / _make_download_node
# ---------------------------------------------------------------------------


def test_write_source_writes_file_with_32_char_digest(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    rel_path = seac._write_source(app, 'pkg.func', 'x = 1\n')
    digest, filename = rel_path.split('/')
    assert len(digest) == 32
    assert filename == 'pkg_func.py'
    assert (tmp_path / '_downloads' / digest / filename).read_text() == 'x = 1\n'


def test_write_source_empty_name_fallback(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    rel_path = seac._write_source(app, '', 'x = 1\n')
    assert rel_path.endswith('/example.py')


def test_make_download_node_single_entry():
    node = seac._make_download_node([('Download Python source code', 'abc123/foo.py')])
    assert isinstance(node, nodes.paragraph)

    reference = node.children[0]
    assert isinstance(reference, addnodes.download_reference)
    assert reference['filename'] == 'abc123/foo.py'
    assert reference.astext() == 'Download Python source code'


def test_make_download_node_multiple_entries_separated():
    node = seac._make_download_node(
        [
            ('Download Python source code', 'abc123/foo.py'),
            ('Download Jupyter notebook', 'def456/foo.ipynb'),
        ]
    )
    assert len(node.children) == 3  # reference, separator text, reference
    assert isinstance(node.children[0], addnodes.download_reference)
    assert node.children[1].astext() == ' | '
    assert isinstance(node.children[2], addnodes.download_reference)
    assert node.children[2]['filename'] == 'def456/foo.ipynb'


# ---------------------------------------------------------------------------
# _strip_comment_prefix / _segments_to_cells / _cell_source / _build_notebook
# ---------------------------------------------------------------------------


def test_strip_comment_prefix_with_space():
    assert seac._strip_comment_prefix('# hello') == 'hello'


def test_strip_comment_prefix_bare_hash():
    assert seac._strip_comment_prefix('#') == ''


def test_strip_comment_prefix_no_space_after_hash():
    assert seac._strip_comment_prefix('#hello') == 'hello'


def test_segments_to_cells_groups_by_kind():
    segments = [
        ('directive', ['# Examples from x', '# --------------']),
        ('text', ['# prose here']),
        ('code', ['import sys']),
        ('code', ['x = 1']),
    ]
    cells = seac._segments_to_cells(segments)
    assert cells == [
        ('markdown', ['Examples from x', '--------------', '', 'prose here']),
        ('code', ['import sys', '', 'x = 1']),
    ]


def test_segments_to_cells_empty_input():
    assert seac._segments_to_cells([]) == []


def test_segments_to_cells_skips_empty_lines_segment():
    # a segment with no lines at all (e.g. a no-op admonition) shouldn't
    # start a new cell or otherwise affect grouping
    segments = [
        ('text', ['# prose']),
        ('text', []),
        ('code', ['x = 1']),
    ]
    cells = seac._segments_to_cells(segments)
    assert cells == [
        ('markdown', ['prose']),
        ('code', ['x = 1']),
    ]


def test_cell_source_code_no_hard_break():
    assert seac._cell_source(['a', 'b', 'c'], 'code') == ['a\n', 'b\n', 'c']


def test_cell_source_markdown_adds_hard_break():
    assert seac._cell_source(['a', 'b', 'c'], 'markdown') == ['a  \n', 'b  \n', 'c  ']


def test_cell_source_markdown_blank_line_no_hard_break():
    # a genuinely blank line stays blank -- adding trailing spaces there
    # wouldn't be a meaningful hard break, just stray whitespace
    assert seac._cell_source(['a', '', 'b'], 'markdown') == ['a  \n', '\n', 'b  ']


def test_cell_source_empty():
    assert seac._cell_source([], 'code') == []


def test_build_notebook_structure():
    cells = [('markdown', ['# Title']), ('code', ['x = 1'])]
    notebook = seac._build_notebook(cells)
    assert notebook['nbformat'] == 4
    assert notebook['nbformat_minor'] == 5
    assert len(notebook['cells']) == 2
    assert notebook['cells'][0]['cell_type'] == 'markdown'
    assert notebook['cells'][0]['id'] == 'cell-0'
    assert 'execution_count' not in notebook['cells'][0]
    assert notebook['cells'][1]['cell_type'] == 'code'
    assert notebook['cells'][1]['execution_count'] is None
    assert notebook['cells'][1]['outputs'] == []
    assert notebook['cells'][1]['id'] == 'cell-1'


def test_build_notebook_valid_nbformat():
    nbformat = pytest.importorskip('nbformat')
    cells = [('markdown', ['# Title']), ('code', ['import sys'])]
    notebook = seac._build_notebook(cells)
    node = nbformat.from_dict(notebook)
    nbformat.validate(node)  # raises if invalid


# ---------------------------------------------------------------------------
# _write_notebook
# ---------------------------------------------------------------------------


def test_write_notebook_writes_valid_json(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    notebook = seac._build_notebook([('code', ['x = 1'])])
    rel_path = seac._write_notebook(app, 'pkg.func', notebook)
    digest, filename = rel_path.split('/')
    assert len(digest) == 32
    assert filename == 'pkg_func.ipynb'
    written = (tmp_path / '_downloads' / digest / filename).read_text()
    assert json.loads(written) == notebook


# ---------------------------------------------------------------------------
# _process_span / _process_doctree / setup
# ---------------------------------------------------------------------------


def _build_examples_doctree(rst: str):
    doctree = _parse(rst)
    heading = doctree[0]
    parent = heading.parent
    start = parent.index(heading) + 1
    end = seac._span_from(parent, start)
    return doctree, parent, start, end, heading


def test_build_segments_bare_rubric_switches_to_see_also():
    _doctree, parent, start, end, _heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1\n\n.. rubric:: See Also\n\n>>> y = 2'
    )
    nodes_in_span = list(parent.children[start:end])
    segments = seac._build_segments(nodes_in_span, _ctx())
    assert segments == [
        ('code', ['x = 1']),
        ('directive', ['# SEE ALSO:', 'y = 2']),
    ]


def test_build_segments_bare_rubric_see_also_affects_only_what_follows():
    _doctree, parent, start, end, _heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n.. rubric:: See Also\n\n>>> x = 1'
    )
    nodes_in_span = list(parent.children[start:end])
    # a reference-only paragraph inserted right after the rubric: since
    # it's *after* the "See Also" switch, it should get the See Also
    # treatment (name + url), not the regular link-omitted treatment
    ref_paragraph = nodes.paragraph()
    ref_paragraph += _reference_with_literal('pyvista.Plotter', 'plotter.html')
    nodes_in_span.insert(1, ref_paragraph)

    ctx = _ctx(fmt='py', base_url='https://docs.pyvista.org/')
    segments = seac._build_segments(nodes_in_span, ctx)
    assert segments == [
        (
            'directive',
            [
                '# SEE ALSO:',
                '# pyvista.Plotter https://docs.pyvista.org/plotter.html',
                'x = 1',
            ],
        )
    ]


def test_process_span_no_code_no_download(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\njust prose, no code'
    )
    original_len = len(parent.children)
    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'bottom',
        ['py', 'ipynb'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )
    assert len(parent.children) == original_len


def test_process_span_code_segment_but_not_real_code(tmp_path: Path):
    # a doctest block containing only a comment line: has a 'code' segment
    # (it matched the >>> prompt), but ast.parse finds no real statements
    app = Mock(outdir=str(tmp_path))
    _, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> # just a comment'
    )
    original_len = len(parent.children)
    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'bottom',
        ['py', 'ipynb'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )
    assert len(parent.children) == original_len


def test_process_span_inserts_at_bottom(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'bottom',
        ['py', 'ipynb'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )
    assert isinstance(parent.children[end], nodes.paragraph)


def test_process_span_inserts_at_top(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'top',
        ['py', 'ipynb'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )
    assert isinstance(parent.children[start], nodes.paragraph)


def test_process_span_includes_external_see_also(tmp_path: Path):
    # numpydoc's own "See Also" field sits before "Examples", outside the
    # normal span -- _process_span should still fold it into the output
    app = Mock(outdir=str(tmp_path))
    doctree = _parse('.. rubric:: Examples\n\n>>> x = 1')
    seealso_node = addnodes.seealso()
    seealso_node += nodes.paragraph('', 'related info')
    doctree.insert(0, seealso_node)

    heading = doctree[1]
    parent = heading.parent
    start = parent.index(heading) + 1
    end = seac._span_from(parent, start)

    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'bottom',
        ['py'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )

    written = next((tmp_path / '_downloads').rglob('*.py'))
    content = written.read_text()
    assert '# SEE ALSO:' in content
    assert '# related info' in content


def test_process_span_appends_footer_with_blank_line_before(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'bottom',
        ['py'],
        'Generated footer text.',
        seac._DEFAULT_LINK_LABELS,
    )

    written = next((tmp_path / '_downloads').rglob('*.py'))
    lines = written.read_text().splitlines()
    sep_idx = lines.index(_FOOTER_SEP_LINE)
    assert lines[sep_idx - 1] == ''  # blank line before the footer's separator
    assert lines[sep_idx + 1] == '# Generated footer text.'


def test_process_span_no_footer_configured_omits_it(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'bottom',
        ['py'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )

    written = next((tmp_path / '_downloads').rglob('*.py'))
    assert 'Generated' not in written.read_text()


def test_process_span_footer_in_notebook_is_linkified(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'bottom',
        ['ipynb'],
        'Report to https://example.com/issues',
        seac._DEFAULT_LINK_LABELS,
    )

    written = next((tmp_path / '_downloads').rglob('*.ipynb'))
    notebook = json.loads(written.read_text())
    all_source = ''.join(''.join(cell['source']) for cell in notebook['cells'])
    assert '[https://example.com/issues](https://example.com/issues)' in all_source


@pytest.mark.parametrize(
    ('formats', 'expected_extensions'),
    [
        (['py'], ['.py']),
        (['ipynb'], ['.ipynb']),
        (['py', 'ipynb'], ['.py', '.ipynb']),
        (['ipynb', 'py'], ['.py', '.ipynb']),  # canonical order regardless of config order
        ([], []),
    ],
)
def test_process_span_respects_formats_config(
    tmp_path: Path, formats: list[str], expected_extensions: list[str]
):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    original_len = len(parent.children)
    seac._process_span(
        app,
        'page',
        parent,
        start,
        end,
        heading,
        1,
        'bottom',
        formats,
        None,
        seac._DEFAULT_LINK_LABELS,
    )

    if not expected_extensions:
        assert len(parent.children) == original_len
        return

    download_paragraph = parent.children[end]
    references = [
        c for c in download_paragraph.children if isinstance(c, addnodes.download_reference)
    ]
    assert len(references) == len(expected_extensions)
    for reference, ext in zip(references, expected_extensions, strict=True):
        assert reference['filename'].endswith(ext)


def test_process_span_custom_link_labels_used_in_download_node(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    custom_labels = {'py': 'Get the script', 'ipynb': 'Get the notebook'}
    seac._process_span(
        app, 'page', parent, start, end, heading, 1, 'bottom', ['py', 'ipynb'], None, custom_labels
    )

    text = parent.children[end].astext()
    assert 'Get the script' in text
    assert 'Get the notebook' in text
    assert 'Download Python source code' not in text


def test_process_doctree_skips_when_no_download_support():
    app = Mock()
    app.builder.download_support = False
    doctree = _parse('.. rubric:: Examples\n\n>>> x = 1')
    original = doctree.pformat()
    seac._process_doctree(app, doctree, 'page')
    assert doctree.pformat() == original


@pytest.mark.parametrize(
    ('position', 'expected_index'),
    [('top', 1), ('bottom', 2)],
)
def test_process_doctree_processes_spans(tmp_path: Path, position: str, expected_index: int):
    app = Mock(outdir=str(tmp_path))
    app.builder.download_support = True
    app.config.sphinx_examples_as_code_conf = {
        'link_position': position,
        'formats': ['py', 'ipynb'],
        'gallery_downloads': False,
        'footer': None,
        'link_labels': seac._DEFAULT_LINK_LABELS,
    }

    doctree = _parse('.. rubric:: Examples\n\n>>> x = 1')

    seac._process_doctree(app, doctree, 'page')

    assert isinstance(doctree[expected_index], nodes.paragraph)
    assert isinstance(doctree[expected_index][0], addnodes.download_reference)


def test_setup_registers_connect_and_config():
    app = Mock()
    result = seac.setup(app)
    app.connect.assert_any_call('doctree-resolved', seac._process_doctree)
    app.connect.assert_any_call('config-inited', seac._finalize_conf)
    app.add_config_value.assert_called_once_with('sphinx_examples_as_code_conf', {}, 'env')
    assert result['version'] == seac.__version__
    assert result['parallel_read_safe'] is True
    assert result['parallel_write_safe'] is True


# ---------------------------------------------------------------------------
# sphinx-gallery integration (sphinx_examples_as_code_conf['gallery_downloads'])
#
# Node shapes below mirror a real sphinx-gallery build, confirmed by
# building one with sphinx-gallery installed and dumping its doctree --
# see e.g. ``language='Python'`` (capitalized), the download-link note
# sitting *outside* the page's section, and the footer/timing/signature
# nested at the end of it.
# ---------------------------------------------------------------------------


def _gallery_note() -> nodes.note:
    note = nodes.note(classes=[seac._GALLERY_NOTE_CLASS])
    note += nodes.paragraph('', 'Go to the end to download the full example code.')
    return note


def _gallery_footer() -> nodes.container:
    footer = nodes.container(classes=[seac._GALLERY_FOOTER_CLASS, 'sphx-glr-footer-example'])
    footer += nodes.container(classes=['sphx-glr-download', 'sphx-glr-download-python'])
    return footer


def _gallery_timing() -> nodes.paragraph:
    timing = nodes.paragraph(classes=[seac._GALLERY_TIMING_CLASS])
    timing += nodes.strong('', 'Total running time of the script:')
    return timing


def _gallery_signature() -> nodes.paragraph:
    signature = nodes.paragraph(classes=[seac._GALLERY_SIGNATURE_CLASS])
    signature += nodes.reference(
        '', 'Gallery generated by Sphinx-Gallery', refuri='https://sphinx-gallery.github.io'
    )
    return signature


def _gallery_output_block() -> nodes.literal_block:
    output = nodes.literal_block('', '2', classes=['sphx-glr-script-out'])
    output['language'] = 'none'
    return output


def _build_gallery_doctree(*, with_code: bool = True, with_title: bool = True) -> nodes.document:
    """Build a doctree shaped like a real sphinx-gallery page."""
    doctree = _parse('')
    doctree += _gallery_note()  # a sibling of the section, not nested inside it

    section = nodes.section(classes=['sphx-glr-example-title'], ids=['ex'])
    if with_title:
        section += nodes.title('', 'An example')
    section += nodes.paragraph('', 'Some intro text.')
    if with_code:
        code_block = nodes.literal_block('', 'x = 1')
        code_block['language'] = 'Python'
        section += code_block
        section += _gallery_output_block()
    section += _gallery_timing()
    section += _gallery_footer()
    section += _gallery_signature()
    doctree += section
    return doctree


def test_strip_gallery_furniture_removes_note_and_footer():
    doctree = _build_gallery_doctree()
    assert seac._strip_gallery_furniture(doctree) is True

    remaining_classes = {
        cls
        for node in doctree.findall()
        for cls in getattr(node, 'get', lambda *_a: [])('classes', [])
    }
    assert seac._GALLERY_NOTE_CLASS not in remaining_classes
    assert seac._GALLERY_FOOTER_CLASS not in remaining_classes


def test_strip_gallery_furniture_keeps_timing_and_signature():
    # unlike the note/footer, these stay on the rendered page -- they're
    # only excluded from the *generated* file, via _SKIP_SUBTREE_CLASSES
    doctree = _build_gallery_doctree()
    seac._strip_gallery_furniture(doctree)

    remaining_classes = {
        cls
        for node in doctree.findall()
        for cls in getattr(node, 'get', lambda *_a: [])('classes', [])
    }
    assert seac._GALLERY_TIMING_CLASS in remaining_classes
    assert seac._GALLERY_SIGNATURE_CLASS in remaining_classes


def test_strip_gallery_furniture_leaves_the_rest_untouched():
    doctree = _build_gallery_doctree()
    seac._strip_gallery_furniture(doctree)
    section = next(doctree.findall(nodes.section))
    assert section[0].astext() == 'An example'
    assert any(isinstance(n, nodes.literal_block) for n in section.children)


def test_strip_gallery_furniture_not_a_gallery_page_returns_false():
    doctree = _parse('Some ordinary page\n\nwith a paragraph.')
    original = doctree.pformat()
    assert seac._strip_gallery_furniture(doctree) is False
    assert doctree.pformat() == original  # nothing removed


def test_convert_node_drops_gallery_output_block():
    # sphx-glr-script-out is captured stdout, not documentation -- same
    # treatment as doctest output, via the generic skip-subtree mechanism.
    assert seac._convert_node(_gallery_output_block(), _ctx()) == []


def test_convert_node_drops_gallery_timing_block():
    assert seac._convert_node(_gallery_timing(), _ctx()) == []


def test_convert_node_drops_gallery_signature_block():
    assert seac._convert_node(_gallery_signature(), _ctx()) == []


def test_gallery_body_spans_from_first_section_to_end_of_document():
    # _gallery_body's parent is always the document itself now -- title
    # exclusion is _process_gallery_page's job (only the *first* section's
    # title gets dropped; sibling sections keep theirs, see below)
    doctree = _build_gallery_doctree()
    seac._strip_gallery_furniture(doctree)
    parent, start, end = seac._gallery_body(doctree)
    assert parent is doctree
    section = next(doctree.findall(nodes.section))
    assert start == doctree.index(section)
    assert end == len(doctree.children)


def _gallery_subsection(title: str, code: str) -> nodes.section:
    """Build a sibling section for a second ``# %%`` cell with its own heading."""
    section = nodes.section(ids=[title.lower().replace(' ', '-')])
    section += nodes.title('', title)
    section += nodes.paragraph('', 'Some prose.')
    code_block = nodes.literal_block('', code)
    code_block['language'] = 'Python'
    section += code_block
    return section


def _build_gallery_doctree_with_sibling_sections() -> nodes.document:
    """Build a doctree shaped like a real multi-``# %%``-cell gallery page.

    The regression case: each ``# %%`` cell with its own heading becomes a
    *sibling* section at the document level (confirmed against a real
    sphinx-gallery build), not nested inside the page's outer section --
    this is what _gallery_body/_process_gallery_page must span across
    rather than just the first section's children.
    """
    doctree = _parse('')
    doctree += _gallery_note()

    first = nodes.section(classes=['sphx-glr-example-title'], ids=['ex'])
    first += nodes.title('', 'An example')
    first += nodes.paragraph('', 'Intro text.')
    import_block = nodes.literal_block('', 'import sys')
    import_block['language'] = 'Python'
    first += import_block
    doctree += first

    second = _gallery_subsection('Second cell', 'x = 1')
    second += _gallery_timing()
    second += _gallery_footer()
    second += _gallery_signature()
    doctree += second

    return doctree


def test_gallery_body_spans_multiple_sibling_sections():
    doctree = _build_gallery_doctree_with_sibling_sections()
    seac._strip_gallery_furniture(doctree)
    parent, start, end = seac._gallery_body(doctree)
    assert parent is doctree
    spanned = doctree.children[start:end]
    assert len(spanned) == 2  # both sibling sections included
    assert all(isinstance(c, nodes.section) for c in spanned)


def test_gallery_body_falls_back_to_document_without_a_section():
    doctree = _parse('just a paragraph, no section')
    parent, start, end = seac._gallery_body(doctree)
    assert parent is doctree
    assert start == 0
    assert end == len(doctree.children)


@pytest.mark.parametrize(
    ('docname', 'expected'),
    [
        ('auto_examples/plot_minimal', 'plot_minimal'),
        ('plot_minimal', 'plot_minimal'),
        ('auto_examples/nested/plot_foo', 'plot_foo'),
    ],
)
def test_gallery_example_name(docname: str, expected: str):
    assert seac._gallery_example_name(docname) == expected


def test_gallery_page_title_extracts_text():
    section = nodes.section()
    section += nodes.title('', 'Create Circular Arcs')
    section += nodes.paragraph('', 'intro')
    assert seac._gallery_page_title(section) == 'Create Circular Arcs'


def test_gallery_page_title_no_title_returns_none():
    section = nodes.section()
    section += nodes.paragraph('', 'no title here')
    assert seac._gallery_page_title(section) is None


def test_gallery_page_title_empty_title_returns_none():
    section = nodes.section()
    section += nodes.title()
    assert seac._gallery_page_title(section) is None


def test_process_gallery_page_not_a_gallery_page_is_a_no_op(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    doctree = _parse('Some ordinary page\n\nwith a paragraph.')
    original = doctree.pformat()
    seac._process_gallery_page(
        app, 'page', doctree, 'bottom', ['py', 'ipynb'], None, seac._DEFAULT_LINK_LABELS
    )
    assert doctree.pformat() == original


def test_process_gallery_page_header_uses_the_pages_own_title(tmp_path: Path):
    # not "# Examples from plot_minimal" -- gallery mode converts the whole
    # page, so its own title reads far better than a docname-derived name
    app = Mock(outdir=str(tmp_path))
    doctree = _build_gallery_doctree()  # title: 'An example'
    seac._process_gallery_page(
        app,
        'auto_examples/plot_minimal',
        doctree,
        'bottom',
        ['py'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )

    written = next((tmp_path / '_downloads').rglob('*.py'))
    lines = written.read_text(encoding='utf-8').splitlines()
    assert lines[0] == '# An example'
    assert lines[1] == '# ' + '-' * len('An example')
    assert not lines[0].startswith('# Examples from')


def test_process_gallery_page_no_title_falls_back_to_examples_from_header(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    doctree = _build_gallery_doctree(with_title=False)
    seac._process_gallery_page(
        app,
        'auto_examples/plot_minimal',
        doctree,
        'bottom',
        ['py'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )

    written = next((tmp_path / '_downloads').rglob('*.py'))
    assert written.read_text(encoding='utf-8').startswith('# Examples from plot_minimal')


def test_process_gallery_page_strips_furniture_and_inserts_download(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    doctree = _build_gallery_doctree()
    seac._process_gallery_page(
        app,
        'auto_examples/plot_minimal',
        doctree,
        'bottom',
        ['py'],
        None,
        seac._DEFAULT_LINK_LABELS,
    )

    assert not any(seac._has_class(n, seac._GALLERY_FOOTER_CLASS) for n in doctree.findall())
    assert not any(seac._has_class(n, seac._GALLERY_NOTE_CLASS) for n in doctree.findall())
    # timing/signature stay on the rendered page ...
    assert any(seac._has_class(n, seac._GALLERY_TIMING_CLASS) for n in doctree.findall())
    assert any(seac._has_class(n, seac._GALLERY_SIGNATURE_CLASS) for n in doctree.findall())

    # 'bottom' -> a sibling of the section, at the end of the document
    download_paragraphs = [
        n
        for n in doctree.children
        if isinstance(n, nodes.paragraph)
        and any(isinstance(c, addnodes.download_reference) for c in n.children)
    ]
    assert len(download_paragraphs) == 1
    assert doctree.children[-1] is download_paragraphs[0]
    reference = download_paragraphs[0].children[0]
    assert reference['filename'].endswith('/plot_minimal.py')

    # ... but never leak into the generated file itself
    written = (tmp_path / '_downloads' / reference['filename']).read_text(encoding='utf-8')
    assert 'Total running time' not in written
    assert 'Gallery generated by Sphinx-Gallery' not in written


def test_process_gallery_page_includes_code_from_sibling_sections(tmp_path: Path):
    # the actual bug this guards against: a "# %%" cell with its own
    # heading becomes a *sibling* section (not nested inside the first),
    # and its code must not be silently dropped
    app = Mock(outdir=str(tmp_path))
    doctree = _build_gallery_doctree_with_sibling_sections()
    seac._process_gallery_page(
        app, 'auto_examples/plot_multi', doctree, 'bottom', ['py'], None, seac._DEFAULT_LINK_LABELS
    )

    written = next((tmp_path / '_downloads').rglob('*.py'))
    content = written.read_text(encoding='utf-8')
    assert 'import sys' in content
    assert '# Second cell' in content  # the sibling section's own title
    assert 'x = 1' in content  # the sibling section's own code


def test_process_gallery_page_respects_top_position(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    doctree = _build_gallery_doctree()
    seac._process_gallery_page(
        app, 'plot_minimal', doctree, 'top', ['py'], None, seac._DEFAULT_LINK_LABELS
    )

    section = next(doctree.findall(nodes.section))
    # right after the title (index 0), before the intro paragraph
    assert isinstance(section[1], nodes.paragraph)
    assert isinstance(section[1][0], addnodes.download_reference)


def test_process_gallery_page_no_real_code_strips_furniture_but_no_download(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    doctree = _build_gallery_doctree(with_code=False)
    seac._process_gallery_page(
        app, 'plot_minimal', doctree, 'bottom', ['py', 'ipynb'], None, seac._DEFAULT_LINK_LABELS
    )

    assert not any(seac._has_class(n, seac._GALLERY_FOOTER_CLASS) for n in doctree.findall())
    assert not any(isinstance(n, addnodes.download_reference) for n in doctree.findall())


def test_process_gallery_page_empty_after_stripping_is_a_no_op(tmp_path: Path):
    # contrived, but defensive: a doctree that's *only* the footer -- once
    # stripped there's nothing left to span at all
    app = Mock(outdir=str(tmp_path))
    doctree = _parse('')
    doctree += _gallery_footer()
    seac._process_gallery_page(
        app, 'page', doctree, 'bottom', ['py'], None, seac._DEFAULT_LINK_LABELS
    )
    assert not any(isinstance(n, addnodes.download_reference) for n in doctree.findall())


def test_process_gallery_page_no_wrapping_section_still_works(tmp_path: Path):
    # defensive fallback: a gallery footer with no enclosing section at all
    app = Mock(outdir=str(tmp_path))
    doctree = _parse('')
    code_block = nodes.literal_block('', 'x = 1')
    code_block['language'] = 'Python'
    doctree += code_block
    doctree += _gallery_footer()

    seac._process_gallery_page(
        app, 'weird_page', doctree, 'bottom', ['py'], None, seac._DEFAULT_LINK_LABELS
    )

    written = next((tmp_path / '_downloads').rglob('*.py'))
    assert 'x = 1' in written.read_text(encoding='utf-8')


def test_process_doctree_gallery_mode_disabled_by_default(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    app.builder.download_support = True
    app.config.sphinx_examples_as_code_conf = {
        'link_position': 'bottom',
        'formats': ['py'],
        'gallery_downloads': False,
        'footer': None,
        'link_labels': seac._DEFAULT_LINK_LABELS,
    }

    doctree = _build_gallery_doctree()
    seac._process_doctree(app, doctree, 'plot_minimal')

    # untouched: the sphinx-gallery footer is still there, no download added
    assert any(seac._has_class(n, seac._GALLERY_FOOTER_CLASS) for n in doctree.findall())
    assert not any(isinstance(n, addnodes.download_reference) for n in doctree.findall())


def test_process_doctree_gallery_mode_enabled(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    app.builder.download_support = True
    app.config.sphinx_examples_as_code_conf = {
        'link_position': 'bottom',
        'formats': ['py'],
        'gallery_downloads': True,
        'footer': None,
        'link_labels': seac._DEFAULT_LINK_LABELS,
    }

    doctree = _build_gallery_doctree()
    seac._process_doctree(app, doctree, 'plot_minimal')

    assert not any(seac._has_class(n, seac._GALLERY_FOOTER_CLASS) for n in doctree.findall())
    assert any(isinstance(n, addnodes.download_reference) for n in doctree.findall())


# ---------------------------------------------------------------------------
# _normalized_html_baseurl
# ---------------------------------------------------------------------------


def _app_with_baseurl(base_url: str | None) -> Mock:
    app = Mock()
    app.config.html_baseurl = base_url
    return app


def test_normalized_html_baseurl_unset_is_none():
    assert seac._normalized_html_baseurl(_app_with_baseurl(None)) is None


def test_normalized_html_baseurl_empty_string_is_none():
    assert seac._normalized_html_baseurl(_app_with_baseurl('')) is None


def test_normalized_html_baseurl_missing_scheme_is_none():
    assert seac._normalized_html_baseurl(_app_with_baseurl('docs.example.com/')) is None


def test_normalized_html_baseurl_garbage_is_none():
    assert seac._normalized_html_baseurl(_app_with_baseurl('not a url at all')) is None


def test_normalized_html_baseurl_non_http_scheme_is_none():
    assert seac._normalized_html_baseurl(_app_with_baseurl('ftp://docs.example.com/')) is None


def test_normalized_html_baseurl_adds_missing_trailing_slash():
    app = _app_with_baseurl('https://docs.example.com/en/stable')
    assert seac._normalized_html_baseurl(app) == 'https://docs.example.com/en/stable/'


def test_normalized_html_baseurl_leaves_trailing_slash_alone():
    app = _app_with_baseurl('https://docs.example.com/en/stable/')
    assert seac._normalized_html_baseurl(app) == 'https://docs.example.com/en/stable/'


def test_normalized_html_baseurl_bare_domain_gets_trailing_slash():
    app = _app_with_baseurl('https://docs.example.com')
    assert seac._normalized_html_baseurl(app) == 'https://docs.example.com/'


# ---------------------------------------------------------------------------
# _coerce_conf_value
# ---------------------------------------------------------------------------


def test_coerce_conf_value_non_string_passes_through():
    # already the real type (set directly in conf.py) -- untouched
    assert seac._coerce_conf_value('formats', ['py']) == ['py']
    assert seac._coerce_conf_value('gallery_downloads', True) is True


def test_coerce_conf_value_formats_splits_on_comma():
    assert seac._coerce_conf_value('formats', 'py,ipynb') == ['py', 'ipynb']


def test_coerce_conf_value_formats_single_value_no_comma():
    assert seac._coerce_conf_value('formats', 'py') == ['py']


@pytest.mark.parametrize(('raw', 'expected'), [('0', False), ('1', True)])
def test_coerce_conf_value_gallery_downloads_bool_strings(raw: str, expected: bool):
    assert seac._coerce_conf_value('gallery_downloads', raw) is expected


def test_coerce_conf_value_gallery_downloads_invalid_string_raises():
    with pytest.raises(ConfigError, match="must be '0' or '1'"):
        seac._coerce_conf_value('gallery_downloads', 'yes')


def test_coerce_conf_value_link_position_string_passes_through():
    assert seac._coerce_conf_value('link_position', 'bottom') == 'bottom'


# ---------------------------------------------------------------------------
# _finalize_conf
# ---------------------------------------------------------------------------


def test_finalize_conf_empty_dict_fills_in_all_defaults():
    config = Mock(sphinx_examples_as_code_conf={})
    seac._finalize_conf(Mock(), config)
    assert config.sphinx_examples_as_code_conf == seac._CONF_DEFAULTS


def test_finalize_conf_partial_dict_keeps_the_rest_default():
    config = Mock(sphinx_examples_as_code_conf={'link_position': 'bottom'})
    seac._finalize_conf(Mock(), config)
    conf = config.sphinx_examples_as_code_conf
    assert conf['link_position'] == 'bottom'
    assert conf['formats'] == ['py', 'ipynb']
    assert conf['gallery_downloads'] is False
    assert conf['footer'] == seac._DEFAULT_FOOTER


def test_finalize_conf_default_footer_text():
    # pinned explicitly, not just via dict equality against _CONF_DEFAULTS --
    # this is user-facing text, so a change to its wording should be a
    # deliberate, visible diff here rather than an incidental one. One line,
    # one link (see _footer_segment/_DEFAULT_FOOTER's own comment for why).
    assert seac._DEFAULT_FOOTER == (
        'Generated by `sphinx-examples-as-code <https://github.com/pyvista/sphinx-examples-as-code>`_'
    )


def test_finalize_conf_footer_none_disables_it():
    config = Mock(sphinx_examples_as_code_conf={'footer': None})
    seac._finalize_conf(Mock(), config)
    assert config.sphinx_examples_as_code_conf['footer'] is None


def test_finalize_conf_footer_custom_string():
    config = Mock(sphinx_examples_as_code_conf={'footer': 'Custom footer.'})
    seac._finalize_conf(Mock(), config)
    assert config.sphinx_examples_as_code_conf['footer'] == 'Custom footer.'


def test_finalize_conf_unknown_key_raises():
    config = Mock(sphinx_examples_as_code_conf={'link_postion': 'bottom'})  # typo
    with pytest.raises(ConfigError, match=r'unknown key\(s\).*link_postion'):
        seac._finalize_conf(Mock(), config)


def test_finalize_conf_unknown_key_error_lists_valid_keys():
    config = Mock(sphinx_examples_as_code_conf={'nope': 1})
    with pytest.raises(ConfigError, match='link_position'):
        seac._finalize_conf(Mock(), config)


def test_finalize_conf_coerces_cli_override_style_strings():
    # what a ``-D sphinx_examples_as_code_conf.key=value`` override looks
    # like by the time it reaches config-inited -- see _coerce_conf_value
    config = Mock(sphinx_examples_as_code_conf={'formats': 'py,ipynb', 'gallery_downloads': '1'})
    seac._finalize_conf(Mock(), config)
    conf = config.sphinx_examples_as_code_conf
    assert conf['formats'] == ['py', 'ipynb']
    assert conf['gallery_downloads'] is True


def test_finalize_conf_base_url_is_an_unknown_key():
    config = Mock(sphinx_examples_as_code_conf={'base_url': 'https://docs.example.com'})
    with pytest.raises(ConfigError, match=r'unknown key\(s\).*base_url'):
        seac._finalize_conf(Mock(), config)


def test_finalize_conf_link_labels_defaults_when_unset():
    config = Mock(sphinx_examples_as_code_conf={})
    seac._finalize_conf(Mock(), config)
    assert config.sphinx_examples_as_code_conf['link_labels'] == seac._DEFAULT_LINK_LABELS


def test_finalize_conf_link_labels_partial_override_keeps_other_default():
    config = Mock(sphinx_examples_as_code_conf={'link_labels': {'py': 'Get the script'}})
    seac._finalize_conf(Mock(), config)
    link_labels = config.sphinx_examples_as_code_conf['link_labels']
    assert link_labels['py'] == 'Get the script'
    assert link_labels['ipynb'] == seac._DEFAULT_LINK_LABELS['ipynb']


def test_finalize_conf_link_labels_not_a_dict_raises():
    config = Mock(sphinx_examples_as_code_conf={'link_labels': 'nope'})
    with pytest.raises(ConfigError, match=r"link_labels'\] must be a dict"):
        seac._finalize_conf(Mock(), config)


def test_finalize_conf_link_labels_unknown_format_key_raises():
    config = Mock(sphinx_examples_as_code_conf={'link_labels': {'pdf': 'Download PDF'}})
    with pytest.raises(ConfigError, match=r"link_labels'\] has unknown key\(s\).*pdf"):
        seac._finalize_conf(Mock(), config)


def test_finalize_conf_does_not_mutate_the_defaults():
    # a regression guard for accidentally sharing/mutating _CONF_DEFAULTS
    # across builds instead of writing a fresh merged dict each time
    original = dict(seac._CONF_DEFAULTS)
    config = Mock(sphinx_examples_as_code_conf={'link_position': 'bottom'})
    seac._finalize_conf(Mock(), config)
    assert original == seac._CONF_DEFAULTS


def test_finalize_conf_link_labels_override_does_not_mutate_the_defaults():
    original = dict(seac._DEFAULT_LINK_LABELS)
    config = Mock(sphinx_examples_as_code_conf={'link_labels': {'py': 'Get the script'}})
    seac._finalize_conf(Mock(), config)
    assert original == seac._DEFAULT_LINK_LABELS


# ---------------------------------------------------------------------------
# ASCII_REPLACEMENTS (smart quotes -> ASCII)
# ---------------------------------------------------------------------------


def test_add_comment_normalizes_smart_quotes():
    # _add_comment is what prose (via _render_inline) goes through; without
    # this, only doctest-code-comment lines (via _clean_code_comment) were
    # normalized, leaving ordinary prose comments with curly quotes.
    lines: list[str] = []
    seac._add_comment(lines, 'an array\u2019s \u201cvalues\u201d')
    assert lines == ['# an array\'s "values"']


def test_convert_node_paragraph_normalizes_smart_quotes():
    doctree = _parse('an array\u2019s \u201cvalues\u201d')
    assert seac._convert_node(doctree[0], _ctx()) == [('text', ['# an array\'s "values"'])]
