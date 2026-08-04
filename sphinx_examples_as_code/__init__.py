"""Generate downloadable Python/Jupyter files from docstring "Examples" sections.

See the README for configuration options and the full conversion rules.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING
from urllib.parse import urljoin
from urllib.parse import urlsplit

from docutils import nodes
from sphinx import addnodes
from sphinx.errors import ConfigError

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.config import Config

# Node types marking the start of another section - bound the end of an
# Examples span. desc/index are included so a class's Examples section
# doesn't swallow its members, which numpydoc renders as flat siblings
# inside the same desc_content.
_BOUNDARY_TYPES = (
    nodes.rubric,
    nodes.title,
    nodes.section,
    addnodes.desc,
    addnodes.index,
)

# Node types with no textual content worth keeping under any circumstance.
_IGNORED_TYPES = (nodes.image, nodes.figure, nodes.comment, nodes.raw)

# sphinx-design containers, matched by CSS class rather than node type so
# this doesn't need to import sphinx_design. Dropped entirely: a dropdown's
# content isn't part of the visible example, and a tab-set here only holds
# figures (already ignored) plus tab-label cruft.
_SKIP_SUBTREE_CLASSES = ('sd-dropdown', 'sd-tab-set')

_CONTAINER_TYPES = (
    nodes.bullet_list,
    nodes.enumerated_list,
    nodes.definition_list,
    nodes.definition_list_item,
    nodes.list_item,
    nodes.definition,
    nodes.term,
    nodes.classifier,
    nodes.block_quote,
    nodes.container,
    nodes.compound,
    addnodes.versionmodified,
)

# Fixed-label admonitions (.. note::, .. warning::, .. seealso::, ...) - as
# opposed to the generic .. admonition:: Custom Title, handled separately.
_ADMONITION_LABELS = {
    nodes.attention: 'ATTENTION',
    nodes.caution: 'CAUTION',
    nodes.danger: 'DANGER',
    nodes.error: 'ERROR',
    nodes.hint: 'HINT',
    nodes.important: 'IMPORTANT',
    nodes.note: 'NOTE',
    addnodes.seealso: 'SEE ALSO',
    nodes.tip: 'TIP',
    nodes.warning: 'WARNING',
}

_PYTHON_LANGUAGES = ('python', 'py', 'python3')

# A chunk of generated lines tagged with how it should be spaced relative to
# its neighbors when segments are joined (see ``_join_segments``):
#   'code'      real Python source
#   'text'      a plain comment (prose, a paragraph, ...)
#   'directive' a comment block that must be visually set off with a blank
#               line both before and after it (the title header; a
#               ``# NOTE:``-style admonition block)
Segment = tuple[str, list[str]]

# Replacements for non-ASCII chars
ASCII_REPLACEMENTS = str.maketrans(
    {
        '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
        '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
        '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK
        '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK
    }
)


def _has_class(node: nodes.Node, css_class: str) -> bool:
    getter = getattr(node, 'get', None)
    return bool(getter) and css_class in getter('classes', [])


def _is_examples_heading(node: nodes.Node) -> bool:
    """Check whether ``node`` is a heading (rubric or title) named "Examples"."""
    return (
        isinstance(node, (nodes.rubric, nodes.title))
        and node.astext().strip().lower() == 'examples'
    )


def _is_see_also_heading(node: nodes.Node) -> bool:
    r"""Check whether ``node`` is a heading (rubric or title) named "See Also".

    Two written forms: a bare ``.. rubric:: See Also`` (flat sibling), or
    ``See Also\\n--------`` underline text (docutils nests it as a section).
    Both get the same treatment as a ``.. seealso::`` directive.
    """
    return (
        isinstance(node, (nodes.rubric, nodes.title))
        and node.astext().strip().lower() == 'see also'
    )


def _is_see_also_section(node: nodes.Node) -> bool:
    """Check whether ``node`` is a nested section titled "See Also"."""
    return (
        isinstance(node, nodes.section)
        and bool(node.children)
        and _is_see_also_heading(node.children[0])
    )


@dataclass(frozen=True)
class _RenderContext:
    """State threaded through the conversion functions for one output format."""

    app: Sphinx
    docname: str
    fmt: str  # 'py' or 'ipynb'
    in_see_also: bool = False


def _resolve_link_url(node: nodes.reference, ctx: _RenderContext) -> str | None:
    """Resolve a reference node's target to an absolute URL, if possible.

    Returns ``None`` for an unresolved target or when no base URL is
    configured -- a standalone downloaded file needs an absolute URL, and
    Sphinx's own ``refuri``/``refid`` are only meaningful relative to the
    current page.
    """
    refuri = node.get('refuri')
    if refuri and urlsplit(refuri).netloc:
        return refuri  # already absolute (an external hyperlink)

    base_url = ctx.app.config.sphinx_examples_as_code_base_url
    if not base_url:
        return None
    current_page_url = urljoin(base_url, ctx.app.builder.get_target_uri(ctx.docname))

    if refuri:
        return urljoin(current_page_url, refuri)

    refid = node.get('refid')
    if refid:
        return f'{current_page_url}#{refid}'

    return None


def _add_comment(lines: list[str], text: str) -> None:
    """Append ``text`` to ``lines`` as one or more Python comment lines."""
    for line in text.splitlines():
        line_ = line.rstrip().translate(ASCII_REPLACEMENTS)
        lines.append(f'# {line_}' if line_ else '#')


# Doctest/code-block content is preformatted, so docutils never resolves
# RST markup (xref roles, hyperlinks) written inside a comment there - it
# passes through as raw text. These clean it up without touching real code.
_STRAY_XREF_RE = re.compile(r':(?:py:)?\w+:`([^`<>]+?)\s*(?:<[^<>]+>)?`')
_STRAY_HYPERLINK_RE = re.compile(r'`([^`<>]+?)\s*(?:<([^`<>]+)>)?`_+')


def _clean_stray_rst_markup(text: str) -> str:
    """Strip unparsed cross-reference/hyperlink RST syntax from a comment line."""
    text = _STRAY_XREF_RE.sub(r'\1', text)
    return _STRAY_HYPERLINK_RE.sub(
        lambda m: f'{m.group(1)} <{m.group(2)}>' if m.group(2) else m.group(1), text
    )


def _clean_code_comment(line: str) -> str:
    """Apply stray-markup cleanup and ASCII replacements to comment lines."""
    if line.lstrip().startswith('#'):
        line = _clean_stray_rst_markup(line)
        line = line.translate(ASCII_REPLACEMENTS)
    return line


def _render_reference(node: nodes.reference, ctx: _RenderContext) -> str:
    """Render a resolved or unresolved cross-reference/hyperlink.

    The target URL, if resolved, is used as a clickable markdown link in
    notebooks (always), and as literal URL text in ``.py`` (only within a
    "See Also" part -- elsewhere in ``.py`` the link is simply omitted).
    """
    display = node.astext()
    is_code = any(isinstance(child, nodes.literal) for child in node.children)
    url = _resolve_link_url(node, ctx)

    if url is None:
        return f'`{display}`' if is_code else display

    if ctx.fmt == 'ipynb':
        return f'[`{display}`]({url})' if is_code else f'[{display}]({url})'

    if ctx.in_see_also:
        # Surrounding newlines set this off on its own line; _add_comment
        # splits on them like any other multi-line text.
        return f'\n{display} {url}\n'

    return f'`{display}`' if is_code else display


def _render_inline(node: nodes.Node, ctx: _RenderContext) -> str:
    """Render a node's inline content to a plain string.

    References go through ``_render_reference``. Other code-like spans
    (double-backtick literals, an unresolved reference's inner literal) are
    backtick-wrapped; everything else is flattened to plain text.
    """
    if isinstance(node, nodes.reference):
        return _render_reference(node, ctx)
    if isinstance(node, nodes.literal):
        return f'`{node.astext()}`'
    if isinstance(node, (nodes.image, nodes.figure, nodes.raw, nodes.comment)):
        return ''
    if isinstance(node, nodes.Text):
        return str(node)
    if hasattr(node, 'children') and node.children:
        return ''.join(_render_inline(child, ctx) for child in node.children)
    return node.astext()


def _join_segments(segments: list[Segment]) -> list[str]:
    """Flatten segments into lines, applying inter-segment spacing rules.

    - a blank line always follows a ``code`` segment, whatever comes next
    - a ``directive`` segment always gets a blank line both before and
      after it
    - otherwise (e.g. prose directly above a code block, or two prose
      segments back to back), no blank line is forced
    """
    lines: list[str] = []
    prev_kind: str | None = None
    for kind, seg_lines in segments:
        if not seg_lines:
            continue
        need_blank = prev_kind is not None and (
            prev_kind == 'code' or kind == 'directive' or prev_kind == 'directive'
        )
        if need_blank:
            lines.append('')
        lines.extend(seg_lines)
        prev_kind = kind
    return lines


def _convert_doctest_block(node: nodes.doctest_block) -> list[Segment]:
    """Convert a doctest block, stripping ``>>> ``/``... `` prompts.

    Non-prompted, non-blank lines are expected doctest *output* -- we only
    care about the input code, so those are dropped entirely rather than
    kept as comments.
    """
    lines: list[str] = []
    has_code = False
    for line in node.astext().splitlines():
        if line.startswith('>>> ') or line == '>>>':
            lines.append(_clean_code_comment(line[4:]))
            has_code = True
        elif line.startswith('... ') or line == '...':
            lines.append(_clean_code_comment(line[4:]))
        elif not line.strip():
            lines.append('')
        # else: doctest output line - dropped
    if not has_code:
        return []
    while lines and not lines[-1].strip():
        lines.pop()
    return [('code', lines)]


def _convert_literal_block(node: nodes.literal_block) -> list[Segment]:
    """Convert a ``.. code-block::``. Python blocks stay code, others become comments."""
    language = node.get('language', '')
    # Case-insensitive: sphinx-gallery emits ``.. code-block:: Python`` (capitalized).
    if language.lower() in _PYTHON_LANGUAGES:
        lines = [_clean_code_comment(line) for line in node.astext().splitlines()]
        while lines and not lines[-1].strip():
            lines.pop()
        return [('code', lines)] if lines else []
    text = node.astext().strip()
    if not text:
        return []
    comment_lines: list[str] = []
    _add_comment(comment_lines, text)
    return [('text', comment_lines)]


def _convert_admonition(
    node: nodes.Element, label: str, ctx: _RenderContext, *, skip_first_title: bool = False
) -> list[Segment]:
    """Convert an admonition-like container to a ``# LABEL:`` directive segment."""
    inner_ctx = replace(ctx, in_see_also=True) if label.upper() == 'SEE ALSO' else ctx
    inner: list[Segment] = [('text', [f'# {label}:'])]
    for child in node.children:
        if skip_first_title and isinstance(child, nodes.title):
            continue
        inner.extend(_convert_node(child, inner_ctx))
    return [('directive', _join_segments(inner))]


def _convert_node(node: nodes.Node, ctx: _RenderContext) -> list[Segment]:
    """Convert ``node`` into zero or more segments."""
    if any(_has_class(node, css_class) for css_class in _SKIP_SUBTREE_CLASSES):
        return []
    if isinstance(node, _IGNORED_TYPES):
        return []
    if isinstance(node, nodes.doctest_block):
        return _convert_doctest_block(node)
    if isinstance(node, nodes.literal_block):
        return _convert_literal_block(node)
    if type(node) in _ADMONITION_LABELS:
        return _convert_admonition(node, _ADMONITION_LABELS[type(node)], ctx)
    if _is_see_also_section(node):
        # a hand-written "See Also\n--------" heading nests as a full
        # section rather than a flat sibling - treat it like the
        # ``.. seealso::`` directive it's standing in for.
        return _convert_admonition(node, 'SEE ALSO', ctx, skip_first_title=True)
    if isinstance(node, nodes.admonition):
        # generic ``.. admonition:: Custom Title`` - use its own title as the label
        title_node = node.next_node(nodes.title)
        label = title_node.astext().strip() if title_node is not None else 'NOTE'
        return _convert_admonition(node, label, ctx, skip_first_title=True)
    if isinstance(node, _CONTAINER_TYPES):
        segments: list[Segment] = []
        for child in node.children:
            segments.extend(_convert_node(child, ctx))
        return segments

    # Plain text-bearing nodes (paragraphs, etc.) - render inline content,
    # backticking code-like cross-references/literals along the way.
    text = _render_inline(node, ctx).strip()
    if not text:
        return []
    comment_lines: list[str] = []
    _add_comment(comment_lines, text)
    return [('text', comment_lines)]


def _has_real_code(source: str) -> bool:
    """Check whether ``source`` contains at least one executable statement."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    return any(
        not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Constant)
        for stmt in tree.body
    )


def _span_from(parent: nodes.Element, start: int) -> int:
    """Return the end index (exclusive) of a content span starting at ``start``.

    Extends until the next boundary-type node, or to the end of ``parent``'s
    children. A "See Also" heading/section is not a boundary -- otherwise it
    would truncate everything after it.
    """
    end = start
    for i in range(start, len(parent.children)):
        child = parent.children[i]
        if _is_see_also_heading(child) or _is_see_also_section(child):
            end = i + 1
            continue
        if isinstance(child, _BOUNDARY_TYPES):
            break
        end = i + 1
    return end


def _find_external_see_also(parent: nodes.Element, start: int, end: int) -> nodes.Node | None:
    """Find a "See Also" part sited outside the normal ``[start, end)`` span.

    numpydoc's own "See Also" field is canonically reordered to sit before
    "Examples", so it would otherwise never be seen at all.
    """
    for i, child in enumerate(parent.children):
        if start <= i < end:
            continue
        if type(child) in _ADMONITION_LABELS and _ADMONITION_LABELS[type(child)] == 'SEE ALSO':
            return child
    return None


def _examples_spans(doctree: nodes.document) -> list[tuple[nodes.Element, int, int, nodes.Node]]:
    """Find every "Examples" heading's content span.

    Returns a list of ``(parent, start, end, heading)`` tuples.
    """
    spans = []
    for heading in doctree.findall(_is_examples_heading):
        parent = heading.parent
        start = parent.index(heading) + 1
        end = _span_from(parent, start)
        spans.append((parent, start, end, heading))
    return spans


def _qualified_name_for(node: nodes.Node, docname: str, counter: int) -> str:
    """Best-effort identifier used to name the generated file and its title header."""
    ancestor: nodes.Node | None = node.parent
    while ancestor is not None:
        if isinstance(ancestor, addnodes.desc):
            signature = ancestor.next_node(addnodes.desc_signature)
            if signature is not None and signature.get('ids'):
                return signature['ids'][0]
        ancestor = ancestor.parent
    base = Path(docname).name or docname
    return f'{base}-example-{counter}'


def _header_segment(qualified_name: str) -> Segment:
    """Build the title-header segment, e.g. ``# pyvista.read examples`` + underline."""
    title = f'Examples from {qualified_name}'
    underline = '-' * len(title)
    return ('directive', [f'# {title}', f'# {underline}'])


def _strip_comment_prefix(line: str) -> str:
    """Remove the leading ``# `` (or bare ``#``) from a generated comment line."""
    if line == '#':
        return ''
    return line.removeprefix('# ') if line.startswith('# ') else line.removeprefix('#')


def _segments_to_cells(segments: list[Segment]) -> list[tuple[str, list[str]]]:
    """Group segments into notebook cells: consecutive runs become one cell each.

    A run of ``code`` segments becomes a code cell; a run of ``text``/
    ``directive`` segments becomes a markdown cell (same spacing rules as
    ``_join_segments``, with the ``#`` prefix stripped).
    """
    cells: list[tuple[str, list[str]]] = []
    run: list[Segment] = []
    run_kind: str | None = None

    def _flush() -> None:
        if not run:
            return
        joined = _join_segments(run)
        cells.append(
            (
                run_kind,
                [_strip_comment_prefix(line) for line in joined]
                if run_kind == 'markdown'
                else joined,
            )
        )

    for kind, lines in segments:
        if not lines:
            continue
        cell_kind = 'code' if kind == 'code' else 'markdown'
        if cell_kind != run_kind:
            _flush()
            run, run_kind = [], cell_kind
        run.append((kind, lines))
    _flush()
    return cells


def _cell_source(lines: list[str], kind: str) -> list[str]:
    r"""Format lines the way nbformat expects: each ending in ``\\n`` but the last.

    Markdown cells get a trailing hard line break (two spaces) on each
    non-blank line -- otherwise adjacent lines collapse into one paragraph,
    since markdown treats a single newline as plain whitespace.
    """
    if not lines:
        return []
    if kind == 'markdown':
        lines = [f'{line}  ' if line else line for line in lines]
    return [f'{line}\n' for line in lines[:-1]] + [lines[-1]]


def _build_notebook(cells: list[tuple[str, list[str]]]) -> dict:
    """Build a minimal, valid nbformat-4.5 notebook dict from grouped cells."""
    nb_cells = []
    for i, (kind, lines) in enumerate(cells):
        cell: dict = {
            'cell_type': kind,
            'metadata': {},
            'source': _cell_source(lines, kind),
            'id': f'cell-{i}',
        }
        if kind == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
        nb_cells.append(cell)

    return {
        'cells': nb_cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def _write_download_file(app: Sphinx, name: str, extension: str, content: str) -> str:
    """Write generated content directly into the builder's downloads dir.

    Returns the written file's path relative to the downloads directory --
    the value to use as a ``download_reference``'s ``filename``.

    Writes straight to ``<outdir>/_downloads/...`` instead of registering
    through ``env.dlfiles``, because the HTML builder copies those files
    during ``copy_assets()`` -- before any ``doctree-resolved`` handler
    (this one included) runs, so registering here would be too late.
    """
    # 32 hex chars, matching Sphinx's own native _downloads/<digest>/... layout.
    digest = hashlib.sha256(content.encode()).hexdigest()[:32]
    safe_name = name.replace('.', '_') if name else 'example'
    filename = f'{safe_name}.{extension}'
    rel_path = f'{digest}/{filename}'

    out_path = Path(app.outdir) / '_downloads' / digest / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding='utf-8')
    return rel_path


def _write_source(app: Sphinx, name: str, source: str) -> str:
    """Write a generated ``.py`` file; see ``_write_download_file``."""
    return _write_download_file(app, name, 'py', source)


def _write_notebook(app: Sphinx, name: str, notebook: dict) -> str:
    """Write a generated ``.ipynb`` file; see ``_write_download_file``."""
    return _write_download_file(app, name, 'ipynb', json.dumps(notebook, indent=1))


#: Download link text per format, and the fixed order they're offered in
#: regardless of how ``sphinx_examples_as_code_formats`` lists them.
_FORMAT_LABELS = {
    'py': 'Download Python source code',
    'ipynb': 'Download Jupyter notebook',
}
_FORMAT_ORDER = ('py', 'ipynb')


def _make_download_node(entries: list[tuple[str, str]]) -> nodes.paragraph:
    """Build one paragraph holding a download link for each ``(label, rel_path)`` entry."""
    paragraph = nodes.paragraph()
    for i, (label, rel_path) in enumerate(entries):
        if i > 0:
            paragraph += nodes.Text(' | ')
        reference = addnodes.download_reference('', reftarget=rel_path)
        reference['filename'] = rel_path
        reference += nodes.Text(label)
        paragraph += reference
    return paragraph


def _build_segments(nodes_in_span: list[nodes.Node], ctx: _RenderContext) -> list[Segment]:
    """Convert a span's nodes into segments.

    A bare ``.. rubric:: See Also`` heading isn't wrapped in a container the
    way ``.. seealso::`` or a nested section are, so this gathers everything
    after it into one merged directive segment instead.
    """
    segments: list[Segment] = []
    for i, node in enumerate(nodes_in_span):
        if isinstance(node, nodes.rubric) and _is_see_also_heading(node):
            inner_ctx = replace(ctx, in_see_also=True)
            inner: list[Segment] = [('text', ['# SEE ALSO:'])]
            for later_node in nodes_in_span[i + 1 :]:
                inner.extend(_convert_node(later_node, inner_ctx))
            segments.append(('directive', _join_segments(inner)))
            break
        segments.extend(_convert_node(node, ctx))
    return segments


def _build_download_entries(
    app: Sphinx,
    docname: str,
    name: str,
    nodes_in_span: list[nodes.Node],
    formats: list[str],
) -> list[tuple[str, str]]:
    """Convert a span of nodes into written ``.py``/``.ipynb`` files, per ``formats``.

    Returns ``(label, rel_path)`` entries for whichever formats are
    requested -- empty if the span has no real code, or ``formats`` itself
    is empty. Shared by the docstring-Examples path and gallery-page path;
    the only difference between them is how ``nodes_in_span`` and ``name``
    get built.
    """
    py_ctx = _RenderContext(app=app, docname=docname, fmt='py')
    py_segments = _build_segments(nodes_in_span, py_ctx)

    if not any(kind == 'code' for kind, _lines in py_segments):
        return []

    source = '\n'.join(_join_segments([_header_segment(name), *py_segments])).rstrip() + '\n\n'

    if not _has_real_code(source):
        return []

    entries = []
    for fmt in _FORMAT_ORDER:
        if fmt not in formats:
            continue
        label = _FORMAT_LABELS[fmt]
        if fmt == 'py':
            rel_path = _write_source(app, name, source)
        else:
            ipynb_ctx = _RenderContext(app=app, docname=docname, fmt='ipynb')
            ipynb_segments = _build_segments(nodes_in_span, ipynb_ctx)
            cells = _segments_to_cells([_header_segment(name), *ipynb_segments])
            rel_path = _write_notebook(app, name, _build_notebook(cells))
        entries.append((label, rel_path))

    return entries


def _process_span(
    app: Sphinx,
    docname: str,
    parent: nodes.Element,
    start: int,
    end: int,
    heading: nodes.Node,
    counter: int,
    position: str,
    formats: list[str],
) -> None:
    """Convert one Examples span and insert download link(s) if it has real code."""
    nodes_in_span = list(parent.children[start:end])
    external_see_also = _find_external_see_also(parent, start, end)
    if external_see_also is not None:
        nodes_in_span.append(external_see_also)

    name = _qualified_name_for(heading, docname, counter)
    entries = _build_download_entries(app, docname, name, nodes_in_span, formats)
    if not entries:
        return

    download_node = _make_download_node(entries)
    parent.insert(start if position == 'top' else end, download_node)


def _process_doctree(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Add a download link to every "Examples" section found on this page."""
    if not getattr(app.builder, 'download_support', False):
        # Only HTML-family builders serve a _downloads/ directory - skip
        # everything else (latex, text, man, epub, ...).
        return

    position = app.config.sphinx_examples_as_code_link_position
    formats = app.config.sphinx_examples_as_code_formats

    # Process spans per shared parent, last to first: inserting a download
    # node shifts every later sibling index by one, so this stays correct
    # even with multiple Examples headings under one parent.
    spans = _examples_spans(doctree)
    numbered_spans = [(*span, i + 1) for i, span in enumerate(spans)]
    for parent, start, end, heading, counter in sorted(
        numbered_spans, key=lambda s: (id(s[0]), -s[1])
    ):
        _process_span(app, docname, parent, start, end, heading, counter, position, formats)


def _validate_base_url(_app: Sphinx, config: Config) -> None:
    """Validate and normalize ``sphinx_examples_as_code_base_url``.

    Catches two common typos loudly instead of silently generating wrong
    links: a missing scheme (parses with no netloc at all), and a subpath
    with no trailing slash (``urljoin`` would drop the last segment).
    """
    base_url = config.sphinx_examples_as_code_base_url
    if not base_url:
        return

    parsed = urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        msg = (
            f'sphinx_examples_as_code_base_url={base_url!r} does not look like a '
            "valid absolute URL (expected something like 'https://docs.example.com/')."
        )
        raise ConfigError(msg)

    if not base_url.endswith('/'):
        config.sphinx_examples_as_code_base_url = base_url + '/'


def setup(app: Sphinx) -> dict:  # numpydoc ignore=RT01
    """Register the extension."""
    app.connect('doctree-resolved', _process_doctree)
    app.connect('config-inited', _validate_base_url)
    app.add_config_value('sphinx_examples_as_code_link_position', 'top', 'env')
    app.add_config_value('sphinx_examples_as_code_formats', ['py', 'ipynb'], 'env')
    app.add_config_value('sphinx_examples_as_code_base_url', None, 'env')

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
