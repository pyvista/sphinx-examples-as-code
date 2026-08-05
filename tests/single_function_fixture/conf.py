"""Sphinx config reproducing a real-world numpydoc section-hoisting setup.

numpydoc renders docstring sections (Notes, Examples, ...) as
``.. rubric::`` by default, which never becomes a real docutils section --
so ``_qualified_name_for``'s plain ancestor walk (desc_content -> desc)
always finds the object's own ``desc`` node directly, no special handling
needed. Some projects (pyvista's real docs included -- see its
``doc/source/conf.py``) override this to get real headings for the page's
"on this page" navbar, then hoist those sections out of the autodoc
``desc`` node to page level so Sphinx's TocTreeCollector can see them. That
lands "Examples" as a *sibling* of the object's own ``desc``, not an
ancestor of its own heading -- the exact structure that produced a
nonsensical ``<docname>-example-1`` header in production for a page that
unambiguously documents one function.

Reproduced here (rather than in ``tinypages/conf.py``) since it changes
docstring-section rendering globally for the build it's applied to, and
``tinypages`` is shared by many other tests that already pin exact output
for the un-hoisted (rubric-based) case.
"""

from __future__ import annotations

from pathlib import Path
import sys

from docutils import nodes
from numpydoc.docscrape_sphinx import SphinxDocString
from sphinx import addnodes

sys.path.insert(0, str(Path(__file__).parent))

extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx_examples_as_code',
]

root_doc = 'index'
project = 'single_function_fixture'
exclude_patterns = ['_build']
numpydoc_show_class_members = False


def _str_header(self, name):
    return [name, '-' * len(name), '']


SphinxDocString._str_header = _str_header


def _is_nested_desc(node: nodes.Node) -> bool:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, addnodes.desc):
            return True
        parent = parent.parent
    return False


def _hoist_docstring_sections(app, doctree) -> None:
    """Move docstring sections out of their ``desc`` node to page level."""
    for desc in list(doctree.findall(addnodes.desc)):
        if _is_nested_desc(desc):
            continue
        parent = desc.parent
        if parent is None:
            continue
        # Only hoist when this object owns the page -- otherwise sections
        # from several objects would collide at page level.
        if len([node for node in parent if isinstance(node, addnodes.desc)]) != 1:
            continue
        content = next((node for node in desc if isinstance(node, addnodes.desc_content)), None)
        if content is None:
            continue
        sections = [node for node in content if isinstance(node, nodes.section)]
        index = parent.index(desc)
        for offset, section in enumerate(sections):
            content.remove(section)
            parent.insert(index + 1 + offset, section)


def setup(app):
    app.connect('doctree-read', _hoist_docstring_sections)
