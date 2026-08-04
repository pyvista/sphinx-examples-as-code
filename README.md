# sphinx-examples-as-code

A Sphinx extension that turns docstring/page "Examples" sections into downloadable,
runnable `.py` and/or `.ipynb` files, with a download link inserted into the section.

Pages or docstrings without an Examples section are left completely untouched. Adding
`sphinx_examples_as_code` to `conf.py`'s `extensions` is the only on/off switch.

## Installation

```bash
pip install sphinx-examples-as-code
```

Add it to your Sphinx `conf.py`:

```python
extensions = [
    ...,
    'sphinx_examples_as_code',
]
```

## Configuration

Set these in `conf.py`:

- `sphinx_examples_as_code_link_position`: where the download link(s) land within the
  Examples section. `'top'` (default) or `'bottom'`.
- `sphinx_examples_as_code_formats`: which downloads to generate. A list containing
  `'py'`, `'ipynb'`, or both (default). Always offered in that order regardless of how
  the list is written.
- `sphinx_examples_as_code_base_url`: the site's published base URL (e.g.
  `'https://docs.pyvista.org/'`), used to turn cross-references into absolute links a
  downloaded, standalone file can actually use. `None` (default) means no links are
  generated anywhere. A missing trailing slash is added automatically; a value with no
  scheme or host raises a configuration error at build start.

## Conversion rules

What happens to the content of an Examples section:

- Doctest blocks (`>>> ...` / `... ...`) keep their input lines, prompts stripped, as
  real Python source. Doctest *output* lines are dropped — only the input code matters.
- `.. code-block:: python` (or `py`) blocks are kept as-is; other languages become
  comments.
- Admonitions (`.. note::`, `.. warning::`, `.. seealso::`, ...) become a `# LABEL:`
  comment followed by their content as comments. "See Also" is recognized in any of its
  three forms (`.. seealso::`, a bare `.. rubric:: See Also`, or a hand-written `See
  Also` heading) and always renders the same way.
- Cross-references and inline code (`:class:`, `:meth:`, `:func:`, `:attr:`,
  double-backtick literals, ...) keep their display text, wrapped in backticks (e.g.
  ``:class:`pyvista.Plotter` `` -> `` `pyvista.Plotter` ``). If `..._base_url` is set and
  the reference resolves: `.ipynb` turns it into a clickable link everywhere; `.py` only
  writes the link inside a "See Also" part (as `name url` on its own line) — everywhere
  else in `.py` the link is simply omitted.
- Plain prose-style references (`:ref:`, `:doc:`) are treated the same way, minus the
  backticks.
- Everything else text-bearing (prose, captions, other non-Python code) becomes a plain
  `#` comment.
- Figures/images, raw HTML, and sphinx-design dropdowns/tab-sets are dropped entirely.

Generated `.py` files start with a `# Examples from <qualified name>` title header and
follow a few whitespace conventions so the result reads like normal Python: prose
directly above a code block stays attached to it, a code block is always followed by a
blank line, and a directive (header, `# NOTE:`-style block) gets blank lines on both
sides.

Generated `.ipynb` notebooks use the same content, split into alternating code/markdown
cells instead.

A download link is only added if the resulting code contains at least one real
executable statement.

## Development

```bash
uv sync --group dev
uv run pytest
uv run pre-commit run --all-files
```
