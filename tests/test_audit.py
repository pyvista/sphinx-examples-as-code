"""Tests for the download audit: ``sphinx_examples_as_code.audit`` and its script runner."""

from __future__ import annotations

import ast
import html
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap

import pytest

import sphinx_examples_as_code as seac
from sphinx_examples_as_code import _audit_runner as runner
from sphinx_examples_as_code import audit

HEADER = '# /// script\n# dependencies = [\n#     "packaging",\n# ]\n# ///\n'
TRY_PYTEST = 'try:\n    import pytest\nexcept ImportError:\n'


def _notebook(*cells: str) -> str:
    """Return a notebook with one code cell per source."""
    code = [
        {
            'cell_type': 'code',
            'execution_count': None,
            'id': f'c{i}',
            'metadata': {},
            'outputs': [],
            'source': source,
        }
        for i, source in enumerate(cells)
    ]
    kernel = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook = {
        'cells': code,
        'metadata': {'kernelspec': kernel},
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    return json.dumps(notebook)


def _marimo(*cells: str) -> str:
    """Return a marimo notebook with one cell per source."""
    body = ''.join(
        f'@app.cell\ndef _():\n{textwrap.indent(source, "    ")}\n    return\n\n\n'
        for source in cells
    )
    main = 'if __name__ == "__main__":\n    app.run()\n'
    return f'import marimo\n\napp = marimo.App()\n\n\n{body}{main}'


def _page(groups: dict[str, str]) -> str:
    """Return an HTML page with a download link group and a doctest block per entry."""
    parts = []
    for links, doctest_text in groups.items():
        anchors = ' | '.join(
            f'<a class="reference download internal" download="" href="_downloads/{link}">x</a>'
            for link in links.split()
        )
        block = (
            '<div class="doctest highlight-default notranslate"><div class="highlight"><pre>'
            f'<span class="gp">{html.escape(doctest_text)}</span></pre></div></div>'
        )
        parts.append(f'<p>{anchors}</p>\n{block}')
    return '<html><body>' + '\n'.join(parts) + '</body></html>'


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """Write a small HTML tree with downloads covering every finding kind."""
    downloads = tmp_path / '_downloads'
    files = {
        'a/good.py': HEADER + 'x = 1\nprint(repr(x))\nprint(repr((x, 2)))\n',
        'a/good.ipynb': _notebook('x = 1\nx', '(x, 2)'),
        'a/good_marimo.py': _marimo('x = 1\nx').replace(
            '    return\n', '    return (x,)\n\n\n@app.cell\ndef _(x):\n    (x, 2)\n    return\n'
        ),
        'b/dropped.py': 'x = 1\nx\n',
        'b/collapsed.ipynb': _notebook('x = 1\nx\nx + 1'),
        'b/wrong.ipynb': _notebook('x = 1\nx'),
        'c/prompt.py': '>>> x = 1\n',
        'c/percent.ipynb': _notebook('%matplotlib inline'),
        'c/magic_marimo.py': _marimo(
            '# magic command not supported in marimo; please file an issue'
        ),
        'c/twice_marimo.py': _marimo('y = 1', 'y = 2'),
        'c/rename_marimo.py': _marimo('_y += 1'),
        'd/undeclared.py': HEADER + 'import pytest\n',
        'd/probe.py': HEADER + 'import importlib.util\nprint(importlib.util.find_spec("pytest"))\n',
        'd/hint.py': HEADER
        + 'import importlib.util\nif not importlib.util.find_spec("pytest"):\n'
        + '    raise ImportError("pip install pytest")\n',
        'd/swallowed.py': HEADER
        + TRY_PYTEST
        + '    pytest = None\nif pytest is None:\n    raise ValueError("no backend")\n',
        'd/optional.py': HEADER + TRY_PYTEST + '    pass\n',
        'd/fails.py': 'raise ValueError("boom")\n',
        'd/fails.ipynb': _notebook('raise ValueError("boom")'),
        'e/writes.ipynb': _notebook('open("out.txt", "w").close()'),
    }
    for name, text in files.items():
        path = downloads / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    page = _page(
        {
            'a/good.py a/good.ipynb a/good_marimo.py': '>>> x = 1\n>>> x\n1\n>>> (x, 2)\n(1, 2)\n',
            'b/dropped.py': '>>> x = 1\n>>> x\n1\n',
            'b/collapsed.ipynb': '>>> x = 1\n>>> x\n1\n>>> x + 1\n2\n',
            'b/wrong.ipynb': '>>> x = 1\n>>> x\n2\n',
        }
    )
    (tmp_path / 'index.html').write_text(page)
    (tmp_path / 'gallery.html').write_text('<div class="sphx-glr-timing"></div>')
    return tmp_path


def _run(site: Path, fmt: str, tmp_path: Path) -> dict[str, list[dict]]:
    """Audit one format of ``site`` and return each file's findings."""
    output = tmp_path / f'{fmt}.jsonl'
    audit.main(
        [
            'run',
            str(site / '_downloads'),
            '--format',
            fmt,
            '--output',
            str(output),
            '--workers',
            '4',
        ]
    )
    results = [json.loads(line) for line in output.read_text().splitlines()]
    return {r['file']: r for r in results}


def _kinds(result: dict) -> list[str]:
    return sorted(f['kind'] for f in result['findings'])


def test_run_python(site: Path, tmp_path: Path):
    output = tmp_path / 'python.jsonl'
    audit.main(
        [
            'run',
            str(site / '_downloads'),
            '--format',
            'python',
            '--output',
            str(output),
            '--html',
            str(site),
        ]
    )
    results = {r['file']: r for r in map(json.loads, output.read_text().splitlines())}
    assert set(results) == {'a/good.py', 'b/dropped.py'}
    assert results['a/good.py']['findings'] == []
    assert results['a/good.py']['compared'] == 2
    assert _kinds(results['b/dropped.py']) == ['dropped-output']

    results = _run(site, 'python', tmp_path)
    assert _kinds(results['c/prompt.py']) == ['error', 'syntax']
    assert _kinds(results['d/undeclared.py']) == ['dependency']
    assert results['d/undeclared.py']['findings'][0]['message'] == "'pytest' from pytest"
    assert results['d/probe.py']['findings'] == []
    assert _kinds(results['d/hint.py']) == ['dependency']
    assert results['d/swallowed.py']['findings'] == [
        {'kind': 'dependency', 'message': "'pytest' from pytest"}
    ]
    assert results['d/optional.py']['findings'] == []
    assert _kinds(results['d/fails.py']) == ['error']

    output = tmp_path / 'excluded.jsonl'
    downloads = str(site / '_downloads')
    args = ['run', downloads, '--format', 'python', '--output', str(output)]
    audit.main([*args, '--exclude', 'fail*', '?ndeclared'])
    names = {json.loads(line)['name'] for line in output.read_text().splitlines()}
    assert names == {'good', 'dropped', 'prompt', 'probe', 'hint', 'swallowed', 'optional'}


@pytest.mark.skipif(importlib.util.find_spec('nbconvert') is None, reason='needs nbconvert')
def test_run_jupyter(site: Path, tmp_path: Path):
    output = tmp_path / 'jupyter.jsonl'
    audit.main(
        [
            'run',
            str(site / '_downloads'),
            '--format',
            'jupyter',
            '--output',
            str(output),
            '--html',
            str(site),
        ]
    )
    results = {r['file']: r for r in map(json.loads, output.read_text().splitlines())}
    assert results['a/good.ipynb']['findings'] == []
    assert results['a/good.ipynb']['compared'] == 2
    collapsed = results['b/collapsed.ipynb']
    assert _kinds(collapsed) == ['collapsed']
    assert collapsed['compared'] == 1
    (mismatch,) = results['b/wrong.ipynb']['findings']
    assert mismatch == {'kind': 'mismatch', 'message': "x -> '1', expected '2'", 'statement': 'x\n'}

    results = _run(site, 'jupyter', tmp_path)
    assert _kinds(results['c/percent.ipynb']) == ['syntax']
    assert _kinds(results['d/fails.ipynb']) == ['error']
    assert 'ValueError: boom' in results['d/fails.ipynb']['findings'][0]['message']
    assert results['e/writes.ipynb']['findings'] == []
    assert not (site / '_downloads' / 'e' / 'out.txt').exists()


@pytest.mark.skipif(importlib.util.find_spec('marimo') is None, reason='needs marimo')
def test_run_marimo(site: Path, tmp_path: Path):
    output = tmp_path / 'marimo.jsonl'
    audit.main(
        [
            'run',
            str(site / '_downloads'),
            '--format',
            'marimo',
            '--output',
            str(output),
            '--html',
            str(site),
            '--timeout',
            '120',
        ]
    )
    results = {r['file']: r for r in map(json.loads, output.read_text().splitlines())}
    assert results['a/good_marimo.py']['findings'] == []
    assert results['a/good_marimo.py']['compared'] == 2

    results = _run(site, 'marimo', tmp_path)
    assert 'magic' in _kinds(results['c/magic_marimo.py'])
    assert 'marimo-check' in _kinds(results['c/twice_marimo.py'])
    assert _kinds(results['c/rename_marimo.py']) == ['marimo-rename']
    assert not list((site / '_downloads').rglob('__marimo__'))


def test_pages(site: Path, tmp_path: Path):
    output = tmp_path / 'pages.jsonl'
    audit.main(['pages', str(site), '--output', str(output)])
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(r['file'], r['findings'][0]['message']) for r in results] == [
        ('gallery.html', 'no jupyter, marimo, python'),
        ('index.html', 'no jupyter, marimo'),
        ('index.html', 'no marimo, python'),
        ('index.html', 'no marimo, python'),
    ]


def test_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    results = [
        {
            'file': 'a/x.py',
            'name': 'x',
            'format': 'python',
            'compared': 1,
            'findings': [{'kind': 'error', 'message': 'Traceback\nValueError: x'}],
        },
        {
            'file': 'a/x.ipynb',
            'name': 'x',
            'format': 'jupyter',
            'compared': 0,
            'findings': [{'kind': 'error', 'message': 'ValueError: x'}],
        },
        {
            'file': 'b/y.ipynb',
            'name': 'y',
            'format': 'jupyter',
            'compared': 3,
            'findings': [
                {'kind': 'error', 'message': 'KeyError: | `y`'},
                {'kind': 'collapsed', 'message': 'y' * 300},
                {'kind': 'mismatch', 'message': 'a -> 1', 'statement': 'a\n'},
                {'kind': 'mismatch', 'message': 'b -> 1', 'statement': 'b\n'},
            ],
        },
        {
            'file': 'b/y.py',
            'name': 'y',
            'format': 'python',
            'compared': 3,
            'findings': [{'kind': 'mismatch', 'message': 'a -> 1', 'statement': 'a\n'}],
        },
        {
            'file': 'c/z.py',
            'name': 'z',
            'format': 'python',
            'findings': [{'kind': 'dependency', 'message': "'q' from q"}] * 2,
        },
        {
            'file': 'c/z_marimo.py',
            'name': 'z',
            'format': 'marimo',
            'findings': [{'kind': 'error', 'message': 'ModuleNotFoundError: q'}],
        },
    ]
    jsonl = tmp_path / 'results.jsonl'
    jsonl.write_text(''.join(json.dumps(r) + '\n' for r in results))
    github_output = tmp_path / 'github_output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    report = tmp_path / 'report.md'
    audit.main(['report', str(jsonl), '--output', str(report), '--max-rows', '1'])
    text = report.read_text()
    assert '| jupyter | 2 | 2 | 3 |' in text
    assert 'Fails as a notebook while the plain Python script runs (1)\n\n| Download' in text
    assert "`KeyError: \\| 'y'`" in text
    assert 'Fails as a plain Python script (3)' in text
    assert '...and 1 more.' in text
    assert '<details>' in text
    assert f'`{"y" * 197}...`' in text
    table = '| Download | Format | Detail |\n| --- | --- | --- |\n'
    assert f"while the script's match (1)\n\n{table}| `y.ipynb` | jupyter | `b -> 1` |" in text
    assert 'in the script too (2)' in text
    assert github_output.read_text() == 'failures=4\n'

    allow = ['error', 'collapsed', 'mismatch', 'dependency', 'python-mismatch', 'python-error']
    monkeypatch.delenv('GITHUB_OUTPUT')
    audit.main(['report', str(jsonl), '--output', str(report), '--allow', *allow])
    assert report.read_text().endswith('No failures.\n')
    assert github_output.read_text() == 'failures=4\n'


def test_report_matches_scripts_within_a_link_group(tmp_path: Path):
    error = [{'kind': 'error', 'message': 'ValueError: x'}]
    results = [
        {'file': 'a/x.py', 'name': 'x', 'format': 'python', 'group': 'a.html#0', 'findings': error},
        {
            'file': 'a/x.ipynb',
            'name': 'x',
            'format': 'jupyter',
            'group': 'a.html#0',
            'findings': error,
        },
        {
            'file': 'b/x.ipynb',
            'name': 'x',
            'format': 'jupyter',
            'group': 'b.html#0',
            'findings': error,
        },
    ]
    jsonl = tmp_path / 'results.jsonl'
    jsonl.write_text(''.join(json.dumps(r) + '\n' for r in results))
    report = tmp_path / 'report.md'
    audit.main(['report', str(jsonl), '--output', str(report)])
    text = report.read_text()
    assert 'Fails as a notebook while the plain Python script runs (1)' in text
    assert 'Fails as a plain Python script (2)' in text


def test_run_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    for i in range(100):
        (tmp_path / f'e{i}.py').write_text('')
    monkeypatch.setattr(audit, 'execute', lambda *_: {'findings': []})
    audit.main(['run', str(tmp_path), '--format', 'python', '--output', str(tmp_path / 'r')])
    assert '100/100 done, 0 with findings' in capsys.readouterr().out


def test_entry_points(site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / 'pages.jsonl'
    monkeypatch.setattr(sys, 'argv', ['audit', 'pages', str(site), '--output', str(output)])
    monkeypatch.delitem(sys.modules, 'sphinx_examples_as_code.audit')
    runpy.run_module('sphinx_examples_as_code.audit', run_name='__main__')
    assert output.exists()
    script = tmp_path / 's.py'
    script.write_text('x = 1\n')
    monkeypatch.setattr(sys, 'argv', ['runner', str(script), str(tmp_path / 'out.json')])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(runner.__file__, run_name='__main__')
    assert exit_info.value.code == 0


@pytest.mark.parametrize(
    ('want', 'got', 'same'),
    [
        ('0.51825\n', '0.5182557447749919\n', True),
        ('0.51825\n', '0.5183\n', False),
        ('[0.0, 1.0, ... 14.0]\n', '[0.0, 1.0, 2.0, 14.0]\n', True),
        ('(1.0, 10.0)\n', '[1.0, 10.0]\n', True),
        ("{'a': (1, 2)}\n", "{'a': [1, 2]}\n", True),
        ('array([1, 2])\n', 'array([1,\n       2])\n', True),
        ('x\n', 'y\n', False),
        ("(<class 'a.B'>, None)\n", '(a.B, None)\n', True),
        ('(np.uint8(0), np.uint8(117))\n', '[0, 117]\n', True),
        ('True\n', '1\n', False),
        ('[False]\n', '[0.0]\n', False),
        ('(True, 1)\n', '[True, 1]\n', True),
    ],
)
def test_same_output(want: str, got: str, same: bool):
    assert audit.same_output(want, got) is same


def test_output_text():
    stream = {'output_type': 'stream', 'name': 'stdout', 'text': ['a\n']}
    stderr = {'output_type': 'stream', 'name': 'stderr', 'text': ['warning\n']}
    plain = {'output_type': 'execute_result', 'data': {'text/plain': ['1']}}
    json_value = {'output_type': 'display_data', 'data': {'application/json': '[1, 2]'}}
    json_list = {'output_type': 'display_data', 'data': {'application/json': [1, 2]}}
    pre = {'output_type': 'display_data', 'data': {'text/html': "<pre class='x'>a &lt; b</pre>"}}
    image = {'output_type': 'display_data', 'data': {'image/png': 'abc', 'text/plain': '<Image>'}}
    other = {'output_type': 'display_data', 'data': {'application/pdf': 'abc'}}
    rich = {'output_type': 'display_data', 'data': {'text/html': '<table></table>'}}
    rich_json = {'output_type': 'display_data', 'data': {'application/json': ['image/png:a']}}
    assert (
        audit.output_text([stream, stderr, plain, json_value, json_list, pre, image, other])
        == 'a\n1\n[1, 2]\n[1, 2]\na < b\n'
    )
    assert audit.output_text([rich]) is None
    assert audit.output_text([rich_json]) is None
    assert audit.notebook_units(Path('missing.ipynb')) == []
    assert audit.notebook_error(Path('missing.ipynb')) == ''
    assignment = ast.parse('x = 1').body[0]
    assert not audit.is_value(assignment)
    assert audit.printed(assignment) is assignment
    assert audit.output_text([{'output_type': 'error'}]) is None


@pytest.mark.parametrize(
    ('value', 'shown'),
    [
        (['text/plain+float:0.5', 'text/plain:x', 'plain'], "[0.5, x, 'plain']"),
        ({'a': [True, None]}, "{'a': [True, None]}"),
        (['text/html:<div></div>'], None),
        ({'a': 'image/png:abc'}, None),
    ],
)
def test_json_repr(value: object, shown: str | None):
    assert audit.json_repr(value) == shown


def test_expectations_skip_blocks_before_any_link(tmp_path: Path):
    page = '<div class="highlight-pycon"><div class="highlight"><pre>&gt;&gt;&gt; 1\n1\n</pre>'
    page += _page({'a/x.py': '>>> import os\n>>> 2\n2\n'})
    (tmp_path / 'p.html').write_text(page)
    assert audit.expectations(tmp_path) == {'a/x.py': ('p.html#0', [audit.Expected('2\n', '2\n')])}


def test_expectations_restore_blanklines(tmp_path: Path):
    page = _page({'a/x.py': '>>> x\na\n\nb\n>>> y\n1\n\n>>> z\n2\n\n'})
    (tmp_path / 'p.html').write_text(page)
    [_, examples] = audit.expectations(tmp_path)['a/x.py']
    assert [ex.want for ex in examples] == ['a\n<BLANKLINE>\nb\n', '1\n', '2\n']
    assert audit.same_output(examples[0].want, 'a\n\nb\n')
    script = 'def f():\n    """\n    >>> f()\n    1\n\n    """\n\n\nx = 1\n'
    assert audit.restore_blanklines(script) == script


def test_commented_statements():
    source = '# Some prose.\n# f(\n#     a,\n# )\nx = 1\n# %%\n#\n'
    [node] = audit.commented_statements(source)
    assert ast.unparse(node) == 'f(a)'
    assert node.lineno == 2
    assert audit.is_skipped(node)
    assert [audit.is_skipped(n) for n in audit.with_skipped(source)] == [True, False]
    assert audit.with_skipped('x +') is None


def test_skipped_statements_are_not_compared(tmp_path: Path):
    expected = [audit.Expected('x\n', '1\n'), audit.Expected('x\n', '2\n')]
    script = tmp_path / 'x.py'
    script.write_text('x = 2\n# x\nprint(repr((x)))\n')
    units = audit.python_units(script, [{'line': 3, 'stdout': '2\n'}])
    assert audit.compare(expected, units, 'python') == ([], 1)
    units = [(audit.with_skipped('x = 2\n# x'), ''), (audit.with_skipped('x\n# y'), '2\n')]
    assert audit.compare(expected, units, 'jupyter') == ([], 1)


def test_page_groups_bottom_links():
    block = '<div class="highlight-pycon"><div class="highlight"><pre>&gt;&gt;&gt; {0}\n{0}\n</pre>'
    link = '<p><a class="reference download internal" href="../_downloads/{0}">x</a></p>'
    text = block.format(1) + link.format('a.py') + block.format(2) + link.format('b.py')
    assert audit.page_groups(text, 'bottom') == [
        (['a.py'], ['>>> 1\n1\n']),
        (['b.py'], ['>>> 2\n2\n']),
    ]
    assert audit.page_groups(text) == [(['a.py'], ['>>> 2\n2\n']), (['b.py'], [])]


def test_run_without_downloads(tmp_path: Path):
    output = tmp_path / 'r.jsonl'
    audit.main(['run', str(tmp_path), '--format', 'marimo', '--output', str(output)])
    [record] = [json.loads(line) for line in output.read_text().splitlines()]
    assert record['findings'] == [{'kind': 'no-downloads', 'message': 'no marimo downloads'}]


def test_marimo_renames_still_match():
    expected = [audit.Expected('mesh\n', '1\n'), audit.Expected('mesh_2\n', 'Mesh()\n')]
    units = [(audit.statements('mesh_1 = f()\n_mesh_1'), '1\n'), (audit.statements('mesh_2'), '')]
    assert audit.compare(expected, units, 'marimo') == ([], 1)


def test_unmatched_and_uncomparable_values_are_skipped():
    expected = [
        audit.Expected('a\n', '1\n'),
        audit.Expected('b\n', '1\n'),
        audit.Expected('print(1)\n', '1\n'),
        audit.Expected('1 +\n', '1\n'),
    ]
    units = [(audit.statements('b'), None), (audit.statements('print(1)\nc = 2'), '1\n')]
    assert audit.compare(expected, units, 'jupyter') == ([], 0)


def test_static_findings_ignore_valid_code(tmp_path: Path):
    path = tmp_path / 'ok.py'
    path.write_text('s = """\n>>> not code\n% not a magic\n"""\n')
    assert audit.static_findings(path, 'python') == []


@pytest.mark.parametrize('fmt', ['python', 'marimo', 'jupyter'])
def test_static_findings_requires_python(tmp_path: Path, fmt: str):
    def download(spec: str) -> str:
        if fmt == 'jupyter':
            run_line = seac._jupyter_run_line(['numpy'], spec, 'x.ipynb')
            cells = seac._segments_to_cells([('text', [f'# {run_line}']), ('code', ['x'])])
            return json.dumps(seac._build_notebook(cells))
        return seac._script_metadata_header([], spec, 'run')

    path = tmp_path / 'x'
    spec = f'>=3.8,<{sys.version_info.major}.{sys.version_info.minor}'
    path.write_text(download(spec))
    message = f'{spec} excludes Python {audit.PYTHON}'
    assert audit.static_findings(path, fmt) == [{'kind': 'requires-python', 'message': message}]
    path.write_text(download('>=3.8'))
    assert audit.static_findings(path, fmt) == []


@pytest.mark.parametrize(
    ('fmt', 'text'),
    [
        ('python', 'x = 1\n'),
        ('marimo', '# /// script\n# dependencies = []\n# ///\n'),
        ('jupyter', _notebook('x')),
    ],
)
def test_requires_python_absent(fmt: str, text: str):
    assert audit.requires_python(text, fmt) is None


def test_stem_and_format():
    assert audit.stem(Path('a_marimo.py')) == 'a'
    assert audit.stem(Path('a.ipynb')) == 'a'
    assert audit.stem(Path('a.txt')) == 'a'
    assert audit.fmt_of('a.txt') is None


def test_timeout(tmp_path: Path):
    path = tmp_path / 'slow.py'
    path.write_text('import time\ntime.sleep(10)\n')
    result = audit.execute(path, 'python', 0.5, [])
    assert result['findings'] == [{'kind': 'timeout', 'message': 'run after 0.5 s'}]


def test_runner_hides_its_own_imports(tmp_path: Path):
    path = tmp_path / 's.py'
    path.write_text('# /// script\n# dependencies = []\n# ///\nimport packaging.version\n')
    findings = audit.execute(path, 'python', 60, [])['findings']
    assert [f['kind'] for f in findings] == ['dependency']


@pytest.mark.parametrize('hide', [False, True])
def test_runner_in_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, hide: bool):
    script = tmp_path / 's.py'
    script.write_text(HEADER + 'print("a")\nx = 1\nimport seac_local\nprint(x)\nimport pytest\n')
    (tmp_path / 'seac_local.py').write_text('')
    output = tmp_path / 'out.json'
    argv = ['runner', str(script), str(output), *(['--hide', 'pytest'] if hide else [])]
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'pytest')
    for name in ('allowed', 'hide'):
        monkeypatch.setattr(runner.Guard, name, set())
    for name in ('providers', 'hidden', 'seen'):
        monkeypatch.setattr(runner.Guard, name, {})
    monkeypatch.setattr(runner.Guard, 'direct', set())
    assert runner.main() == int(hide)
    records = [{'line': 6, 'stdout': 'a\n'}, {'line': 7, 'stdout': ''}, {'line': 8, 'stdout': ''}]
    records.append({'line': 9, 'stdout': '1\n'})
    if not hide:
        records.append({'line': 10, 'stdout': ''})
    assert json.loads(output.read_text()) == records
    err = capsys.readouterr().err
    assert (f"{runner.BLOCKED} of 'pytest' from pytest" in err) is hide
    seen = {} if hide else {'pytest': 'pytest'}
    assert audit.undeclared(err)[1:] == (seen, set(seen))


def test_runner_baseline(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['runner', '--baseline', 'seac-fake', 'packaging'])
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    monkeypatch.delitem(sys.modules, 'pytest')
    monkeypatch.setattr(runner.Guard, 'seen', {})
    providers = {'pytest': ['pytest'], 'seac_fake_mod': ['seac-fake'], 'seac_broken': ['seac-fake']}
    monkeypatch.setattr(runner.importlib.metadata, 'packages_distributions', lambda: providers)
    real_import = __import__

    asked = []

    def fake_import(name, *args, **kwargs):
        asked.append(name)
        if name == 'seac_fake_mod':
            return real_import('pytest')
        if name == 'seac_broken':
            raise RuntimeError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', fake_import)
    assert runner.main() == 0
    assert 'pytest' not in asked
    assert audit.undeclared(capsys.readouterr().err)[1:] == ({'pytest': 'pytest'}, set())


class _Everything:
    """Contain every module name."""

    def __contains__(self, name: object) -> bool:
        """Return True."""
        return True


@pytest.mark.parametrize(('known', 'kinds'), [(True, []), (False, ['dependency'])])
def test_library_lookups_the_baseline_explains_are_not_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, known: bool, kinds: list[str]
):
    path = tmp_path / 's.py'
    library = "exec(compile('import pytest', 'library.py', 'exec'))"
    path.write_text(HEADER + f'try:\n    {library}\nexcept ImportError:\n    raise ValueError\n')
    if known:
        monkeypatch.setattr(audit, 'baseline', lambda deps, timeout: _Everything())
    assert _kinds(audit.execute(path, 'python', 60, [])) == kinds


def test_direct_import_is_a_suspect_despite_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / 's.py'
    path.write_text(HEADER + 'import pytest\nprint("x", end="")\n')
    monkeypatch.setattr(audit, 'baseline', lambda deps, timeout: _Everything())
    findings = audit.execute(path, 'python', 60, [])['findings']
    assert findings == [{'kind': 'dependency', 'message': "'pytest' from pytest"}]


def test_values_come_from_the_first_run(tmp_path: Path):
    path = tmp_path / 's.py'
    path.write_text(HEADER + TRY_PYTEST + '    x = 2\nelse:\n    x = 1\nprint(repr(x))\n')
    expected = [audit.Expected('x', '1')]
    result = audit.execute(path, 'python', 60, expected)
    assert result['findings'] == []
    assert result['compared'] == 1


def test_baseline_and_rerun_time_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(audit.subprocess, 'run', timeout)
    assert audit.baseline.__wrapped__(('packaging',), 1) == frozenset()
    path = tmp_path / 's.py'
    path.write_text(HEADER)
    monkeypatch.setattr(audit, 'baseline', lambda deps, timeout: frozenset())
    seen = {'pytest': 'pytest'}
    finding = audit.dependency_finding(path, seen, set(), passed=True, tmp=str(tmp_path), timeout=1)
    assert finding is None


def test_failing_script_keeps_its_error(tmp_path: Path):
    path = tmp_path / 's.py'
    path.write_text(HEADER + TRY_PYTEST + '    pass\nraise ValueError("boom")\n')
    assert _kinds(audit.execute(path, 'python', 60, [])) == ['error']


def test_runner_records_writelines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    script = tmp_path / 's.py'
    script.write_text('import sys\nsys.stdout.writelines(["w\\n"])\n')
    output = tmp_path / 'out.json'
    monkeypatch.setattr(sys, 'argv', ['runner', str(script), str(output)])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    assert runner.main() == 0
    assert json.loads(output.read_text()) == [
        {'line': 1, 'stdout': ''},
        {'line': 2, 'stdout': 'w\n'},
    ]


def test_runner_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / 's.py'
    output = tmp_path / 'out.json'
    monkeypatch.setattr(sys, 'path', list(sys.path))
    for source, code in [
        ('import sys\nsys.exit(0)\n', 0),
        ('import sys\nsys.exit(3)\n', 1),
        (
            'import sys\nsys.stdout.flush()\nsys.stdout.buffer.flush()\n',
            0,
        ),
    ]:
        script.write_text(source)
        monkeypatch.setattr(sys, 'argv', ['runner', str(script), str(output)])
        assert runner.main() == code


def test_closure_and_metadata(monkeypatch: pytest.MonkeyPatch):
    assert runner.declared_dependencies('print(1)\n') is None
    assert runner.declared_dependencies(HEADER) == ['packaging']
    assert runner.closure(['packaging', 'packaging', 'not-installed-at-all']) == {
        'packaging',
        'not-installed-at-all',
    }
    assert 'iniconfig' in runner.closure(['pytest'])
    monkeypatch.setattr(runner.Guard, 'hidden', {'q': 'q-dist'})
    assert runner.hidden_import(ValueError()) is None
    assert runner.hidden_import(ImportError('pip install q-dist')) == "'q' from q-dist"
    assert runner.hidden_import(ValueError('pip install q-dist')) is None
    assert runner.hidden_import(ModuleNotFoundError(name='q.sub')) == "'q' from q-dist"
    assert runner.hidden_import(ModuleNotFoundError(name='other')) is None


def test_closure_follows_requested_extras(monkeypatch: pytest.MonkeyPatch):
    requires = {'p': ['a; extra == "x"', 'b']}
    monkeypatch.setattr(importlib.metadata, 'requires', lambda name: requires.get(name, []))
    assert runner.closure(['p[x]']) == {'p', 'a', 'b'}
    assert runner.closure(['p']) == {'p', 'b'}


def test_guard_unloads_packaging(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    monkeypatch.setattr(sys, 'modules', dict(sys.modules))
    monkeypatch.setattr(runner.Guard, 'allowed', set())
    monkeypatch.setattr(runner.Guard, 'providers', {})
    runner.Guard.install(set())
    assert 'packaging.version' not in sys.modules
    assert runner.Guard.find_spec('json') is not None


def test_guard_skips_modules_it_cannot_find(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner.Guard, 'providers', {'seac_missing': ['seac-missing']})
    monkeypatch.setattr(runner.Guard, 'seen', {})
    assert runner.Guard.find_spec('seac_missing') is None
    assert runner.Guard.seen == {}
