"""Tests for the download audit: ``sphinx_examples_as_code.audit`` and its script runner."""

from __future__ import annotations

import html
import importlib.util
import json
from pathlib import Path
import sys
import textwrap

import pytest

from sphinx_examples_as_code import _audit_runner as runner
from sphinx_examples_as_code import audit

HEADER = '# /// script\n# dependencies = [\n#     "packaging",\n# ]\n# ///\n'


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
        'd/fails.py': 'raise ValueError("boom")\n',
        'd/fails.ipynb': _notebook('raise ValueError("boom")'),
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
    assert "'pytest' from pytest" in results['d/undeclared.py']['findings'][0]['message']
    assert results['d/probe.py']['findings'] == []
    assert _kinds(results['d/hint.py']) == ['dependency']
    assert _kinds(results['d/fails.py']) == ['error']

    output = tmp_path / 'excluded.jsonl'
    downloads = str(site / '_downloads')
    args = ['run', downloads, '--format', 'python', '--output', str(output)]
    audit.main([*args, '--exclude', 'fail*', '?ndeclared'])
    names = {json.loads(line)['name'] for line in output.read_text().splitlines()}
    assert names == {'good', 'dropped', 'prompt', 'probe', 'hint'}


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
            'findings': [{'kind': 'dependency', 'message': "'q' from q"}],
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
    audit.main(['report', str(jsonl), '--output', str(report), '--allow', *allow])
    assert report.read_text().endswith('No failures.\n')


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
        ('(np.uint8(0), np.uint8(117))\n', '[0, 117]\n', True),
    ],
)
def test_same_output(want: str, got: str, same: bool):
    assert audit.same_output(want, got) is same


def test_output_text():
    stream = {'output_type': 'stream', 'name': 'stdout', 'text': ['a\n']}
    stderr = {'output_type': 'stream', 'name': 'stderr', 'text': ['warning\n']}
    plain = {'output_type': 'execute_result', 'data': {'text/plain': ['1']}}
    json_value = {'output_type': 'display_data', 'data': {'application/json': '[1, 2]'}}
    pre = {'output_type': 'display_data', 'data': {'text/html': "<pre class='x'>a &lt; b</pre>"}}
    image = {'output_type': 'display_data', 'data': {'image/png': 'abc'}}
    rich = {'output_type': 'display_data', 'data': {'text/html': '<table></table>'}}
    assert (
        audit.output_text([stream, stderr, plain, json_value, pre, image])
        == 'a\n1\n[1, 2]\na < b\n'
    )
    assert audit.output_text([rich]) is None
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
    assert audit.expectations(tmp_path) == {'a/x.py': [audit.Expected('2\n', '2\n')]}


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


def test_runner_in_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    script = tmp_path / 's.py'
    script.write_text(HEADER + 'print("a")\nx = 1\nprint(x)\nimport pytest\n')
    output = tmp_path / 'out.json'
    monkeypatch.setattr(sys, 'argv', ['runner', str(script), str(output)])
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'pytest')
    assert runner.main() == 1
    assert json.loads(output.read_text()) == [
        {'line': 6, 'stdout': 'a\n'},
        {'line': 7, 'stdout': ''},
        {'line': 8, 'stdout': '1\n'},
    ]
    assert f"{runner.BLOCKED} of 'pytest' from pytest" in capsys.readouterr().err


def test_runner_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / 's.py'
    output = tmp_path / 'out.json'
    monkeypatch.setattr(sys, 'path', list(sys.path))
    for source, code in [('import sys\nsys.exit(0)\n', 0), ('import sys\nsys.exit(3)\n', 1)]:
        script.write_text(source)
        monkeypatch.setattr(sys, 'argv', ['runner', str(script), str(output)])
        assert runner.main() == code


def test_closure_and_metadata():
    assert runner.declared_dependencies('print(1)\n') is None
    assert runner.declared_dependencies(HEADER) == ['packaging']
    assert runner.closure(['packaging', 'not-installed-at-all']) == {
        'packaging',
        'not-installed-at-all',
    }
    assert 'iniconfig' in runner.closure(['pytest'])
    assert runner.hidden_import(ValueError()) is None
