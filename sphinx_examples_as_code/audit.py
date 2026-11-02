"""Run every example download of a built documentation tree and report what is wrong with it.

``python -m sphinx_examples_as_code.audit --help`` lists the subcommands.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
import doctest
import fnmatch
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from . import _FORMAT_ORDER as FORMATS
from . import _SUFFIXES as SUFFIXES
from ._audit_runner import BLOCKED

RUNNER = Path(__file__).with_name('_audit_runner.py')
OUTPUT_TAIL = 4000
ENV = {
    'PYVISTA_OFF_SCREEN': 'true',
    'PYVISTA_JUPYTER_BACKEND': 'static',
    'MPLBACKEND': 'Agg',
    'PYDEVD_DISABLE_FILE_VALIDATION': '1',
}
# Left in place of any line marimo's converter mistook for an IPython magic.
MAGIC_MARKER = '# magic command not supported in marimo'
GALLERY_PAGE = 'sphx-glr-timing'
DOWNLOAD_LINK = re.compile(
    r'<a class="reference download internal"[^>]*href="[^"]*?_downloads/([^"]+)"'
)
CODE_BLOCK = re.compile(
    r'<div class="[^"]*highlight-[\w-]+[^"]*"><div class="highlight"><pre>(.*?)</pre>', re.DOTALL
)
TAG = re.compile(r'<[^>]+>')
NUMBER = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?')
# marimo's converter renames a redefinition ``x`` to ``x_1`` and a cell-local ``x`` to ``_x``.
MARIMO_NAME = re.compile(r'^_?(.+?)(?:_\d+)?$')
# A value marimo shows inside a JSON container, e.g. ``text/plain+float:0.5``.
MIME_LEAF = re.compile(r'^((?:text|application|image)/[\w.+-]+):(.*)$', re.DOTALL)
NUMPY_SCALAR = re.compile(r'\bnp\.\w+\(([^()]*)\)')
PRIVATE_NAME_ERROR = re.compile(r"NameError: name '_\w+' is not defined")
# Script findings that mean the script itself does not run.
SCRIPT_FAILURES = ('error', 'dependency', 'timeout')
CHECK_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
# Finding kinds, in report order, with their headings.
KINDS = {
    'syntax': 'Code that is not valid Python (doctest prompts, `%` lines)',
    'magic': 'Lines the marimo converter turned into IPython magics',
    'marimo-check': 'marimo check failures (e.g. multiple definitions)',
    'dependency': 'Imports the download does not declare as dependencies',
    'collapsed': 'Doctest values hidden inside a notebook cell',
    'dropped-output': 'Doctest values the script does not print',
    'mismatch': "Notebook values that differ from the doctest's output while the script's match",
    'marimo-rename': 'marimo converter renamed a name to a private `_name`',
    'error': 'Fails as a notebook while the plain Python script runs',
    'timeout': 'Timed out',
    'missing-download': 'Pages missing a download format',
    'python-mismatch': "Values that differ from the doctest's output in the script too",
    'python-error': 'Fails as a plain Python script',
}


@dataclass(frozen=True)
class Expected:
    """One doctest statement with output, as rendered on a page."""

    source: str
    want: str


def stem(path: Path) -> str:
    """Return the example name shared by all formats of one download."""
    for suffix in (SUFFIXES['marimo'], SUFFIXES['jupyter'], SUFFIXES['python']):
        if path.name.endswith(suffix):
            return path.name.removesuffix(suffix)
    return path.stem


def fmt_of(name: str) -> str | None:
    """Return the format of a download file name, or None for anything else."""
    if name.endswith(SUFFIXES['marimo']):
        return 'marimo'
    return next((f for f in ('jupyter', 'python') if name.endswith(SUFFIXES[f])), None)


def page_groups(text: str) -> list[tuple[list[str], list[str]]]:
    """Return each download link group of a page with the code blocks that follow it."""
    events = [(m.start(), 'link', m[1]) for m in DOWNLOAD_LINK.finditer(text)]
    events += [(m.start(), 'block', m[1]) for m in CODE_BLOCK.finditer(text)]
    groups: list[tuple[list[str], list[str]]] = []
    last_link = -1
    for pos, kind, value in sorted(events):
        if kind == 'link':
            if groups and not groups[-1][1] and '</p>' not in text[last_link:pos]:
                groups[-1][0].append(value)
            else:
                groups.append(([value], []))
            last_link = pos
        elif groups:
            groups[-1][1].append(html.unescape(TAG.sub('', value)))
    return groups


def expectations(html_dir: Path) -> dict[str, list[Expected]]:
    """Map each download to the doctest statements with output shown after its links."""
    parser = doctest.DocTestParser()
    found: dict[str, list[Expected]] = {}
    for page in sorted(html_dir.rglob('*.html')):
        for links, blocks in page_groups(page.read_text(errors='replace')):
            examples = [
                Expected(ex.source, ex.want)
                for block in blocks
                if '>>> ' in block
                for ex in parser.get_examples(block)
                if ex.want.strip()
            ]
            for link in links:
                found.setdefault(link, examples)
    return found


def collect(downloads: Path, fmt: str, linked: set[str] | None) -> list[Path]:
    """Return the downloads of one format, only those a page links to when ``linked`` is set."""
    files = (f for f in downloads.rglob(f'*{SUFFIXES[fmt]}') if fmt_of(f.name) == fmt)
    if linked is not None:
        files = (f for f in files if f.relative_to(downloads).as_posix() in linked)
    return sorted(files)


def statements(source: str) -> list[ast.stmt] | None:
    """Return the top-level statements of ``source``, or None if it is not valid Python."""
    try:
        return ast.parse(source).body
    except SyntaxError:
        return None


class _Unrename(ast.NodeTransformer):
    """Undo marimo's renames so converted cells compare equal to their doctest."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = MARIMO_NAME.match(node.id)[1]
        return node


def dump(node: ast.stmt, *, unrename: bool = False) -> str:
    """Return a comparable dump of a statement."""
    if unrename:
        node = _Unrename().visit(ast.parse(ast.unparse(node)).body[0])
    return ast.dump(node)


def is_value(node: ast.stmt) -> bool:
    """Check whether a statement is an expression whose value a notebook would display."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    return not (isinstance(call, ast.Call) and getattr(call.func, 'id', None) == 'print')


def printed(node: ast.stmt) -> ast.stmt:
    """Return ``node`` the way a seac script prints it: ``print(repr(value))``."""
    if not is_value(node):
        return node
    wrapped = ast.parse(f'print(repr(({ast.unparse(node.value)})))').body[0]
    return ast.fix_missing_locations(wrapped)


def round_like(want: str, got: str) -> str:
    """Round each float in ``got`` to the precision of the matching float in ``want``."""
    wants, gots = list(NUMBER.finditer(want)), list(NUMBER.finditer(got))
    if len(wants) != len(gots):
        return got
    for w, g in reversed(list(zip(wants, gots, strict=True))):
        mantissa, _, exponent = w[0].lower().partition('e')
        digits = len(mantissa.partition('.')[2]) - int(exponent or 0)
        if abs(float(w[0]) - float(g[0])) <= 10.0**-digits:
            got = got[: g.start()] + w[0] + got[g.end() :]
    return got


def as_lists(value: object) -> object:
    """Return a literal with every tuple turned into a list, as JSON would show it."""
    if isinstance(value, (list, tuple)):
        return [as_lists(item) for item in value]
    if isinstance(value, dict):
        return {key: as_lists(item) for key, item in value.items()}
    return value


def same_output(want: str, got: str) -> bool:
    """Compare a displayed value with a doctest's expected output (``ELLIPSIS``, ``NUMBER``)."""
    got = round_like(want, got)
    if doctest.OutputChecker().check_output(want, got, CHECK_FLAGS):
        return True
    want, got = (NUMPY_SCALAR.sub(r'\1', text.strip()) for text in (want, got))
    try:
        return as_lists(ast.literal_eval(want)) == as_lists(ast.literal_eval(got))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False


def is_literal(text: str) -> bool:
    """Check whether a doctest's output is a Python literal, the only kind marimo shows as is."""
    try:
        ast.literal_eval(NUMPY_SCALAR.sub(r'\1', text.strip()))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    return True


def output_text(outputs: list[dict]) -> str | None:
    """Return the stdout and displayed values of a notebook cell, or None if one is not text."""
    parts = []
    for out in outputs:
        kind = out['output_type']
        if kind == 'stream' and out['name'] == 'stdout':
            parts.append(''.join(out['text']))
        elif kind in ('execute_result', 'display_data'):
            data = out['data']
            text = {key: ''.join(v) for key, v in data.items() if isinstance(v, (str, list))}
            if 'text/plain' in text:
                parts.append(text['text/plain'] + '\n')
            elif 'application/json' in data:
                value = data['application/json']
                shown = json_repr(json.loads(value) if isinstance(value, str) else value)
                if shown is None:
                    return None
                parts.append(shown + '\n')
            elif text.get('text/html', '').startswith('<pre'):
                parts.append(html.unescape(TAG.sub('', text['text/html'])) + '\n')
            elif kind == 'execute_result' or 'text/html' in text:
                return None
        elif kind == 'error':
            return None
    return ''.join(parts)


def json_repr(value: object) -> str | None:
    """Return the repr of a JSON value marimo shows, or None if a leaf is not plain text."""
    if isinstance(value, str):
        match = MIME_LEAF.match(value)
        if match is None:
            return repr(value)
        return match[2] if match[1].startswith('text/plain') else None
    if isinstance(value, (list, dict)):
        pairs = value.items() if isinstance(value, dict) else [(None, v) for v in value]
        shown = [(key, json_repr(item)) for key, item in pairs]
        if any(item is None for _, item in shown):
            return None
        if isinstance(value, list):
            return '[' + ', '.join(item for _, item in shown) + ']'
        return '{' + ', '.join(f'{key!r}: {item}' for key, item in shown) + '}'
    return repr(value)


def find(cells: list[list[str]], target: list[str], used: set) -> tuple[int, int] | None:
    """Return the first unused (cell, statement) position where ``target`` starts."""
    for u, cell in enumerate(cells):
        for i in range(len(cell) - len(target) + 1):
            if cell[i : i + len(target)] == target and (u, i) not in used:
                return u, i
    return None


def compare(
    expected: list[Expected], units: list[tuple[list[ast.stmt], str | None]], fmt: str
) -> tuple[list[dict], int]:
    """Match each doctest statement to a cell (or script statements) and check its output."""
    findings, compared = [], 0
    unrename = fmt == 'marimo'
    if fmt == 'python':
        # One cell holding the whole script, with each statement's stdout.
        outputs = [out for _, out in units]
        units = [([node for body, _ in units for node in body], None)]
    cells = [[dump(node, unrename=unrename) for node in body] for body, _ in units]
    used: set[tuple[int, int]] = set()
    for ex in expected:
        body = statements(ex.source)
        if not body:
            continue
        line = ex.source.strip().splitlines()[-1]
        target = [dump(printed(n) if fmt == 'python' else n, unrename=unrename) for n in body]
        hit = find(cells, target, used)
        if hit is None:
            bare = [dump(n) for n in body]
            if fmt == 'python' and is_value(body[-1]) and find(cells, bare, used):
                findings.append({'kind': 'dropped-output', 'message': line})
            continue
        used.add(hit)
        u, i = hit
        if fmt == 'python':
            ran = outputs[i : i + len(target)]
            got = None if None in ran else ''.join(ran)
        elif i + len(target) < len(cells[u]):
            if is_value(body[-1]):
                findings.append({'kind': 'collapsed', 'message': line})
            continue
        else:
            got = units[u][1]
        if got is None or (fmt == 'marimo' and not is_literal(ex.want)):
            continue
        compared += 1
        if not same_output(ex.want, got):
            want, got = ex.want.strip(), got.strip()
            message = f'{line} -> {got!r}, expected {want!r}'
            findings.append({'kind': 'mismatch', 'message': message, 'statement': ex.source})
    return findings, compared


def python_units(path: Path, records: list[dict]) -> list[tuple[list[ast.stmt], str | None]]:
    """Return each top-level statement of a script with the stdout the runner recorded."""
    stdout = {r['line']: r['stdout'] for r in records}
    body = statements(path.read_text(encoding='utf-8')) or []
    return [([node], stdout.get(node.lineno)) for node in body]


def notebook_units(path: Path) -> list[tuple[list[ast.stmt], str | None]]:
    """Return each parsable code cell of an executed notebook with its text output."""
    if not path.exists():
        return []
    code = [c for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code']
    return [
        (body, output_text(cell.get('outputs', [])))
        for cell in code
        if (body := statements(''.join(cell['source'])))
    ]


def notebook_error(path: Path) -> str:
    """Return the first error recorded in an executed notebook, or an empty string."""
    if not path.exists():
        return ''
    for cell in json.loads(path.read_text())['cells']:
        for out in cell.get('outputs', []):
            if out['output_type'] == 'error' and out['ename'] != 'Ancestor raised':
                trace = TAG.sub('', re.sub(r'\x1b\[[\d;]*m', '', '\n'.join(out['traceback'])))
                return f'{trace}\n{out["ename"]}: {out["evalue"]}'
    return ''


def static_findings(path: Path, fmt: str) -> list[dict]:
    """Return the problems visible without running a download."""
    text = path.read_text(encoding='utf-8')
    if fmt == 'jupyter':
        cells = json.loads(text)['cells']
        sources = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
    else:
        sources = [text]
    findings = []
    for source in sources:
        try:
            ast.parse(source)
        except SyntaxError as exc:
            line = (exc.text or '').strip()
            findings.append({'kind': 'syntax', 'message': f'{exc.msg}: {line}'})
    if fmt == 'marimo' and MAGIC_MARKER in text:
        line = next(i for i, s in enumerate(text.splitlines(), 1) if MAGIC_MARKER in s)
        findings.append({'kind': 'magic', 'message': f'line {line}'})
    return findings


def commands(path: Path, fmt: str, tmp: str) -> list[tuple[str, list[str]]]:
    """Return the named stages that check and execute one download."""
    if fmt == 'python':
        return [('run', [sys.executable, str(RUNNER), str(path), f'{tmp}/stdout.json'])]
    if fmt == 'jupyter':
        execute = [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute']
        options = ['--allow-errors', '--ExecutePreprocessor.timeout=-1']
        out = ['--output-dir', tmp, '--output', 'executed']
        return [('run', [*execute, *options, *out, str(path)])]
    export = [sys.executable, '-m', 'marimo', 'export', 'ipynb', '--include-outputs']
    return [
        ('check', [sys.executable, '-m', 'marimo', 'check', str(path)]),
        ('run', [*export, '--no-sandbox', '-f', '-o', f'{tmp}/executed.ipynb', str(path)]),
    ]


def error_finding(fmt: str, output: str) -> dict:
    """Classify a failed run by its output."""
    if BLOCKED in output:
        line = next(s for s in output.splitlines() if BLOCKED in s)
        return {'kind': 'dependency', 'message': line.removeprefix(BLOCKED).strip()}
    if fmt == 'marimo' and (match := PRIVATE_NAME_ERROR.search(output)):
        return {'kind': 'marimo-rename', 'message': match[0]}
    return {'kind': 'error', 'message': output[-OUTPUT_TAIL:]}


def execute(path: Path, fmt: str, timeout: float, expected: list[Expected]) -> dict:
    """Check and run one download in its own temporary directory and return its findings."""
    start = time.monotonic()
    findings = static_findings(path, fmt)
    compared = 0
    env = {**os.environ, **ENV}
    with tempfile.TemporaryDirectory(prefix='seac-audit-') as tmp:
        for stage, cmd in commands(path, fmt, tmp):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                code, output = proc.returncode, proc.stdout + proc.stderr
            except subprocess.TimeoutExpired:
                findings.append({'kind': 'timeout', 'message': f'{stage} after {timeout:g} s'})
                break
            if stage == 'check':
                if code:
                    findings.append({'kind': 'marimo-check', 'message': output[:OUTPUT_TAIL]})
                continue
            executed = Path(tmp) / 'executed.ipynb'
            if fmt != 'python':
                output = notebook_error(executed) or (output if code else '')
            if code or (output and fmt != 'python'):
                findings.append(error_finding(fmt, output))
            if fmt == 'python':
                records_file = Path(tmp) / 'stdout.json'
                records = json.loads(records_file.read_text()) if records_file.exists() else []
                units = python_units(path, records)
            else:
                units = notebook_units(executed)
            more, compared = compare(expected, units, fmt)
            findings += more
    return {
        'name': stem(path),
        'format': fmt,
        'findings': findings,
        'expected': len(expected),
        'compared': compared,
        'seconds': round(time.monotonic() - start, 1),
    }


def run(args: argparse.Namespace) -> None:
    """Check and execute every download of one format and write the results as JSON lines."""
    downloads = args.downloads.resolve()
    expected = expectations(args.html.resolve()) if args.html else {}
    linked = set(expected) if args.html else None
    files = [
        f
        for f in collect(downloads, args.format, linked)
        if not any(fnmatch.fnmatch(stem(f), pattern) for pattern in args.exclude)
    ][: args.limit]
    print(f'{len(files)} {args.format} downloads in {downloads}', flush=True)
    failed = 0
    with ThreadPoolExecutor(args.workers) as pool, args.output.open('w') as out:
        futures = {}
        for f in files:
            rel = f.relative_to(downloads).as_posix()
            future = pool.submit(execute, f, args.format, args.timeout, expected.get(rel, []))
            futures[future] = rel
        for i, future in enumerate(as_completed(futures), 1):
            result = {'file': futures[future], **future.result()}
            out.write(json.dumps(result) + '\n')
            out.flush()
            if result['findings']:
                failed += 1
                kinds = sorted({f['kind'] for f in result['findings']})
                print(f'FAIL {result["file"]} ({", ".join(kinds)})', flush=True)
            if i % 100 == 0:
                print(f'{i}/{len(files)} done, {failed} with findings', flush=True)
    print(f'{len(files)} run, {failed} with findings', flush=True)


def pages(args: argparse.Namespace) -> None:
    """Record every download link group, or gallery page, that lacks a format others have."""
    html_dir = args.html.resolve()
    groups: list[tuple[str, set[str]]] = []
    for page in sorted(html_dir.rglob('*.html')):
        text = page.read_text(errors='replace')
        found = page_groups(text)
        rel = page.relative_to(html_dir).as_posix()
        if not found and GALLERY_PAGE in text:
            groups.append((rel, set()))
        groups += [(rel, {fmt_of(link) for link in links} - {None}) for links, _ in found]
    expected = set().union(*(formats for _, formats in groups))
    missing = 0
    with args.output.open('w') as out:
        for page, formats in groups:
            if lacking := sorted(expected - formats):
                missing += 1
                finding = {'kind': 'missing-download', 'message': f'no {", ".join(lacking)}'}
                record = {'file': page, 'name': Path(page).stem, 'format': 'page'}
                out.write(json.dumps({**record, 'findings': [finding]}) + '\n')
    print(f'{len(groups)} download groups, {missing} missing a format', flush=True)


def summary_line(message: str) -> str:
    """Return the line of a message that names the problem, safe for a table cell."""
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    text = lines[-1] if lines else '-'
    text = text.replace('|', '\\|').replace('`', "'")
    return text if len(text) <= 200 else text[:197] + '...'


def report(args: argparse.Namespace) -> None:
    """Group the findings of every result file by kind and write a Markdown report."""
    results = [
        json.loads(line)
        for path in args.results
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    script_findings = {
        (r['name'], 'error' if f['kind'] in SCRIPT_FAILURES else f['kind'], f.get('statement'))
        for r in results
        if r['format'] == 'python'
        for f in r['findings']
    }
    rows: dict[str, list[str]] = defaultdict(list)
    for r in sorted(results, key=lambda r: (r['name'], r['format'])):
        for f in r['findings']:
            kind = f['kind']
            in_script = (r['name'], kind, f.get('statement')) in script_findings
            if kind in ('error', 'mismatch') and (r['format'] == 'python' or in_script):
                kind = f'python-{kind}'
            row = f'| `{Path(r["file"]).name}` | {r["format"]} | `{summary_line(f["message"])}` |'
            if row not in rows[kind]:
                rows[kind].append(row)

    lines = [
        '| Format | Downloads | With findings | Values compared |',
        '| --- | --- | --- | --- |',
    ]
    for fmt in FORMATS:
        mine = [r for r in results if r['format'] == fmt]
        bad = sum(bool(r['findings']) for r in mine)
        values = sum(r.get('compared', 0) for r in mine)
        lines.append(f'| {fmt} | {len(mine)} | {bad} | {values} |')
    for kind, heading in KINDS.items():
        found = rows.get(kind, [])
        if not found:
            continue
        lines += ['', f'#### {heading} ({len(found)})', '']
        if kind.startswith('python-'):
            lines += ['<details>', '<summary>Show</summary>', '']
        lines += ['| Download | Format | Detail |', '| --- | --- | --- |', *found[: args.max_rows]]
        if len(found) > args.max_rows:
            lines.append(f'\n...and {len(found) - args.max_rows} more.')
        if kind.startswith('python-'):
            lines += ['', '</details>']
    failures = sum(len(rows.get(kind, [])) for kind in KINDS if kind not in args.allow)
    if not failures:
        lines += ['', 'No failures.']
    text = '\n'.join(lines) + '\n'
    args.output.write_text(text)
    print(text)
    if github_output := os.environ.get('GITHUB_OUTPUT'):
        with Path(github_output).open('a') as out:
            out.write(f'failures={failures}\n')


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the chosen subcommand."""
    parser = argparse.ArgumentParser(prog='sphinx-examples-as-code-audit', description=__doc__)
    sub = parser.add_subparsers(required=True)

    run_parser = sub.add_parser('run', help='check and execute the downloads of one format')
    run_parser.add_argument('downloads', type=Path, help='the _downloads directory')
    run_parser.add_argument('--format', choices=FORMATS, required=True)
    run_parser.add_argument('--output', type=Path, required=True, help='JSON lines file')
    run_parser.add_argument('--html', type=Path, help='the built HTML, for expected outputs')
    run_parser.add_argument('--workers', type=int, default=os.cpu_count())
    run_parser.add_argument('--timeout', type=float, default=300, help='seconds per stage')
    run_parser.add_argument('--limit', type=int, help='run only the first N files')
    run_parser.add_argument(
        '--exclude', nargs='*', default=[], help='glob patterns of example names not to run'
    )
    run_parser.set_defaults(func=run)

    pages_parser = sub.add_parser('pages', help='find download links missing a format')
    pages_parser.add_argument('html', type=Path, help='the built HTML directory')
    pages_parser.add_argument('--output', type=Path, required=True, help='JSON lines file')
    pages_parser.set_defaults(func=pages)

    report_parser = sub.add_parser('report', help='summarize result files as Markdown')
    report_parser.add_argument('results', type=Path, nargs='*', help='JSON lines files')
    report_parser.add_argument('--output', type=Path, required=True, help='Markdown file')
    report_parser.add_argument(
        '--allow',
        nargs='*',
        default=['python-mismatch', 'python-error'],
        choices=KINDS,
        help='finding kinds that do not count as failures',
    )
    report_parser.add_argument('--max-rows', type=int, default=50, help='rows per section')
    report_parser.set_defaults(func=report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
