"""Run a script by statement, recording each one's stdout and allowing only its dependencies.

Usage: ``python _audit_runner.py SCRIPT OUTPUT_JSON``, by path so the package is not imported.
"""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys
import tomllib
import traceback
from typing import ClassVar

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

METADATA = re.compile(r'(?m)^# /// script$\s(?P<content>(^#(| .*)$\s)+)^# ///$')
# Printed to stderr for each import the dependencies do not cover.
BLOCKED = 'sphinx-examples-as-code-audit: blocked import'


def declared_dependencies(source: str) -> list[str] | None:
    """Return the ``dependencies`` of a script's inline metadata, or None without a block."""
    match = METADATA.search(source)
    if match is None:
        return None
    content = ''.join(line[2:] for line in match['content'].splitlines(keepends=True))
    return tomllib.loads(content).get('dependencies', [])


def closure(requirements: list[str]) -> set[str]:
    """Return the normalized names of the distributions installing ``requirements`` pulls in."""
    names: set[str] = set()
    todo = [Requirement(req) for req in requirements]
    seen: set[tuple[str, frozenset[str]]] = set()
    while todo:
        req = todo.pop()
        key = (canonicalize_name(req.name), frozenset(req.extras))
        if key in seen:
            continue
        seen.add(key)
        names.add(key[0])
        try:
            requires = importlib.metadata.requires(req.name) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        for text in requires:
            sub = Requirement(text)
            extras = key[1] | {''}
            if sub.marker is None or any(sub.marker.evaluate({'extra': e}) for e in extras):
                todo.append(sub)
    return names


class Guard(importlib.machinery.PathFinder):
    """Hide top-level modules that only undeclared distributions provide."""

    allowed: ClassVar[set[str]] = set()
    providers: ClassVar[dict[str, list[str]]] = {}
    hidden: ClassVar[dict[str, str]] = {}

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        """Return None for a hidden module, else what the path finder finds."""
        if path is None and fullname not in sys.stdlib_module_names:
            dists = {canonicalize_name(d) for d in cls.providers.get(fullname, [])}
            if dists and not dists & cls.allowed:
                cls.hidden[fullname] = ', '.join(sorted(dists))
                return None
        return super().find_spec(fullname, path, target)

    @classmethod
    def install(cls, allowed: set[str]) -> None:
        """Replace the path finder with this one."""
        cls.allowed = allowed
        cls.providers = importlib.metadata.packages_distributions()
        sys.meta_path[sys.meta_path.index(importlib.machinery.PathFinder)] = cls


def hidden_import(exc: BaseException) -> str | None:
    """Return the hidden module an exception failed to import, or that its traceback names."""
    text = ''.join(traceback.format_exception(exc))
    chained: BaseException | None = exc
    while chained is not None:
        if isinstance(chained, ModuleNotFoundError) and chained.name:
            top = chained.name.partition('.')[0]
            if top in Guard.hidden:
                return f'{top!r} from {Guard.hidden[top]}'
        chained = chained.__cause__ or chained.__context__
    for module, dists in Guard.hidden.items():
        names = '|'.join(map(re.escape, [module, *dists.split(', ')]))
        if re.search(rf'\b({names})\b', text):
            return f'{module!r} from {dists}'
    return None


class Tee(io.TextIOBase):
    """Write to a stream while keeping a copy of what was written."""

    def __init__(self, stream: io.TextIOBase) -> None:
        self.stream = stream
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        """Write ``text`` through and keep it."""
        self.parts.append(text)
        return self.stream.write(text)

    def flush(self) -> None:
        """Flush the underlying stream."""
        self.stream.flush()

    def take(self) -> str:
        """Return and forget what was written since the last call."""
        text, self.parts = ''.join(self.parts), []
        return text


def main() -> int:
    """Run the script given on the command line and write each statement's stdout."""
    script, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    source = script.read_text(encoding='utf-8')
    deps = declared_dependencies(source)
    if deps is not None:
        Guard.install(closure(deps))
    sys.argv = [str(script)]
    sys.path[0] = str(script.parent)
    namespace = {'__name__': '__main__', '__file__': str(script), '__builtins__': __builtins__}
    tee = Tee(sys.stdout)
    sys.stdout = tee
    records = []
    code = 0
    try:
        for node in ast.parse(source, str(script)).body:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(script), 'exec'), namespace)  # noqa: S102
            records.append({'line': node.lineno, 'stdout': tee.take()})
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            pass
        else:
            traceback.print_exc()
            if module := hidden_import(exc):
                print(f'{BLOCKED} of {module}', file=sys.stderr)
            code = 1
    finally:
        sys.stdout = tee.stream
        output.write_text(json.dumps(records))
    return code


if __name__ == '__main__':
    sys.exit(main())
