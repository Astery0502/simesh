"""Check implementation dependencies, including local imports and Cython cimports.

This source check protects numerical boundaries without importing extensions.
It follows declared imports, not dynamic attribute access or runtime call graphs.
The root package is a public facade; implementation modules depend on definitions.
"""

import argparse
import ast
from importlib.util import resolve_name
import io
from pathlib import Path
import tokenize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"

# Allowed internal dependencies for reusable foundations. Other modules may
# compose these services; scientific consumers still cannot prepare/read input.
FOUNDATIONS = {
    "simesh._validation": (),
    "simesh._kernels.primitives": ("simesh._kernels.primitives",),
    "simesh._kernels.rk4": (),
    "simesh._kernels.cartesian": (),
    "simesh._kernels.interpolation": (),
    "simesh._kernels": ("simesh._kernels",),
    "simesh._amr": ("simesh._amr", "simesh._kernels.primitives", "simesh._validation"),
    "simesh.mesh": ("simesh._validation", "simesh._amr", "simesh._kernels.native"),
    "simesh.fields": ("simesh._validation", "simesh.mesh"),
    "simesh.spatial": ("simesh._validation",),
    "simesh.geometry": ("simesh.spatial", "simesh.mesh"),
    "simesh._field_data": ("simesh._validation", "simesh.fields"),
    "simesh.field_ops": ("simesh._validation", "simesh.fields", "simesh._field_data", "simesh.mesh", "simesh.spatial"),
    "simesh._execution": ("simesh._validation", "simesh._kernels"),
    "simesh.physics.composition": (),
    "simesh.physics.units": ("simesh.physics.composition",),
    "simesh.physics.emission": ("simesh.physics.composition", "simesh.physics._aia171_table",
                                "simesh.physics._euv_tables", "simesh._validation"),
    "simesh.physics.thermodynamics": ("simesh._field_data", "simesh.physics.emission", "simesh.fields",
                                      "simesh.field_ops", "simesh._validation"),
    "simesh.physics.radiation": ("simesh.physics.emission", "simesh.physics.thermodynamics",
                                 "simesh.physics.composition", "simesh.physics._euv_tables",
                                 "simesh.fields"),
    "simesh.preparation": ("simesh.preparation", "simesh._amr", "simesh._kernels",
                           "simesh._validation", "simesh._execution", "simesh.mesh", "simesh.fields"),
    "simesh.operators": ("simesh._field_data", "simesh.operators", "simesh.fields", "simesh.field_ops",
                         "simesh._validation", "simesh._execution", "simesh._kernels"),
    "simesh._uniform": ("simesh.geometry", "simesh.fields", "simesh._validation",
                        "simesh._execution", "simesh._kernels"),
    "simesh.io._v5": ("simesh.io._v5", "simesh._amr"),
    "simesh.io.source": ("simesh.io.metadata", "simesh.io._boundary", "simesh.fields",
                         "simesh.mesh", "simesh._amr", "simesh._validation"),
    "simesh.io.metadata": (),
    "simesh.io._boundary": ("simesh._amr", "simesh._validation", "simesh.fields"),
    "simesh.tools": ("simesh.tools",),
}
CONSUMERS = tuple("simesh."+name for name in (
    "applications", "connectivity", "current_proxy", "diagnostics", "line_profiles",
    "projection", "reductions", "slices", "tracing", "physics", "operators"))
INPUT_SERVICES = ("simesh.io", "simesh.preparation", "simesh._pool", "simesh.bounded")


def within(name, prefix):
    return name == prefix or name.startswith(prefix+".")


def module_name(path):
    parts = path.relative_to(SOURCE).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def cython_import_nodes(content):
    """Read complete import statements without treating strings/comments as code."""
    statement = []
    for token in tokenize.generate_tokens(io.StringIO(content).readline):
        if token.type in (tokenize.NEWLINE, tokenize.ENDMARKER) or token.string == ";":
            for index, (kind, word) in enumerate(statement):
                if kind == tokenize.NAME and word in ("import", "cimport"):
                    start = next((i for i in range(index-1, -1, -1)
                                  if statement[i] == (tokenize.NAME, "from")), index)
                    normalized = [(k, "import" if w == "cimport" else w)
                                  for k, w in statement[start:]]
                    yield ast.parse(tokenize.untokenize(normalized)).body[0]
                    break
            statement = []
        elif token.type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            statement.append((token.type, token.string))


def imports(path, modules):
    module = module_name(path)
    package = module if path.stem == "__init__" else module.rpartition(".")[0]
    content = path.read_text()
    nodes = ast.walk(ast.parse(content)) if path.suffix == ".py" else cython_import_nodes(content)
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("."*node.level+(node.module or ""), package)
            for alias in node.names:
                child = base+"."+alias.name
                yield child if child in modules else base


def dependency_graph():
    paths = sorted(path for path in (SOURCE/"simesh").rglob("*")
                   if path.suffix in (".py", ".pyx", ".pxd"))
    modules = {module_name(path) for path in paths}
    graph = {name: set() for name in modules}
    for path in paths:
        origin = module_name(path)
        graph[origin].update(name for name in imports(path, modules)
                             if name in modules and name != origin)
    return graph


def violations(graph):
    errors = []
    for origin, dependencies in sorted(graph.items()):
        rule = next((allowed for prefix, allowed in FOUNDATIONS.items()
                     if within(origin, prefix)), None)
        for target in sorted(dependencies):
            if rule is not None and not any(within(target, prefix) for prefix in rule):
                errors.append(f"Foundation imports an upper layer: {origin} -> {target}")
            if (any(within(origin, prefix) for prefix in CONSUMERS)
                    and any(within(target, prefix) for prefix in INPUT_SERVICES)):
                errors.append(f"Consumer imports input orchestration: {origin} -> {target}")
            if origin != "simesh" and target == "simesh":
                errors.append(f"Implementation imports public facade: {origin} -> {target}")
            if origin == "simesh.applications" and within(target, "simesh._kernels"):
                errors.append(f"Application binds a kernel instead of an operation: {origin} -> {target}")

    visited, active = set(), []

    def visit(module):
        if module in active:
            errors.append("Import cycle: "+" -> ".join(active[active.index(module):]+[module]))
            return
        if module in visited:
            return
        active.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        active.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List direct implementation dependencies")
    args = parser.parse_args()
    graph = dependency_graph()
    if args.list:
        for module, dependencies in sorted(graph.items()):
            print(f"{module}: {', '.join(sorted(dependencies))}")
    errors = violations(graph)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(graph)} modules and {sum(map(len, graph.values()))} internal dependencies; "
          "no cycles or boundary violations.")


if __name__ == "__main__":
    main()
