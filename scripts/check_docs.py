"""Check registered public APIs and local documentation links without importing simesh."""

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from griffe import GriffeLoader, Parser
import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DIRECTIVE = re.compile(r"^::: (simesh[\w.]*)[ \t]*$", re.MULTILINE)
ENTRY = re.compile(r"^::: (simesh[\w.]*)\n((?:[ \t]+[^\n]*\n)*)", re.MULTILINE)
LINK = re.compile(r"\[[^\]\n]*\]\(([^\s)]+)\)")


def target(obj):
    return obj.final_target if obj.is_alias else obj


def section_names(obj, kind):
    if not obj.docstring:
        return set()
    return {
        item.name.lstrip("*")
        for section in obj.docstring.parsed
        if section.kind.value == kind
        for item in section.value
    }


def main():
    loader = GriffeLoader(search_paths=[ROOT / "src"], allow_inspection=False,
                          docstring_parser=Parser.numpy)
    package = loader.load("simesh")
    loader.resolve_aliases(implicit=True)
    errors, documented = [], {}
    paths = [ROOT / "README.md", ROOT / "ASSETS.md", ROOT / "legacy/README.md", *DOCS.rglob("*.md")]
    for path in DOCS.rglob("*.md"):
        for name, settings in ENTRY.findall(path.read_text()):
            try:
                obj = target(package[name.removeprefix("simesh.")])
            except (KeyError, ValueError) as exc:
                errors.append(f"{path.relative_to(ROOT)}: cannot resolve {name}: {exc}")
                continue
            if obj.path in documented:
                errors.append(f"{name}: duplicates {documented[obj.path]}")
            documented[obj.path] = name
            if not obj.docstring:
                errors.append(f"{name}: missing source docstring")
            members = (yaml.safe_load(settings) or {}).get("options", {}).get("members", [])
            if isinstance(members, list):
                for member in members:
                    item = obj.members.get(member)
                    if item is None:
                        errors.append(f"{name}: unknown selected member {member}")
                    elif (item.is_function or "property" in item.labels) and not item.docstring:
                        errors.append(f"{name}.{member}: missing source docstring")
            if obj.is_function or obj.is_class:
                described = section_names(obj, "parameters")
                actual = {p.name for p in obj.parameters} - {"self", "cls"}
                if described:
                    for unknown in sorted(described - actual):
                        errors.append(f"{name}: documented parameter {unknown!r} is absent from signature")
                    for missing in sorted(actual - described):
                        errors.append(f"{name}: parameter {missing!r} is missing from Parameters")

    exported = []
    for module in (package, package["applications"], package["amrvac"], package["tools"]):
        exported.extend(module[name] for name in module.exports)
    # These advanced modules have no __all__; include locally defined public
    # callables, not imported implementation helpers or dependencies.
    for module in (package["bounded"], package["tools.configurations"]):
        exported.extend(obj for name, obj in module.members.items()
                        if not name.startswith("_") and not obj.is_alias
                        and (obj.is_function or obj.is_class))
    exported.extend(package["bounded"][name] for name in ("PreparedPool", "CurlPool"))
    for obj in exported:
        if target(obj).path not in documented:
            errors.append(f"{obj.path}: public API has no documentation page")

    # This wrapper exposes **kwargs publicly. Check its one authoritative
    # Other Parameters table against the actual forwarding destination.
    forwarded = package["projection._integrate_views"]
    expected = {p.name for p in forwarded.parameters if p.kind.value == "keyword-only"}
    described = section_names(target(package["integrate_los_views"]), "other parameters")
    if described != expected:
        errors.append(f"integrate_los_views: forwarded controls differ: {sorted(described ^ expected)}")

    links = 0
    for path in paths:
        content = path.read_text()
        if content.count("\n```") % 2:
            errors.append(f"{path.relative_to(ROOT)}: unmatched code fence")
        for match in LINK.finditer(content):
            url = match.group(1)
            if urlsplit(url).scheme:
                continue
            relative, _, fragment = unquote(url).partition("#")
            destination = (path.parent / relative).resolve() if relative else path.resolve()
            links += 1
            if not destination.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link {url}")
            elif fragment and destination.suffix == ".md":
                text = destination.read_text()
                anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-")
                           for h in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, re.MULTILINE)}
                anchors.update(DIRECTIVE.findall(text))
                if fragment not in anchors:
                    errors.append(f"{path.relative_to(ROOT)}: missing anchor {url}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(documented)} source API entries, public coverage, parameters and {links} local links.")


if __name__ == "__main__":
    main()
