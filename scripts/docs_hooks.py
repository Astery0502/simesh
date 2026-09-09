"""Keep checkout-relative source links useful in the built documentation."""

from pathlib import Path
import ast
import re
from urllib.parse import quote, unquote, urlsplit

from griffe import Extension, ExprName


ROOT = Path(__file__).resolve().parents[1]


class LiteralDefaults(Extension):
    """Display same-scope literal constants without executing package code."""

    def on_function_instance(self, *, func, **kwargs):
        for parameter in func.parameters:
            if not isinstance(parameter.default, ExprName):
                continue
            member = func.parent.members.get(parameter.default.name)
            if member is None or member.is_alias or not member.is_attribute:
                continue
            value = str(member.value)
            try:
                ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
            parameter.default = value


def on_page_markdown(markdown, page, config, files):
    """Convert links outside docs to repository URLs; leave source Markdown local."""
    docs = Path(config.docs_dir).resolve()
    parent = Path(page.file.abs_src_path).parent

    def link(match):
        label, url = match.groups()
        if urlsplit(url).scheme or url.startswith("#"):
            return match.group(0)
        path, _, fragment = unquote(url).partition("#")
        destination = (parent / path).resolve()
        if not destination.is_relative_to(ROOT) or destination.is_relative_to(docs):
            return match.group(0)
        relative = destination.relative_to(ROOT).as_posix()
        target = config.repo_url.rstrip("/") + "/blob/main/" + quote(relative)
        return f"[{label}]({target}" + (f"#{fragment}" if fragment else "") + ")"

    return re.sub(r"\[([^\]\n]*)\]\(([^\s)]+)\)", link, markdown)
