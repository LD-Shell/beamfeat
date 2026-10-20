"""Show the docstrings' Sphinx cross-reference roles as code on the docs site.

The docstrings write references as Sphinx roles, ``:class:`~beamfeat.search.BeamSearch```,
which read well in a terminal and in tools that understand them. mkdocstrings
does not, and would print the role markup as it stands. This griffe extension,
loaded by ``mkdocs.yml``, turns each role into a code span of the name a reader
should see: the last component after ``~``, with parentheses for functions and
methods, as Sphinx renders it.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

_ROLE = re.compile(r":(?:py:)?(attr|class|data|exc|func|meth|mod|obj):`(~?)([^`]+)`")


def _code(match: re.Match[str]) -> str:
    role, tilde, target = match.groups()
    name = target.rsplit(".", 1)[-1] if tilde else target
    if role in ("func", "meth"):
        name += "()"
    return f"`{name}`"


class SphinxRoles(griffe.Extension):
    """Rewrite Sphinx roles in every docstring griffe loads."""

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.docstring is not None:
            obj.docstring.value = _ROLE.sub(_code, obj.docstring.value)
