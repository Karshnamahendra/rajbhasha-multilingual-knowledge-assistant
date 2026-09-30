"""Request-scoped document selection.

When the user selects several documents in the frontend, the query should only
see those documents. Most of the pipeline already threads a single
``document_id`` through retrieval; instead of changing every signature to take
a list, the selected ids are held in a ContextVar for the duration of one
request. Every "search all documents" code path (document_id is None) consults
this scope and restricts itself to the selected ids.

Usage (see app.py):
    token = set_scope(["a.pdf", "b.docx"])
    try:
        pipeline.ask(query, document_id=None, ...)
    finally:
        reset_scope(token)
"""
from contextvars import ContextVar, Token
from typing import FrozenSet, Iterable, Optional

_scope: ContextVar[Optional[FrozenSet[str]]] = ContextVar("document_scope", default=None)


def set_scope(document_ids: Optional[Iterable[str]]) -> Token:
    ids = frozenset(str(d) for d in (document_ids or []) if d)
    return _scope.set(ids or None)


def reset_scope(token: Token) -> None:
    _scope.reset(token)


def get_scope() -> Optional[FrozenSet[str]]:
    """Selected document ids for this request, or None when all documents are allowed."""
    return _scope.get()


def in_scope(document_id: Optional[str]) -> bool:
    scope = _scope.get()
    return scope is None or (document_id is not None and str(document_id) in scope)


def qdrant_scope_condition(models):
    """FieldCondition restricting document_id to the scope, or None when unscoped."""
    scope = _scope.get()
    if not scope:
        return None
    return models.FieldCondition(key="document_id", match=models.MatchAny(any=sorted(scope)))