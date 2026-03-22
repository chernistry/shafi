"""Internal pipeline helper types."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, cast


def get_pipeline_helper_instance(owner: type[Any]) -> Any:
    """Return a lightweight builder-shaped helper for class-level helper calls.

    The historical pipeline API exposes many underscore helpers directly on
    ``RAGPipelineBuilder`` and tests call them via the class. After splitting the
    hot path into free-function modules, those helpers still need a pipeline-like
    object for internal composition. This function creates a cached uninitialized
    owner instance with only the minimal settings needed by pure helper paths.

    Args:
        owner: Pipeline builder class that owns the helper surface.

    Returns:
        A cached helper instance suitable for calling split helper methods.
    """
    instance = cast("Any", getattr(owner, "__compat_helper_instance__", None))
    if instance is None:
        instance = cast("Any", object.__new__(cast("type[object]", owner)))
        instance._settings = _SimpleNamespace(pipeline=_SimpleNamespace(strict_types_append_citations=False))
        cast("Any", owner).__compat_helper_instance__ = instance
    return instance


class CompatClassOrInstanceMethod:
    """Descriptor that preserves class-level helper calls after method extraction.

    Access via an instance behaves like a normal bound method. Access via the
    class resolves the call against a lightweight cached helper instance so
    historical ``RAGPipelineBuilder._helper(...)`` tests keep working.

    Args:
        target_name: Public helper attribute name on the owning class.
    """

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def __get__(self, instance: Any | None, owner: type[Any]) -> Any:
        target_owner = instance if instance is not None else get_pipeline_helper_instance(owner)
        return getattr(target_owner, self._target_name)


@dataclass(frozen=True)
class SupportChunkMaps:
    """Snippet and page-hint maps for support localization."""

    snippets: dict[str, str]
    page_hints: dict[str, str]


@dataclass(frozen=True)
class SupportLocalizationResult:
    """Localized support selection with citations and page ids."""

    chunk_ids: list[str]
    citations: list[str]
    page_ids: list[str]
