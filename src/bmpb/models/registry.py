"""Model families, keyed by the `family` field of an experiment config.

Adding a model to the comparison means adding a YAML file under configs/, not
editing training code. A family is a callable that takes an ExperimentConfig and
returns `(model, processor)`, where `processor` is whatever turns raw text and
images into tensors (a tokenizer, an image processor, or a joint processor).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bmpb.config import ExperimentConfig

_REGISTRY: dict[str, Callable[[ExperimentConfig], tuple[Any, Any]]] = {}


def register(name: str) -> Callable:
    def decorator(fn: Callable[[ExperimentConfig], tuple[Any, Any]]):
        if name in _REGISTRY:
            raise ValueError(f"family {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def build(cfg: ExperimentConfig) -> tuple[Any, Any]:
    # Import for side effects: each module registers its families on import.
    from bmpb.models import baselines, multimodal, text  # noqa: F401

    if cfg.family not in _REGISTRY:
        raise KeyError(f"unknown family {cfg.family!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.family](cfg)


def available() -> list[str]:
    from bmpb.models import baselines, multimodal, text  # noqa: F401

    return sorted(_REGISTRY)
