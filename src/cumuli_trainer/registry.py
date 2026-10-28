# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Plugin registry.

Four kinds of plugin, each chosen by name in the run config:

    representation   what the Gaussians are and how they are sliced in time
    strategy         densification, pruning and relocation
    loss             one weighted loss term
    renderer         rasterises a sliced set of Gaussians

Built-in plugins register with the ``register`` decorator when their module is
imported. External packages register through entry points, one group per kind:

    [project.entry-points."cumuli_trainer.strategies"]
    mrnf = "my_private_pkg.mrnf:MRNFStrategy"

Entry points are read the first time a name is looked up and not found, so a
private package only has to be installed into the same environment. The
training loop only ever calls ``get(kind, name)``; it never imports a plugin
module itself.
"""

from __future__ import annotations

from importlib import metadata
from typing import Callable, Dict, Type

KINDS = ("representation", "strategy", "loss", "renderer")

ENTRY_POINT_GROUPS = {
    "representation": "cumuli_trainer.representations",
    "strategy": "cumuli_trainer.strategies",
    "loss": "cumuli_trainer.losses",
    "renderer": "cumuli_trainer.renderers",
}

_REGISTRY: Dict[str, Dict[str, type]] = {kind: {} for kind in KINDS}
_ENTRY_POINTS_LOADED: Dict[str, bool] = {kind: False for kind in KINDS}


class RegistryError(KeyError):
    """Unknown kind or name."""


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise RegistryError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")


def register(kind: str, name: str) -> Callable[[Type], Type]:
    """Class decorator: register ``cls`` as plugin ``name`` of ``kind``.

    Registering the same class twice under one name is a no-op (a module can
    be imported both directly and through its entry point). Registering a
    different class under a taken name is an error.
    """
    _check_kind(kind)

    def deco(cls: Type) -> Type:
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not cls and _qualname(existing) != _qualname(cls):
            raise RegistryError(
                f"{kind} {name!r} is already registered to {_qualname(existing)}; "
                f"refusing to replace it with {_qualname(cls)}")
        _REGISTRY[kind][name] = cls
        cls.plugin_name = name
        cls.plugin_kind = kind
        return cls

    return deco


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _load_entry_points(kind: str) -> None:
    if _ENTRY_POINTS_LOADED[kind]:
        return
    _ENTRY_POINTS_LOADED[kind] = True
    group = ENTRY_POINT_GROUPS[kind]
    try:
        eps = metadata.entry_points(group=group)
    except TypeError:  # Python < 3.10 signature
        eps = metadata.entry_points().get(group, [])
    for ep in eps:
        if ep.name in _REGISTRY[kind]:
            continue
        try:
            cls = ep.load()
        except Exception as exc:  # a broken third-party plugin must not hide the others
            print(f"warning: could not load {kind} plugin {ep.name!r} ({ep.value}): {exc}")
            continue
        _REGISTRY[kind].setdefault(ep.name, cls)
        if not hasattr(cls, "plugin_name"):
            cls.plugin_name = ep.name
            cls.plugin_kind = kind


def get(kind: str, name: str) -> type:
    """The class registered as ``name`` for ``kind``."""
    _check_kind(kind)
    if name not in _REGISTRY[kind]:
        _load_entry_points(kind)
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise RegistryError(
            f"no {kind} plugin named {name!r}; available: {sorted(available(kind))}") from None


def available(kind: str) -> list[str]:
    """Every registered name for ``kind``, including entry points."""
    _check_kind(kind)
    _load_entry_points(kind)
    return sorted(_REGISTRY[kind])


def create(kind: str, name: str, *args, **kwargs):
    """Instantiate plugin ``name`` of ``kind``."""
    return get(kind, name)(*args, **kwargs)


def unregister(kind: str, name: str) -> None:
    """Remove a registration. For tests."""
    _check_kind(kind)
    _REGISTRY[kind].pop(name, None)
