# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
from importlib import metadata

import pytest

from cumuli_trainer import registry


def test_builtins_registered():
    assert "spacetime" in registry.available("representation")
    assert {"mcmc", "default"} <= set(registry.available("strategy"))
    assert {"l1", "ssim", "opacity_mask", "opacity_reg", "scale_reg"} <= set(registry.available("loss"))
    assert "gsplat" in registry.available("renderer")


def test_register_and_get():
    @registry.register("strategy", "unit_test_strategy")
    class S:
        pass
    try:
        assert registry.get("strategy", "unit_test_strategy") is S
        assert S.plugin_name == "unit_test_strategy"
        assert registry.create("strategy", "unit_test_strategy").__class__ is S
    finally:
        registry.unregister("strategy", "unit_test_strategy")


def test_name_clash_is_an_error():
    @registry.register("loss", "unit_test_clash")
    class A:
        pass
    try:
        with pytest.raises(registry.RegistryError):
            @registry.register("loss", "unit_test_clash")
            class B:
                pass
    finally:
        registry.unregister("loss", "unit_test_clash")


def test_unknown_kind_and_name():
    with pytest.raises(registry.RegistryError):
        registry.get("optimizer", "adam")
    with pytest.raises(registry.RegistryError, match="available"):
        registry.get("strategy", "does_not_exist")


class _FakeEP:
    def __init__(self, name, obj):
        self.name = name
        self.value = f"fake:{name}"
        self._obj = obj

    def load(self):
        return self._obj


def test_entry_point_plugin_is_found(monkeypatch):
    class External:
        pass

    def fake_entry_points(group=None):
        if group == "cumuli_trainer.strategies":
            return [_FakeEP("external_strategy", External)]
        return []

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    monkeypatch.setitem(registry._ENTRY_POINTS_LOADED, "strategy", False)
    try:
        assert registry.get("strategy", "external_strategy") is External
    finally:
        registry.unregister("strategy", "external_strategy")


def test_broken_entry_point_does_not_hide_others(monkeypatch, capsys):
    class Good:
        pass

    class Broken(_FakeEP):
        def load(self):
            raise ImportError("boom")

    def fake_entry_points(group=None):
        if group == "cumuli_trainer.losses":
            return [Broken("broken_loss", None), _FakeEP("good_loss", Good)]
        return []

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    monkeypatch.setitem(registry._ENTRY_POINTS_LOADED, "loss", False)
    try:
        assert registry.get("loss", "good_loss") is Good
        assert "broken_loss" in capsys.readouterr().out
    finally:
        registry.unregister("loss", "good_loss")
