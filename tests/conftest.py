"""Stubs out heavy optional dependencies (mem0, FlagEmbedding/torch,
weaviate-client, llama_cloud, elevenlabs) so that unit tests exercising pure
business logic (validators, prompt formatting, etc.) can run fast in CI
without installing the full ML/vector-db stack.

Integration tests that need the real services should install the full
dependency set (`uv sync`) and mark themselves accordingly (not provided
here -- see tests/test_health.py for the pattern of skipping when real
credentials aren't configured).
"""

from __future__ import annotations

import sys
import types


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


class _DummyMemory:
    @classmethod
    def from_config(cls, cfg):
        return cls()


_stub_module("mem0", Memory=_DummyMemory)
