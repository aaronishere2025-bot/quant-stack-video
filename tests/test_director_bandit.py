"""Tests for LLMDirector bandit integration (no GPU required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_segment_directive_has_arm_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as bm
    reload(bm)
    bm.prompt_bandit = bm.WanPromptBandit()
    from src.llm.director import LLMDirector
    director = LLMDirector("a samurai in a bamboo forest", engine="ltx")
    directive = director.next_segment(0)
    assert isinstance(directive.arm_ids, dict)
    assert len(directive.arm_ids) == 5  # domain, subject, lighting, quality, atmosphere


def test_next_segment_prompt_is_longer_than_base(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as bm
    reload(bm)
    bm.prompt_bandit = bm.WanPromptBandit()
    from src.llm.director import LLMDirector
    director = LLMDirector("a samurai", engine="ltx")
    directive = director.next_segment(0)
    # Bandit enhancement appends comma-separated elements — prompt must be longer
    assert len(directive.prompt) > len("a samurai, cinematic, high quality, detailed")


def test_record_segment_quality_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as bm
    reload(bm)
    bm.prompt_bandit = bm.WanPromptBandit()
    from src.llm.director import LLMDirector
    director = LLMDirector("a samurai", engine="ltx")
    directive = director.next_segment(0)
    director.record_segment_quality(directive.arm_ids, score=7.0)  # must not raise


def test_fallback_segment_also_has_arm_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as bm
    reload(bm)
    bm.prompt_bandit = bm.WanPromptBandit()
    from src.llm.director import LLMDirector
    director = LLMDirector("a samurai", engine="ltx")
    directive = director.next_segment(1)  # no LLM → fallback path
    assert isinstance(directive.arm_ids, dict)
    assert len(directive.arm_ids) > 0
