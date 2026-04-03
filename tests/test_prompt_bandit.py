"""Unit tests for Thompson sampling prompt bandit (no GPU required)."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_build_enhanced_prompt_starts_with_base(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as m
    reload(m)
    bandit = m.WanPromptBandit()
    prompt, _ = bandit.build_enhanced_prompt("a lone samurai", engine="ltx")
    assert prompt.startswith("a lone samurai, ")


def test_ltx_arm_ids_cover_all_categories(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as m
    reload(m)
    bandit = m.WanPromptBandit()
    _, arm_ids = bandit.build_enhanced_prompt("test", engine="ltx")
    assert set(arm_ids.keys()) == {"domain", "subject", "lighting", "quality", "atmosphere"}


def test_wan_arm_ids_cover_all_categories(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as m
    reload(m)
    bandit = m.WanPromptBandit()
    _, arm_ids = bandit.build_enhanced_prompt("test", engine="wan")
    assert set(arm_ids.keys()) == {"domain", "camera", "lighting", "style", "motion"}


def test_update_reward_increases_alpha_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as m
    reload(m)
    bandit = m.WanPromptBandit()
    _, arm_ids = bandit.build_enhanced_prompt("test", engine="ltx")
    chosen_idx = arm_ids["domain"]
    alpha_before = bandit._bandit_for("ltx")._state["categories"]["domain"][chosen_idx]["alpha"]
    bandit.update_reward(arm_ids, quality_score=8.0, engine="ltx")
    alpha_after = bandit._bandit_for("ltx")._state["categories"]["domain"][chosen_idx]["alpha"]
    assert alpha_after > alpha_before


def test_update_reward_increases_beta_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as m
    reload(m)
    bandit = m.WanPromptBandit()
    _, arm_ids = bandit.build_enhanced_prompt("test", engine="ltx")
    chosen_idx = arm_ids["domain"]
    beta_before = bandit._bandit_for("ltx")._state["categories"]["domain"][chosen_idx]["beta"]
    bandit.update_reward(arm_ids, quality_score=3.0, engine="ltx")
    beta_after = bandit._bandit_for("ltx")._state["categories"]["domain"][chosen_idx]["beta"]
    assert beta_after > beta_before


def test_state_persists_to_json(tmp_path, monkeypatch):
    from importlib import reload
    import src.llm.prompt_bandit as m
    reload(m)
    monkeypatch.setattr(m, "BANDIT_DIR", tmp_path)
    bandit = m.WanPromptBandit()
    _, arm_ids = bandit.build_enhanced_prompt("test", engine="ltx")
    bandit.update_reward(arm_ids, 7.0, engine="ltx")
    state_file = tmp_path / "ltx-prompt-bandit.json"
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert "categories" in state
    assert state["totalGenerations"] == 1


def test_gamma_decay_reduces_pumped_alpha(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as m
    reload(m)
    bandit = m.WanPromptBandit()
    bandit._bandit_for("ltx")._state["categories"]["domain"][0]["alpha"] = 5.0
    bandit.apply_gamma_decay(engine="ltx")
    alpha_after = bandit._bandit_for("ltx")._state["categories"]["domain"][0]["alpha"]
    assert alpha_after < 5.0


def test_unknown_engine_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.prompt_bandit.BANDIT_DIR", tmp_path)
    from importlib import reload
    import src.llm.prompt_bandit as m
    reload(m)
    bandit = m.WanPromptBandit()
    with pytest.raises(ValueError, match="Unknown engine"):
        bandit.build_enhanced_prompt("test", engine="stable-diffusion")
