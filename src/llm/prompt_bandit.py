"""
Thompson Sampling Prompt Bandit for video generation.

Python port of unity-repo/server/services/wan-prompt-bandit.ts.
State files use the same JSON format as the TypeScript version —
data/bandit/wan-prompt-bandit.json and ltx-prompt-bandit.json
are cross-language compatible.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BANDIT_DIR = Path(__file__).parent.parent.parent / "data" / "bandit"
DEFAULT_GAMMA = 0.97

_DOMAIN_ARMS = [
    "music video aesthetic, beat-driven visual energy",
    "lofi aesthetic, nostalgic study-beats vibe",
    "trap / urban streetwear culture visual",
    "marketing / brand identity, clean professional look",
    "history / documentary, archival period aesthetic",
    "science / educational, clear informative composition",
    "nature / travel, breathtaking landscape visual",
    "abstract art, generative motion graphic",
    "cinematic narrative, story-driven scene",
    "wellness / meditation, calm therapeutic visual",
]

_DEFAULT_ARMS_WAN: Dict[str, list] = {
    "domain": _DOMAIN_ARMS,
    "camera": [
        "static wide shot", "slow pan left to right", "slow zoom in",
        "tracking shot following subject", "low angle looking up",
        "aerial establishing shot", "close-up detail shot", "dolly zoom effect",
    ],
    "lighting": [
        "golden hour warm light", "dramatic side lighting", "neon glow cyberpunk",
        "soft overcast diffused", "harsh midday sun", "candlelit warm interior",
        "blue hour twilight", "studio rim lighting",
    ],
    "style": [
        "cinematic film grain", "photorealistic 4k", "oil painting aesthetic",
        "watercolor dreamlike", "noir high contrast", "anime cel-shaded",
        "vintage 1970s footage", "clean modern minimal",
    ],
    "motion": [
        "smooth fluid movement", "subtle parallax layers", "dynamic fast cuts",
        "slow motion dramatic", "steady locked-off frame", "handheld documentary feel",
    ],
}

_DEFAULT_ARMS_LTX: Dict[str, list] = {
    "domain": _DOMAIN_ARMS,
    "subject": [
        "intricate fabric texture, fine detail", "weathered stone surface, moss-covered",
        "glowing bioluminescent particles", "polished chrome and glass reflections",
        "dense forest undergrowth, dappled light", "cracked earth, dry desert texture",
        "flowing water surface, caustics", "neon-lit urban street, rain-slicked pavement",
    ],
    "lighting": [
        "warm amber backlight, rim highlight", "cool blue moonlight, deep shadows",
        "overcast soft diffused light, even tones", "harsh direct sunlight, high contrast",
        "orange fire glow, flickering light", "purple twilight, gradient sky",
        "neon pink and cyan fill light", "golden sunrise haze, lens flare",
    ],
    "quality": [
        "hyperrealistic 4K render", "photographic, sharp focus",
        "cinematic depth of field, bokeh", "painterly impasto texture",
        "high-detail macro photography", "film grain 35mm analog",
        "ultra-sharp microscopic detail", "soft dreamlike illustration",
    ],
    "atmosphere": [
        "morning mist, ethereal fog", "dust particles in air, volumetric light",
        "heavy rain, puddle reflections", "dry heat shimmer, desert haze",
        "underwater caustics, blue tint", "smoke and embers drifting",
        "clear crisp air, high altitude", "humid jungle atmosphere",
    ],
}

_DEFAULT_ARMS = {"wan": _DEFAULT_ARMS_WAN, "ltx": _DEFAULT_ARMS_LTX}


class EnginePromptBandit:
    """Per-engine Thompson sampling bandit with Beta distribution arm selection."""

    def __init__(self, engine: str) -> None:
        if engine not in _DEFAULT_ARMS:
            raise ValueError(f"Unknown engine: {engine!r}. Expected one of {list(_DEFAULT_ARMS)}")
        self.engine = engine
        self._state = self._load_state()

    def _persistence_path(self) -> Path:
        return BANDIT_DIR / f"{self.engine}-prompt-bandit.json"

    def _load_state(self) -> dict:
        path = self._persistence_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                logger.warning("[%s Bandit] Corrupted state file, reinitializing", self.engine.upper())
        categories = {
            cat: [
                {"name": name, "alpha": 1.0, "beta": 1.0, "pulls": 0, "totalReward": 0.0}
                for name in arms
            ]
            for cat, arms in _DEFAULT_ARMS[self.engine].items()
        }
        return {"categories": categories, "totalGenerations": 0, "gamma": DEFAULT_GAMMA}

    def persist(self) -> None:
        BANDIT_DIR.mkdir(parents=True, exist_ok=True)
        self._persistence_path().write_text(json.dumps(self._state, indent=2))

    def select_prompt_elements(self) -> Tuple[Dict[str, str], Dict[str, int]]:
        elements: Dict[str, str] = {}
        arm_ids: Dict[str, int] = {}
        for cat, arms in self._state["categories"].items():
            samples = [np.random.beta(a["alpha"], a["beta"]) for a in arms]
            best_idx = int(np.argmax(samples))
            elements[cat] = arms[best_idx]["name"]
            arm_ids[cat] = best_idx
            arms[best_idx]["pulls"] += 1
        self._state["totalGenerations"] += 1
        return elements, arm_ids

    def build_enhanced_prompt(self, base_prompt: str) -> Tuple[str, Dict[str, int]]:
        elements, arm_ids = self.select_prompt_elements()
        enhancement = ", ".join(elements.values())
        prompt = f"{base_prompt}, {enhancement}"
        logger.debug("[%s Bandit] %s", self.engine.upper(),
                     " | ".join(f'{k}="{v}"' for k, v in elements.items()))
        return prompt, arm_ids

    def update_reward(self, arm_ids: Dict[str, int], quality_score: float,
                      threshold: float = 6.0) -> None:
        success = quality_score >= threshold
        for cat, idx in arm_ids.items():
            arms = self._state["categories"].get(cat, [])
            if not arms or idx >= len(arms):
                continue
            if success:
                arms[idx]["alpha"] += quality_score / 10.0
            else:
                arms[idx]["beta"] += (10.0 - quality_score) / 10.0
            arms[idx]["totalReward"] += quality_score
        self.persist()
        logger.debug("[%s Bandit] score=%.1f %s", self.engine.upper(),
                     quality_score, "success" if success else "failure")

    def apply_gamma_decay(self) -> None:
        gamma = self._state["gamma"]
        for arms in self._state["categories"].values():
            for arm in arms:
                arm["alpha"] = max(1.0, arm["alpha"] * gamma)
                arm["beta"] = max(1.0, arm["beta"] * gamma)
        self.persist()

    def get_stats(self) -> dict:
        cats = {
            cat: sorted(
                [{"name": a["name"], "winRate": a["alpha"] / (a["alpha"] + a["beta"]),
                  "pulls": a["pulls"]} for a in arms],
                key=lambda x: -x["winRate"],
            )
            for cat, arms in self._state["categories"].items()
        }
        return {"engine": self.engine, "totalGenerations": self._state["totalGenerations"],
                "categories": cats}


class WanPromptBandit:
    """Multi-engine facade. Routes calls to the correct per-engine bandit."""

    def __init__(self) -> None:
        self._bandits: Dict[str, EnginePromptBandit] = {
            "wan": EnginePromptBandit("wan"),
            "ltx": EnginePromptBandit("ltx"),
        }

    def _bandit_for(self, engine: str) -> EnginePromptBandit:
        if engine not in self._bandits:
            raise ValueError(f"Unknown engine: {engine!r}. Expected 'wan' or 'ltx'.")
        return self._bandits[engine]

    def build_enhanced_prompt(self, base_prompt: str,
                               engine: str = "wan") -> Tuple[str, Dict[str, int]]:
        return self._bandit_for(engine).build_enhanced_prompt(base_prompt)

    def update_reward(self, arm_ids: Dict[str, int], quality_score: float,
                      engine: str = "wan", threshold: float = 6.0) -> None:
        self._bandit_for(engine).update_reward(arm_ids, quality_score, threshold)

    def apply_gamma_decay(self, engine: Optional[str] = None) -> None:
        if engine:
            self._bandit_for(engine).apply_gamma_decay()
        else:
            for b in self._bandits.values():
                b.apply_gamma_decay()

    def get_stats(self, engine: Optional[str] = None) -> dict:
        if engine:
            return self._bandit_for(engine).get_stats()
        return {e: b.get_stats() for e, b in self._bandits.items()}


# Module-level singleton — import and use directly
prompt_bandit = WanPromptBandit()
