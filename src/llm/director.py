"""
LLM Continuity Director — Phase 5 of the layered infinite video pipeline.

The "virtual director" generates evolving prompts per segment, tracks narrative
state across an infinite video, and signals scene changes to the SVI recycler.

Design:
  - Stateful: maintains a NarrativeState that evolves across segments
  - LLM-agnostic: calls any OpenAI-compatible API (local Ollama, OpenAI, Anthropic)
  - Deterministic fallback: if no LLM is configured, advances prompt via simple templates
  - EchoShot-aware: exposes character reference fields for downstream consistency

Narrative state tracks:
  scene_number, current_location, characters, mood, time_of_day,
  recent_actions, pending_story_beats
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class NarrativeState:
    """Tracks story continuity across video segments."""
    scene_number: int = 0
    current_location: str = ""
    current_characters: List[str] = field(default_factory=list)
    mood: str = "neutral"
    time_of_day: str = "day"
    recent_actions: List[str] = field(default_factory=list)
    pending_story_beats: List[str] = field(default_factory=list)
    is_scene_change: bool = False  # Set True when this segment starts a new scene

    def to_context_str(self) -> str:
        """Format state as a compact context string for LLM prompting."""
        parts = [
            f"Scene {self.scene_number}",
            f"Location: {self.current_location}" if self.current_location else None,
            f"Characters: {', '.join(self.current_characters)}" if self.current_characters else None,
            f"Mood: {self.mood}",
            f"Time: {self.time_of_day}",
        ]
        if self.recent_actions:
            parts.append(f"Recent: {'; '.join(self.recent_actions[-3:])}")
        if self.pending_story_beats:
            parts.append(f"Next beat: {self.pending_story_beats[0]}")
        return " | ".join(p for p in parts if p)


@dataclass
class SegmentDirective:
    """Output from the director for a single video segment."""
    prompt: str                          # Generation prompt for this segment
    negative_prompt: Optional[str] = None
    state: Optional[NarrativeState] = None   # Updated narrative state after this segment
    is_scene_change: bool = False        # Whether this segment starts a new scene
    character_refs: List[str] = field(default_factory=list)  # Paths to EchoShot reference images
    segment_idx: int = 0


@dataclass
class DirectorConfig:
    """Configuration for the LLM director."""
    # LLM connection (OpenAI-compatible endpoint)
    api_base: Optional[str] = None       # e.g. "http://localhost:11434/v1" for Ollama
    api_key: str = "ollama"              # API key (ignored for local)
    model: str = "llama3.2"              # Model name

    # Behavior
    max_recent_actions: int = 10         # How many recent actions to keep in state
    scene_change_probability: float = 0.1  # Probability of scene change per segment
    temperature: float = 0.7
    max_tokens: int = 300

    # Fallback (used when no LLM configured or LLM call fails)
    use_static_prompt_fallback: bool = True
    fallback_prompt_suffix: str = ", cinematic, high quality, detailed"

    @property
    def has_llm(self) -> bool:
        return self.api_base is not None


_SYSTEM_PROMPT = """\
You are a cinematic video director. Your job is to generate a short text prompt
for the next 5-second video segment in an ongoing continuous video.

You will receive the current narrative state and must output a JSON object with:
- "prompt": the generation prompt for this segment (1-3 sentences, vivid and specific)
- "negative_prompt": what to avoid (optional)
- "updated_location": updated location if changed
- "updated_mood": updated mood/tone
- "updated_time_of_day": updated time if progressed
- "action_summary": one sentence describing what happened in this segment
- "next_story_beat": what should happen in the following segment
- "is_scene_change": true if this segment starts a completely new scene

Keep prompts consistent with previous segments. Advance the story naturally.
Output ONLY valid JSON, no markdown, no explanation.
"""


class LLMDirector:
    """
    Virtual director for coherent long-form video narrative.

    Generates per-segment prompts by querying an LLM with the current narrative
    state and producing a directive for the next segment.

    When no LLM is configured (api_base=None), uses a deterministic fallback
    that suffixes the base prompt with state-derived context.
    """

    def __init__(self, base_prompt: str, config: Optional[DirectorConfig] = None):
        """
        Args:
            base_prompt: The initial/anchor prompt for the video (e.g. "A mountain hike at dawn")
            config:      Director configuration. If None, uses defaults (LLM disabled).
        """
        self.base_prompt = base_prompt
        self.config = config or DirectorConfig()
        self._state = NarrativeState(
            scene_number=0,
            current_location=self._extract_location_hint(base_prompt),
            mood="cinematic",
            time_of_day="day",
        )
        self._segment_history: List[SegmentDirective] = []

    @property
    def current_state(self) -> NarrativeState:
        return self._state

    @property
    def segment_count(self) -> int:
        return len(self._segment_history)

    def _extract_location_hint(self, prompt: str) -> str:
        """Best-effort: extract a location from the initial prompt."""
        # Simple heuristic — good enough for first-segment initialization
        words = prompt.lower().split()
        location_keywords = ["mountain", "forest", "city", "ocean", "beach", "street",
                              "room", "house", "field", "desert", "jungle", "lake"]
        for kw in location_keywords:
            if kw in words:
                return kw
        return ""

    def next_segment(self, segment_idx: int) -> SegmentDirective:
        """
        Generate a directive for the next video segment.

        Args:
            segment_idx: 0-based index of the upcoming segment

        Returns:
            SegmentDirective with prompt and updated state
        """
        if segment_idx == 0:
            # First segment always uses the base prompt directly
            directive = SegmentDirective(
                prompt=self.base_prompt + self.config.fallback_prompt_suffix,
                state=self._state,
                is_scene_change=False,
                segment_idx=0,
            )
            self._segment_history.append(directive)
            return directive

        if self.config.has_llm:
            try:
                directive = self._query_llm(segment_idx)
                self._segment_history.append(directive)
                return directive
            except Exception as exc:
                logger.warning("LLM query failed (segment %d): %s — using fallback", segment_idx, exc)

        directive = self._fallback_directive(segment_idx)
        self._segment_history.append(directive)
        return directive

    def _query_llm(self, segment_idx: int) -> SegmentDirective:
        """Query the LLM to generate the next segment directive."""
        import urllib.request

        context = self._state.to_context_str()
        user_msg = (
            f"Base video concept: {self.base_prompt}\n"
            f"Current state: {context}\n"
            f"Segment index: {segment_idx}\n"
            "Generate the next segment directive as JSON."
        )

        payload = json.dumps({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }).encode()

        url = self.config.api_base.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        content = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        return self._parse_llm_response(parsed, segment_idx)

    def _parse_llm_response(self, parsed: Dict[str, Any], segment_idx: int) -> SegmentDirective:
        """Parse LLM JSON response into a SegmentDirective and update narrative state."""
        is_scene_change = bool(parsed.get("is_scene_change", False))

        # Update narrative state
        new_state = NarrativeState(
            scene_number=self._state.scene_number + (1 if is_scene_change else 0),
            current_location=parsed.get("updated_location", self._state.current_location),
            current_characters=self._state.current_characters,
            mood=parsed.get("updated_mood", self._state.mood),
            time_of_day=parsed.get("updated_time_of_day", self._state.time_of_day),
            recent_actions=(
                self._state.recent_actions + [parsed.get("action_summary", "")]
            )[-self.config.max_recent_actions:],
            pending_story_beats=(
                [parsed.get("next_story_beat")]
                if parsed.get("next_story_beat")
                else self._state.pending_story_beats[1:]
            ),
            is_scene_change=is_scene_change,
        )
        self._state = new_state

        return SegmentDirective(
            prompt=parsed["prompt"],
            negative_prompt=parsed.get("negative_prompt"),
            state=new_state,
            is_scene_change=is_scene_change,
            segment_idx=segment_idx,
        )

    def _fallback_directive(self, segment_idx: int) -> SegmentDirective:
        """
        Deterministic fallback when no LLM is configured or LLM call fails.

        Advances the prompt by appending state context. No scene changes in fallback mode.
        """
        context_parts = [self.base_prompt]

        if self._state.current_location:
            context_parts.append(f"in the {self._state.current_location}")
        if self._state.mood and self._state.mood != "neutral":
            context_parts.append(f"{self._state.mood} atmosphere")
        if self._state.time_of_day:
            context_parts.append(self._state.time_of_day)

        # Slowly evolve time of day
        time_progression = ["dawn", "morning", "day", "afternoon", "dusk", "evening", "night"]
        try:
            current_idx = time_progression.index(self._state.time_of_day)
            if segment_idx % 6 == 0 and current_idx < len(time_progression) - 1:
                self._state.time_of_day = time_progression[current_idx + 1]
        except ValueError:
            pass

        prompt = ", ".join(context_parts) + self.config.fallback_prompt_suffix

        return SegmentDirective(
            prompt=prompt,
            state=self._state,
            is_scene_change=False,
            segment_idx=segment_idx,
        )

    def get_history_summary(self) -> List[Dict[str, Any]]:
        """Return a compact summary of all generated segment directives."""
        return [
            {
                "segment_idx": d.segment_idx,
                "prompt": d.prompt[:80] + "..." if len(d.prompt) > 80 else d.prompt,
                "is_scene_change": d.is_scene_change,
            }
            for d in self._segment_history
        ]
