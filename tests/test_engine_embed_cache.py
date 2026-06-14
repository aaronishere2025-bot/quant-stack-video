"""The stacked engine encodes the prompt once and reuses it across passes.

UMT5-XXL CPU encoding is the dominant per-pass cost; since the prompt is identical
across stacking passes, _prompt_kwargs caches the encoded embeds (keyed by
prompt/neg/cfg) so only the first pass pays for the encode. Regression for the
2026-06-14 perf work (575s for a 2-pass job was ~all re-encoding).
"""
from src.quant.config import StackConfig
from src.quant.engine import QuantStackEngine


class _Fake:
    """Stands in for both a tensor (.to → self) and the text encoder (.to → self)."""
    def to(self, *args, **kwargs):
        return self


class _FakePipe:
    def __init__(self, counter):
        self._counter = counter
        self.text_encoder = _Fake()

    def encode_prompt(self, prompt, negative_prompt, do_classifier_free_guidance,
                      num_videos_per_prompt, device):
        self._counter["n"] += 1
        neg = _Fake() if do_classifier_free_guidance else None
        return _Fake(), neg


def test_embeds_encoded_once_across_passes_and_per_distinct_prompt():
    eng = QuantStackEngine(StackConfig())
    counter = {"n": 0}
    pipe = _FakePipe(counter)

    # Same prompt, three passes (each loads a fresh pipe in production) → encode once.
    for _ in range(3):
        kw = eng._prompt_kwargs(pipe, "a calm lake at dawn", "", 5.0)
        assert "prompt_embeds" in kw
        assert "negative_prompt_embeds" in kw  # cfg 5.0 > 1 → cfg on
    assert counter["n"] == 1, "prompt should be encoded exactly once across passes"

    # A different prompt is a cache miss → one more encode.
    eng._prompt_kwargs(pipe, "a mountain at dusk", "", 5.0)
    assert counter["n"] == 2

    # Back to the first prompt → still cached, no new encode.
    eng._prompt_kwargs(pipe, "a calm lake at dawn", "", 5.0)
    assert counter["n"] == 2

    # run_stacked clears the cache (memory hygiene) so a reused engine re-encodes.
    eng._embed_cache.clear()
    eng._prompt_kwargs(pipe, "a calm lake at dawn", "", 5.0)
    assert counter["n"] == 3
