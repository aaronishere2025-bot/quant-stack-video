# V2 Architecture Research: Layered Infinite Video Generation

**Status**: Research / Planning  
**Last updated**: 2026-03-31

---

## Overview

V2 extends the quantization stacking engine (v1) into a full infinite layered video system. The key innovations are:

1. **RGBA layer generation** — 3 independent transparent layers composited together
2. **VACE temporal extension** — seamless segment-to-segment continuity
3. **SVI error recycling** — prevents quality drift over infinite iterations
4. **LLM continuity agent** — "virtual director" for 3-minute+ coherent narratives

---

## Phase 1: RGBA Generation with Wan-Alpha

### Model
- **Wan-Alpha** (Wan 2.1 variant trained on RGBA data with alpha channel)
- Generates 4-channel video: RGB + alpha matte
- Three passes: background layer, midground layer, foreground layer

### Quantization
- Same stacking engine as v1 (NF4/FP4 4-bit BnB or GGUF Q4_0/Q3_K_M)
- GGUF Q4_0 vs BnB 4-bit tradeoff: **TODO benchmark on Wan-Alpha**

### Alpha Quality
- Target: clean mattes with no edge boiling
- Key problem: RGBA jitter between frames where alpha changes value
- Mitigation: temporal smoothing in alpha channel before compositing

### Tensor Shape
```python
# RGBA video tensor
layer: torch.Tensor  # shape [B, 4, F, H, W], dtype float32, range [0, 1]
# B=batch, 4=RGBA channels, F=frames, H=height, W=width
```

---

## Phase 2: Alpha Compositing

Standard Porter-Duff "over" operation:

```python
def composite_over(top: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
    """
    Args:
        top, bottom: [B, 4, F, H, W] RGBA tensors, values in [0, 1]
    Returns:
        [B, 4, F, H, W] composited RGBA tensor
    """
    top_rgb, top_a = top[:, :3], top[:, 3:4]
    bot_rgb, bot_a = bottom[:, :3], bottom[:, 3:4]

    out_a = top_a + bot_a * (1 - top_a)
    out_rgb = (top_rgb * top_a + bot_rgb * bot_a * (1 - top_a)) / (out_a + 1e-8)
    return torch.cat([out_rgb, out_a], dim=1)
```

Compositing order: background → midground → foreground (bottom-to-top).

---

## Phase 3: VACE Temporal Extension

### Parameters (validated)
- **Shift = 1** (required for VACE — do not change)
- **CFG = 2.0–3.0** (lower = more consistency, higher = more variation)
- **Overlap = 16 frames** (frames 65–81 of previous segment)

### Mask Convention
```
known frames (from previous segment): mask = black (0)
unknown frames (to generate):         mask = white (1)
padding frames:                       pixel = #7F7F7F grey
```

### Latent-Space Handoff (Critical)
**Never decode + re-encode for segment handoffs.** This causes color shift.
- Keep the last N latents from the previous segment
- Pass them directly to VACE as conditioning latents
- Only decode to pixel space for final output

### Slerp vs VACE Overlap
- **VACE overlap** (bidirectional, 16 frames): better physics/momentum continuity
- **Slerp blending**: simpler but can produce ghosting artifacts
- **Recommendation**: use VACE overlap for physics-heavy content, slerp for static scenes

---

## Phase 4: SVI Error Recycling

### Problem
Autoregressive video generation accumulates error over iterations:
- Each segment conditions on the previous segment's output
- Quantization error compounds across segments
- After ~10 segments, quality degrades noticeably (drift)

### SVI Solution
SVI (Score-based Video Inference) error recycling:
1. During generation of segment N, capture the DiT prediction errors (residuals)
2. Store these errors as a running buffer
3. When generating segment N+1, inject the buffered errors as a correction term into the flow matching denoiser
4. This "recycles" the error signal, counteracting drift

### Implementation Note
SVI-Shot variant for single-scene generation:
- Only recycle errors within the same scene context
- Reset error buffer when the LLM continuity agent signals a scene change

---

## Phase 5: LLM Continuity Agent

### Role
"Virtual director" that ensures narrative coherence across 3-minute+ videos.

### State Tracking
The agent maintains:
```yaml
narrative_state:
  scene_number: int
  current_location: str
  current_characters: list[str]
  mood: str
  time_of_day: str
  recent_actions: list[str]
  pending_story_beats: list[str]
```

### Prompt Generation
For each new segment:
1. LLM receives the current narrative state
2. Generates a prompt for the next segment that advances the story
3. Updates narrative state with what happened

### Character Consistency
- **EchoShot** for maintaining consistent character appearance across segments
- Reference images from early segments used to condition later segments

### Integration Points
- Segment prompt → Wan-Alpha (Phase 1)
- Scene change signal → SVI error buffer reset (Phase 4)

---

## VRAM Budget (RTX 4070, 12 GB)

| Component | VRAM (4-bit) | VRAM (bf16) |
|-----------|-------------|-------------|
| Wan 2.1 14B transformer | ~7 GB | ~28 GB |
| VAE (bfloat16) | ~1.5 GB | ~1.5 GB |
| T5 text encoder (CPU) | 0 | 0 |
| VACE conditioning | ~0.5 GB | ~0.5 GB |
| Activations | ~2 GB | ~3+ GB |
| **Total (4-bit)** | **~11 GB** | **>30 GB** |

4-bit quantization is the only feasible path on a 12 GB card. This is why stacking matters: we need quality recovery without needing more VRAM.

---

## Open Questions / TODO

- [ ] Benchmark GGUF Q4_0 vs BnB NF4 for Wan-Alpha quality/VRAM tradeoff
- [ ] Measure VACE overlap quality vs slerp on physics-heavy content
- [ ] Test SVI error recycling over 20+ segments to measure drift reduction
- [ ] Evaluate EchoShot character consistency across 10+ segments
- [ ] Test TeaCache + SageAttention2 speedup on Wan 2.1 1.3B vs 14B
- [ ] ComfyUI integration for node-based workflow

---

## References

- Wan 2.1: [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)
- VACE: Video-conditioned Action and Content Editing
- SVI: Score-based Video Inference
- BitsAndBytes: [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- EchoShot: Multi-shot character consistency for video diffusion
