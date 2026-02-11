# ComfyUI-AutoGuidance

Adds an **AutoGuidance + CFG** **GUIDER** node for ComfyUI.

AutoGuidance steers the sampler by comparing a “good” model path vs a “bad” model path and injecting an extra guidance term (in addition to normal CFG). This implementation is based on the paper:

- `https://arxiv.org/abs/2406.02507`

---

## What this node does (high-level)

Each sampling step, the guider runs:

- **Good path**: your normal CFG inference (positive + negative) using `good_model`
- **Bad path**: an auxiliary inference using `bad_model` (typically “positive only” reference)

Then it forms an **AutoGuidance delta** from the difference between good vs bad predictions (mode-dependent) and adds it to the normal CFG result, with safety limits so it doesn’t explode.

**Why**: you can use a “bad” reference (different LoRAs, different style bias, early draft LoRA, etc.) and push away from it while still following the prompt.

---

## Install

Clone into:

`ComfyUI/custom_nodes/ComfyUI-AutoGuidance`

Restart ComfyUI.

---

## Quick start workflow

1) Load two models (or one model twice, depending on swap mode)
2) Create guider with **AutoGuidance CFG Guider (good+bad)**
3) Feed the guider into **SamplerCustomAdvanced**

Typical setup:
- `good_model`: your normal checkpoint + “good” LoRAs
- `bad_model`: same checkpoint but with “bad” LoRAs (or none), or a deliberately different LoRA stack

---

## Swap modes (speed + VRAM + correctness)

### `dual_models_2x_vram` (FASTEST, but needs 2 real model instances)
This mode is intended to run good+bad as two separate model instances so sampling speed is close to normal CFG.

**CRITICAL: you must force ComfyUI to load the checkpoint twice as two distinct instances.**
ComfyUI will otherwise **deduplicate** and reuse the same underlying model object when the checkpoint path resolves to the same file.

If you do not force two instances, `dual_models_2x_vram` will **definitely behave like a shared-model setup internally**, and performance can become **dramatically worse** (often *much* slower than normal CFG).

✅ Correct way:
- Make a physical copy of the checkpoint (different filename/path) and load each one separately:

  - good_model loads: `MyCheckpoint.safetensors`
  - bad_model loads:  `MyCheckpoint_COPY.safetensors`

(Using symlinks may still dedupe depending on how the path resolves; a real copy is the safest.)

VRAM cost: ~2× the checkpoint VRAM footprint (plus LoRAs).

---

### `shared_safe_low_vram` (MOST COMPATIBLE, can be VERY slow)
Good and bad share the same underlying model instance, and the node swaps patch state (LoRAs/object patches) between them.

This mode is correctness-first, but it can be **much slower than standard CFG** — in worst cases **orders of magnitude slower** — because it does:
- multiple UNet passes per step (good + bad), **and**
- patch/unpatch work to swap LoRA stacks.

Use this only when you cannot afford 2× VRAM or you need maximum compatibility.

---

### `shared_fast_extra_vram` (shared swapping, less overhead, still can be slow)
Still shared-model swapping, but uses more aggressive patcher settings and keeps more on-device state to reduce overhead.

This can be faster than `shared_safe_low_vram`, but it is still a shared swapping strategy and can still be significantly slower than standard CFG depending on your LoRA stack and Comfy internals.

If you see any output corruption, revert to `shared_safe_low_vram`.

---

## Node inputs

### Required
- **good_model (MODEL)**: The model used for normal CFG.
- **bad_model (MODEL)**: The model used as AutoGuidance reference.
- **positive (CONDITIONING)**: Your prompt conditioning.
- **negative (CONDITIONING)**: Your negative prompt conditioning.
- **cfg (FLOAT)**: Standard CFG scale.
- **w_autoguide (FLOAT)**: Paper-style AutoGuidance strength:
  - `1.0` = off (behaves like normal CFG)
  - `2.0` = moderate
  - `3.0+` = strong

- **swap_mode**: One of:
  - `shared_safe_low_vram`
  - `shared_fast_extra_vram`
  - `dual_models_2x_vram`

---

## Optional settings (what they do + how to use them)

### 1) `ag_delta_mode`
Controls *which* difference vector is used for the AutoGuidance direction.

- **`bad_conditional` (recommended default)**  
  Uses the difference between **good CFG output** and **bad conditional-only** prediction.  
  Usually the most sensitive to LoRA differences and the most “noticeable”.

- **`raw_delta`**  
  Uses the difference between the fully guided outputs of good vs bad.

- **`project_cfg`**  
  Projects the AutoGuidance direction onto the CFG direction.  
  This is more conservative and can look closer to standard CFG.

**If your results look too similar to normal CFG**, try:
- `bad_conditional`
- higher `ag_max_ratio`
- and a ramp mode that emphasizes earlier steps (see below)

---

### 2) `ag_max_ratio` (how strong AG is allowed to get)
AutoGuidance delta is capped relative to the magnitude of the CFG direction:

- Higher = more effect (but can destabilize)
- Lower = subtle

Practical ranges:
- `0.35` subtle
- `0.6–1.0` noticeable
- `1.0–2.0` very strong / can get unstable depending on model/LoRAs

If you set this too high without a ramp/floor strategy, you can get “wrong VAE / collapse-ish” artifacts.

---

### 3) Ramp settings (how the cap changes across denoise progress)

These settings scale the *effective* `ag_max_ratio` over the sampling trajectory.

- **`ag_ramp_mode`**
  - `flat`: constant cap across steps (best default; avoids “AG=0 early” surprises)
  - `detail_late`: stronger late steps (detail refinement emphasis)
  - `compose_early`: stronger early steps (composition/layout emphasis)
  - `mid_peak`: strongest in the middle (often a balanced compromise)

- **`ag_ramp_power`**
  Controls how sharply the ramp changes:
  - `1.0` = gentle
  - `2.0` = stronger curve
  - `4.0+` = very concentrated

- **`ag_ramp_floor`**
  Minimum always-on fraction of `ag_max_ratio` (0..1).  
  Use this to prevent “near-zero AG” regions.
  - Example: `0.10` means “at least 10% of the cap is always available”

#### Suggested presets
**A) Composition / pose changes (more NAG-like behavior)**
- `ag_ramp_mode = compose_early`
- `ag_ramp_floor = 0.10–0.25`
- `ag_max_ratio = 0.8–1.5`
- `w_autoguide = 3.0–5.0`

**B) Detail-only refinement (minimal composition drift)**
- `ag_ramp_mode = detail_late`
- `ag_ramp_floor = 0.05–0.15`
- `ag_max_ratio = 0.5–1.0`
- `w_autoguide = 2.0–3.5`

**C) Conservative / close to CFG**
- `ag_delta_mode = project_cfg`
- `ag_ramp_mode = flat`
- `ag_max_ratio = 0.3–0.6`
- `w_autoguide = 1.5–2.5`

---

### 4) `ag_allow_negative` (mainly for `project_cfg`)
If `False`, AutoGuidance components that point “opposite” to the projected CFG direction are clamped away.

- `True` (default): allows signed projection
- `False`: more conservative

---

### 5) Swap/ownership safety knobs

- **`safe_force_clean_swap`**
  Forces a “clean” unpatch/patch cycle during shared swapping.
  - Slower
  - Fixes “washed out”, “same output”, or other swap-state leakage on some Comfy builds

- **`uuid_only_noop`**
  Debug/edge-case setting. Treat “same patch UUID” as a no-op without extra validation.
  - Can hide broken swapping; leave `False` unless debugging.

---

### 6) Debug settings

- **`debug_swap`**
  Prints patch/ownership info and signatures to help confirm swaps are actually happening.

- **`debug_metrics`**
  Prints magnitude diagnostics (CFG magnitude, AG magnitude, cap ratio, etc.).  
  Very useful to detect cases where AG is effectively being clamped to ~0.

---

## Troubleshooting

### “Looks almost the same as normal CFG”
Most common reasons:
1) **AG is getting clamped** (effective ratio near 0, or `scale` near 0).  
   Fix: increase `ag_max_ratio`, use `flat` ramp or raise `ag_ramp_floor`.

2) **Good vs bad are not meaningfully different (in early steps)**  
   Fix: make bad model *actually different* (remove a key LoRA, use a deliberately “bad” LoRA, etc.), and/or use `compose_early`.

3) **Using `detail_late` with `ag_ramp_floor=0.0`**  
   That intentionally gives near-zero effect early (composition won’t change much).

### “dual_models_2x_vram is not faster”
Ensure you truly loaded two distinct model instances:
- Copy the checkpoint file and load the copy for the other branch.
If both branches reference the same underlying file/path, Comfy may reuse the instance.

### “washed out / corrupted / wrong VAE-looking collapse”
- Reduce `ag_max_ratio`
- Reduce `w_autoguide`
- Use `shared_safe_low_vram`
- Use `flat` ramp + a small `ag_ramp_floor`
- If it only happens with shared swapping, enable `safe_force_clean_swap`

---

## Recommended baseline settings (starting point)

If you just want “it works”:
- `ag_delta_mode = bad_conditional`
- `w_autoguide = 2.0–3.0`
- `ag_max_ratio = 0.6–1.0`
- `ag_ramp_mode = flat`
- `ag_ramp_floor = 0.10`
- `swap_mode = dual_models_2x_vram` (with checkpoint copy) if VRAM allows

---

## Use

- Build `good_model` and `bad_model` (e.g. SDXL + good LoRA, SDXL + early/bad LoRA)
- Create GUIDER with `AutoGuidance CFG Guider (good+bad)`
- Feed GUIDER into `SamplerCustomAdvanced`
