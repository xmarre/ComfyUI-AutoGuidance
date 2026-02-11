# ComfyUI-AutoGuidance

A ComfyUI custom node implementing **AutoGuidance** from the paper:

> **Guiding a Diffusion Model with a Bad Version of Itself** (Karras et al., 2024)  
> https://arxiv.org/abs/2406.02507

Repository: https://github.com/xmarre/ComfyUI-AutoGuidance

## What this does

Classic CFG steers generation by contrasting conditional and unconditional predictions.  
**AutoGuidance** adds a second model path (“bad model”) and guides relative to that weaker reference.

In practice, this gives you another control axis for balancing:
- quality/faithfulness,
- collapse/overcooking risk,
- structure vs detail emphasis (via ramping).

## Included nodes

This extension registers two nodes:

- **AutoGuidance CFG Guider (good+bad)** (`AutoGuidanceCFGGuider`)  
  Produces a `GUIDER` for use with `SamplerCustomAdvanced`.
- **AutoGuidance Detailer Hook (Impact Pack)** (`AutoGuidanceImpactDetailerHookProvider`)  
  Produces a `DETAILER_HOOK` for Impact Pack detailer workflows (including FaceDetailer).

## Installation

Clone into your ComfyUI custom nodes directory and restart ComfyUI:

```bash
git clone https://github.com/xmarre/ComfyUI-AutoGuidance
```

No extra dependencies.

## Basic wiring (SamplerCustomAdvanced)

1. Load two models:
   - `good_model`
   - `bad_model`
2. Build conditioning normally:
   - `positive`
   - `negative`
3. Add **AutoGuidance CFG Guider (good+bad)**.
4. Connect its `GUIDER` output to **SamplerCustomAdvanced** `guider` input.

## Impact Pack / FaceDetailer integration

Use **AutoGuidance Detailer Hook (Impact Pack)** when your detailer nodes accept a `DETAILER_HOOK`.

This injects AutoGuidance into detailer sampling passes without editing Impact Pack source files.

## Important: dual-model mode must use truly distinct model instances

If you use:

- `swap_mode = dual_models_2x_vram`

then ensure ComfyUI does **not** dedupe the two model loads into one shared instance.

### Recommended setup

Make a real file copy of your checkpoint (same bytes, different filename), for example:

- `SDXL_base.safetensors`
- `SDXL_base_BADCOPY.safetensors`

Then:

- Loader A (file 1) → `good_model`
- Loader B (file 2) → `bad_model`

If both loaders point to the exact same path, ComfyUI may share/collapse model state and dual-mode behavior/performance can be incorrect.

## Parameters (AutoGuidance CFG Guider)

### Required

- `cfg`
- `w_autoguide` (effect is effectively off at `1.0`; stronger above `1.0`)
- `swap_mode`
  - `shared_safe_low_vram` (safest/slowest)
  - `shared_fast_extra_vram` (faster shared swap, extra VRAM)
  - `dual_models_2x_vram` (fastest, highest VRAM, requires distinct instances)

### Optional core controls

- `ag_delta_mode`
  - `bad_conditional` (default, common starting point)
  - `raw_delta`
  - `project_cfg`
  - `reject_cfg`
- `ag_max_ratio` (caps AutoGuidance push relative to CFG update magnitude)
- `ag_allow_negative`
- `ag_ramp_mode`
  - `flat`
  - `detail_late`
  - `compose_early`
  - `mid_peak`
- `ag_ramp_power`
- `ag_ramp_floor`
- `ag_post_cfg_mode`
  - `keep`
  - `apply_after`
  - `skip`

### Swap/debug controls

- `safe_force_clean_swap`
- `uuid_only_noop`
- `debug_swap`
- `debug_metrics`

## Example setup (one working recipe)

> This is one practical baseline, not a rule.

### Models

- **Good side**:
  - Base checkpoint + more fully-trained/specialized stack (e.g., 40-epoch character LoRA + DMD2/LCM, etc.)
- **Bad side** options:
  - Base checkpoint + earlier/weaker checkpoint/LoRA (e.g., 10-epoch) with intentionally poor weighting
  - Base checkpoint + fewer adaptation modules
  - Base checkpoint only

Core idea: bad side should be meaningfully weaker/less specialized than good side.

### Node settings example

- `cfg: 1.1`
- `w_autoguide: 3.00`
- `swap_mode: dual_models_2x_vram`
- `ag_delta_mode: reject_cfg`
- `ag_max_ratio: 0.75`
- `ag_allow_negative: true`
- `ag_ramp_mode: compose_early`
- `ag_ramp_power: 2.0`
- `ag_ramp_floor: 0.00`
- `ag_post_cfg_mode: skip`
- `safe_force_clean_swap: true`
- `uuid_only_noop: false`
- `debug_swap: true`
- `debug_metrics: true`

## Practical tuning notes

- Increase `w_autoguide` above `1.0` to strengthen effect.
- Use `ag_max_ratio` to prevent runaway/cooked outputs.
- `compose_early` tends to affect composition/structure earlier in denoise.
- Try `detail_late` for a more late-step/detail-leaning influence.

## VRAM and speed

AutoGuidance adds extra forward work versus plain CFG.

- `dual_models_2x_vram`: fastest but highest VRAM and strict dual-instance requirement.
- Shared modes: lower VRAM, slower due to swapping.

## Suggested A/B evaluation

At fixed seed/steps, compare:

- CFG-only vs CFG + AutoGuidance
- different `ag_ramp_mode`
- different `ag_max_ratio` caps
- different `ag_delta_mode`

## Feedback wanted

Useful community feedback includes:

- what “bad model” definitions work best in real SD pipelines,
- parameter combos that outperform standard CFG or NAG,
- reproducible A/B examples with fixed seed + settings.

---

If this node helps, share workflows and side-by-side comparisons in issues/discussions.
