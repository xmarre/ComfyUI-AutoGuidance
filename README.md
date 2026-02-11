# ComfyUI-AutoGuidance

Adds an **AutoGuidance + CFG** **GUIDER** node for **ComfyUI**, based on:

- **Guiding a Diffusion Model with a Bad Version of Itself** (arXiv:2406.02507)  
  https://arxiv.org/abs/2406.02507

This node lets you run normal CFG on a “good” model while also using a “bad” model to compute an additional **AutoGuidance delta** that is added on top of the CFG output (optionally capped and ramped over steps).

You provide:
- a **good_model** (what you actually want),
- a **bad_model** (intentionally different/worse),
- and this guider mixes them according to the selected **delta mode**, **cap**, and **ramp**.

---

## Install

Clone into:

`ComfyUI/custom_nodes/ComfyUI-AutoGuidance`

Restart ComfyUI.

---

## Node

- **AutoGuidance CFG Guider (good+bad)** → outputs a **GUIDER**  
  Plug it into **SamplerCustomAdvanced** (GUIDER input).

---

## Quick start (how to wire it)

1) Load your two models:
- `good_model`: checkpoint + your normal LoRA stack
- `bad_model`: intentionally different/worse setup (examples below)

2) Add:
- **AutoGuidance CFG Guider (good+bad)**

3) Connect:
- its **GUIDER** output → **SamplerCustomAdvanced** → **GUIDER**

That’s it.

---

## Extremely important: `dual_models_2x_vram` requires a second checkpoint file path

If you pick **`swap_mode = dual_models_2x_vram`**, you must ensure ComfyUI loads **two distinct model instances**.

If both loaders point to the same checkpoint path, ComfyUI’s caching can reuse the same underlying model object. In that situation, `dual_models_2x_vram` cannot behave as intended.

### Do this (reliable)

Make a second checkpoint file path and load each one:

- Copy the checkpoint file (simple + reliable), e.g.  
  `model.safetensors` → `model_copy.safetensors`

Then in ComfyUI:
- `CheckpointLoaderSimple` → load `model.safetensors` (good path)
- `CheckpointLoaderSimple` → load `model_copy.safetensors` (bad path)

**VRAM note:** dual-model mode uses ~2× the base checkpoint VRAM (plus LoRAs, plus activations).

---

## Choosing a useful “bad_model”

AutoGuidance only has leverage if **good_model and bad_model produce meaningfully different denoiser predictions**. If they’re too similar, the effect will be subtle regardless of ramp settings.

Ways to make `bad_model` meaningfully different (examples, not promises):
- Remove some LoRAs on the bad path (good has them, bad doesn’t)
- Use a different LoRA stack on the bad path (style/character/etc.)
- Use the same LoRA at a deliberately different weight (including negative if your workflow supports it)
- Change conditioning strategy on the bad path (e.g. different prompt/conditioning pipeline)

The point isn’t “worse” aesthetically—it’s **different** in a way that changes predictions.

---

## Node parameters

### Core knobs

#### `cfg`
Normal CFG scale.

#### `w_autoguide`
Paper-style parameterization:
- `w_autoguide = 1.0` → AutoGuidance delta effectively off
- `w_autoguide > 1.0` → strength increases with `(w_autoguide - 1)`

Internally this is used as a multiplier for the AutoGuidance delta direction (added on top of CFG).

---

## Swap modes (performance + correctness)

### `shared_safe_low_vram` (default)
- Lowest VRAM overhead
- Most compatibility / correctness across Comfy variants
- Can be **very slow** because it must swap LoRA stacks on a shared model safely

### `shared_fast_extra_vram`
- Faster swapping than safe mode (uses inplace updates / device-friendly behavior)
- Still shares one model object, so there’s still unavoidable overhead
- Uses extra VRAM

### `dual_models_2x_vram`
- Fastest **when you truly load two separate checkpoint instances** (see the “second checkpoint path” section above)
- Uses ~2× checkpoint VRAM
- Avoids the expensive shared swapping path entirely

---

## AutoGuidance delta mode (`ag_delta_mode`)

### `bad_conditional` (recommended default)
Uses the most “LoRA-sensitive” direction in practice:
- compare **good conditional output** vs **bad conditional output**

This tends to show the biggest differences when you change LoRAs on the bad path.

### `raw_delta`
Uses a raw difference direction between guided outputs. Can be harsher / less predictable.

### `project_cfg`
Projects the “push-away-from-bad” direction onto the **actual CFG update direction**.  
This keeps changes more aligned with CFG, often more conservative.

### `reject_cfg`
Removes the component of the “push-away-from-bad” direction that is parallel to the **actual CFG update direction**.  
This can increase composition changes when AG otherwise behaves like “CFG++”.

---

## Safety cap: `ag_max_ratio`

AutoGuidance is capped relative to the magnitude of the **actual CFG update** (`cfg_out - uncond_good`):

- Higher `ag_max_ratio` = stronger visible effect (up to destabilization if extreme)
- Lower `ag_max_ratio` = subtler

If your output looks “basically like normal CFG”, this cap (and/or the ramp) is the first place to inspect.

> This repo includes `debug_metrics` specifically so you can see whether you are being clamped (look at `scale` and `n_delta_applied`).

---

## Ramp over steps (this controls “composition vs detail”)

These parameters scale **the cap** (`ag_max_ratio`) over denoise progress.

Important concept:
- Progress is defined as **0 early (high sigma)** → **1 late (low sigma)**
- In this implementation, progress is tracked by step index: `prog = step / (total_steps - 1)`

### `ag_ramp_mode`
- `flat`  
  Constant cap across all steps.  
  **Use this if you want composition changes.**
- `detail_late`  
  Weak early, strong late → mostly affects fine detail.  
  If `ag_ramp_floor = 0`, early steps can get near-zero AG and the image composition can stay close to normal CFG.
- `compose_early`  
  Strong early, weaker late → pushes structure/composition more than detail.
- `mid_peak`  
  Strongest mid-steps, weaker at ends.

### `ag_ramp_power`
Controls steepness of the selected ramp curve.

### `ag_ramp_floor`
Minimum always-on fraction of the cap.
- `0.0` means the ramp can reduce AG close to zero in parts of the schedule.
- If you use `detail_late` but still want some early influence, raise `ag_ramp_floor`.

> This README intentionally does not give “magic” preset numbers. Use `debug_metrics` to confirm what’s actually being applied in your workflow.

---

## Advanced / safety toggles

### `safe_force_clean_swap`
Only relevant in shared-model modes. Forces clean swaps between patch stacks to avoid state leakage / washed-out results on some Comfy builds.
- Safer
- Slower

### `uuid_only_noop`
Debug option. Treat “same patches_uuid” as a no-op even if ownership tracking is imperfect.  
Use only for debugging.

### `debug_swap`
Prints swap/patch diagnostics (patch counts, uuids, etc.).

### `debug_metrics`
Prints direction / magnitude diagnostics (useful for confirming AG is actually being applied).  
It also logs active `sampler_post_cfg_function` hooks at step 0, which helps diagnose post-CFG rescale/clamp behavior that can suppress visible AG effects.

---

## Troubleshooting

### “It looks almost the same as normal CFG”
Common causes:

1) **Your ramp/cap is effectively zero early**, so composition won’t move.
   - If you used `detail_late` with `ag_ramp_floor = 0`, early AG can be near-zero by design.
   - Fix: use `flat` or `compose_early`, or raise `ag_ramp_floor`.

2) **Bad model is not actually “bad enough”**
   - Make the bad path deliberately different for one test (remove key LoRAs, use an intentionally different stack, etc.).
   - Confirm with `debug_metrics` that `n_good_minus_bad_cond` is meaningfully non-zero.

3) **You’re not truly in dual-model mode**
   - If you selected `dual_models_2x_vram` but did not load the checkpoint via two different file paths, you may not actually have two model instances.

4) **Post-CFG hooks are suppressing the visible effect**
   - `debug_metrics` prints active `sampler_post_cfg_function` hooks at step 0.
   - Some workflows rescale/normalize/clamp after CFG, which can reduce the visible impact of AG.

### “dual_models_2x_vram is insanely slow”
That can happen if both loaders resolve to the same underlying model instance (same checkpoint path / caching). Fix it by creating a second checkpoint file path and loading both.

### “My LoRA changes on the bad path do nothing”
Check `debug_swap` / `patch_info`:
- If patch counts change when you edit bad LoRAs, the patches are applied.
- If the image still doesn’t move, inspect `debug_metrics`:
  - are you being heavily clamped (`scale` near 0)?
  - is `n_good_minus_bad_cond` meaningfully > 0?
  - do you have post-CFG hooks that might normalize the change?

---

## Credits

- Paper: https://arxiv.org/abs/2406.02507  
- This repository implements an AutoGuidance-style guider node for ComfyUI’s custom sampler pipeline.
