# ComfyUI-AutoGuidance

Adds an AutoGuidance+CFG GUIDER node for ComfyUI.

## Install

Clone into:
`ComfyUI/custom_nodes/ComfyUI-AutoGuidance`

Restart ComfyUI.

## Use

- Build `good_model` and `bad_model` (e.g. SDXL + good LoRA, SDXL + early/bad LoRA)
- Create GUIDER with `AutoGuidance CFG Guider (good+bad)`
- Feed GUIDER into `SamplerCustomAdvanced`
