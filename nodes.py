from __future__ import annotations

import math
from typing import Any, Dict, List

import comfy.sampler_helpers
import comfy.samplers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks


class Guider_AutoGuidanceCFG(comfy.samplers.CFGGuider):
    """
    AutoGuidance + CFG guider.

    - "good" model: used for CFG (positive + negative)
    - "bad" model: used as the autoguide reference (positive only)

    Output matches what ComfyUI guiders return from predict_noise(): "denoised" prediction
    (same convention used by other custom guiders that subclass CFGGuider).
    """

    def __init__(self, good_model, bad_model):
        super().__init__(good_model)
        self.bad_model_patcher = bad_model
        self.inner_bad_model = None
        self.bad_conds = None
        self.w_ag: float = 1.0  # paper-style weight; (w_ag - 1) is the delta scale

    def set_conds(self, positive, negative) -> None:
        # Keep the same conditioning dict style used by CFGGuider-based guiders.
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_scales(self, cfg: float, w_ag: float) -> None:
        self.cfg = float(cfg)
        self.w_ag = float(w_ag)

    def outer_sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        self.inner_model, self.conds, loaded_good = comfy.sampler_helpers.prepare_sampling(
            self.model_patcher, noise.shape, self.conds, self.model_options
        )
        self.inner_bad_model, self.bad_conds, loaded_bad = comfy.sampler_helpers.prepare_sampling(
            self.bad_model_patcher, noise.shape, self.bad_conds, self.model_options
        )

        seen = set()
        loaded_all = []
        for model in (loaded_good + loaded_bad):
            model_id = id(model)
            if model_id in seen:
                continue
            seen.add(model_id)
            loaded_all.append(model)
        self.loaded_models = loaded_all

        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = comfy.sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)

        comfy.samplers.cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            self.bad_model_patcher.pre_run()

            output = self.inner_sample(
                noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed
            )
        finally:
            comfy.samplers.cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.bad_model_patcher.cleanup()
            self.model_patcher.cleanup()

            cleanup_conds = {}
            keys = set(self.conds.keys()) | (set(self.bad_conds.keys()) if self.bad_conds else set())
            for k in keys:
                cleanup_conds[k] = (self.conds.get(k, []) or []) + (
                    (self.bad_conds.get(k, []) or []) if self.bad_conds else []
                )
            comfy.sampler_helpers.cleanup_models(cleanup_conds, self.loaded_models)

            del self.inner_model
            del self.inner_bad_model
            del self.loaded_models

        return output

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        """
        Override CFGGuider.sample to mirror upstream CFGGuider sample/outer_sample lifecycle,
        extended to prepare BOTH the good and bad models.
        """
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        self.bad_conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
            self.bad_conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        comfy.samplers.preprocess_conds_hooks(self.conds)
        comfy.samplers.preprocess_conds_hooks(self.bad_conds)

        try:
            orig_model_options = self.model_options
            if not isinstance(self.model_options, dict):
                self.model_options = {}
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            self.model_options.setdefault("transformer_options", {})

            orig_hook_mode_good = self.model_patcher.hook_mode
            orig_hook_mode_bad = self.bad_model_patcher.hook_mode
            if comfy.samplers.get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
                self.bad_model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram

            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            comfy.sampler_helpers.prepare_model_patcher(self.bad_model_patcher, self.bad_conds, self.model_options)

            comfy.samplers.filter_registered_hooks_on_conds(self.conds, self.model_options)
            comfy.samplers.filter_registered_hooks_on_conds(self.bad_conds, self.model_options)

            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True
                ),
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            comfy.samplers.cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)

            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode_good
            self.bad_model_patcher.hook_mode = orig_hook_mode_bad
            self.model_patcher.restore_hook_patches()
            self.bad_model_patcher.restore_hook_patches()

            del self.conds
            del self.bad_conds

        return output

    def predict_noise(self, x, timestep, model_options: Dict[str, Any] | None = None, seed=None, **kwargs):
        """
        Mirrors comfy.samplers.sampling_function + cfg_function, but injects
        AutoGuidance delta BEFORE sampler_post_cfg_function hooks.
        """
        if model_options is None:
            model_options = {}
        # Ensure wrapper-related options always exist
        if model_options.get("transformer_options", None) is None:
            model_options["transformer_options"] = {}

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        # SDXL needs the extra y/adm inputs; do not use None-uncond optimization.
        uncond_for_good = negative_cond

        conds_good: List[Any] = [positive_cond, uncond_for_good]
        self.inner_model.current_patcher = self.model_patcher
        out_good = comfy.samplers.calc_cond_batch(
            self.inner_model,
            conds_good,
            x,
            timestep,
            model_options,
        )

        for fn in model_options.get("sampler_pre_cfg_function", []):
            args = {
                "conds": conds_good,
                "conds_out": out_good,
                "cond_scale": self.cfg,
                "timestep": timestep,
                "input": x,
                "sigma": timestep,
                "model": self.inner_model,
                "model_options": model_options,
            }
            out_good = fn(args)

        cond_pred = out_good[0]
        uncond_pred = out_good[1]

        if "sampler_cfg_function" in model_options:
            args = {
                "cond": x - cond_pred,
                "uncond": x - uncond_pred,
                "cond_scale": self.cfg,
                "timestep": timestep,
                "input": x,
                "sigma": timestep,
                "cond_denoised": cond_pred,
                "uncond_denoised": uncond_pred,
                "model": self.inner_model,
                "model_options": model_options,
            }
            cfg_out = x - model_options["sampler_cfg_function"](args)
        else:
            cfg_out = uncond_pred + (cond_pred - uncond_pred) * self.cfg

        # For SDXL, the "bad_conds" version can lose pooled/adm info via hooks.
        # AutoGuidance expects SAME prompt conditioning; only the model differs.
        pos_bad_cond = positive_cond

        # IMPORTANT: SDXL-style models require the extra "y/adm" inputs.
        # Passing `None` as a second conditioning can drop those and trip
        # "must specify y if and only if the model is class-conditional".
        # So: run ONLY a single conditional forward for the bad model.
        conds_bad_single: List[Any] = [pos_bad_cond]
        self.inner_bad_model.current_patcher = self.bad_model_patcher
        out_bad_single = comfy.samplers.calc_cond_batch(
            self.inner_bad_model,
            conds_bad_single,
            x,
            timestep,
            model_options,
        )
        bad_cond_pred = out_bad_single[0]

        # If any pre-cfg hooks exist, provide a 2-slot "view" (duplicate) for compatibility
        # without doing a second forward pass.
        if model_options.get("sampler_pre_cfg_function", []):
            self.inner_bad_model.current_patcher = self.bad_model_patcher
            conds_bad_hooks: List[Any] = [pos_bad_cond, pos_bad_cond]
            out_bad_hooks = [bad_cond_pred, bad_cond_pred]
            for fn in model_options.get("sampler_pre_cfg_function", []):
                out_bad_hooks = fn(
                    {
                        "conds": conds_bad_hooks,
                        "conds_out": out_bad_hooks,
                        "cond_scale": self.cfg,
                        "timestep": timestep,
                        "input": x,
                        "sigma": timestep,
                        "model": self.inner_bad_model,
                        "model_options": model_options,
                    }
                )
            bad_cond_pred = out_bad_hooks[0]

        ag_delta = (self.w_ag - 1.0) * (cond_pred - bad_cond_pred)
        denoised = cfg_out + ag_delta

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": denoised,
                "cond": positive_cond,
                "uncond": uncond_for_good,
                "model": self.inner_model,
                "uncond_denoised": uncond_pred,
                "cond_denoised": cond_pred,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                "cond_scale": self.cfg,
                "bad_model": self.inner_bad_model,
                "bad_cond_denoised": bad_cond_pred,
                "autoguidance_w": self.w_ag,
            }
            denoised = fn(args)

        return denoised


class AutoGuidanceCFGGuider:
    """
    Node that constructs the GUIDER object.
    Use with SamplerCustomAdvanced (guider input).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "good_model": ("MODEL",),
                "bad_model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1, "round": 0.01}),
                # Paper-style parameterization:
                # w_ag = 1.0 => off, 2.0 => moderate, 3.0 => strong
                "w_autoguide": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.05, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/guiders"

    def get_guider(self, good_model, bad_model, positive, negative, cfg: float, w_autoguide: float):
        guider = Guider_AutoGuidanceCFG(good_model, bad_model)
        guider.set_conds(positive, negative)
        guider.set_scales(cfg=cfg, w_ag=w_autoguide)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "AutoGuidanceCFGGuider": AutoGuidanceCFGGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoGuidanceCFGGuider": "AutoGuidance CFG Guider (good+bad)",
}
