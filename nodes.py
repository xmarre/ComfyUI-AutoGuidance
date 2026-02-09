from __future__ import annotations

import math
from typing import Any, Dict, List

import comfy.sampler_helpers
import comfy.samplers


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
        Override CFGGuider.sample so BOTH good and bad models go through prepare_sampling()
        and get cleaned up via cleanup_models().

        This mirrors comfy.samplers.CFGGuider.sample's lifecycle.
        """
        if sigmas.shape[-1] == 0:
            return latent_image

        # ComfyUI's prepare_sampling now expects a non-None model_options dict
        # (wrapper plumbing uses model_options.get(...)).
        model_options = getattr(self, "model_options", None)
        if not isinstance(model_options, dict):
            model_options = {"transformer_options": {}}
            self.model_options = model_options
        else:
            # Some setups may have transformer_options missing/None; make it a dict.
            if model_options.get("transformer_options", None) is None:
                model_options["transformer_options"] = {}

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, loaded_good = comfy.sampler_helpers.prepare_sampling(
            self.model_patcher, noise.shape, self.conds, model_options=model_options
        )

        bad_conds = {}
        for k in self.original_conds:
            bad_conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_bad_model, self.bad_conds, loaded_bad = comfy.sampler_helpers.prepare_sampling(
            self.bad_model_patcher, noise.shape, bad_conds, model_options=model_options
        )

        self.loaded_models = loaded_good + loaded_bad

        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = comfy.sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)

        try:
            output = self.inner_sample(
                noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed
            )
        finally:
            cleanup_conds = {}
            keys = set(self.conds.keys()) | (set(self.bad_conds.keys()) if self.bad_conds else set())
            for k in keys:
                cleanup_conds[k] = (self.conds.get(k, []) or []) + (
                    (self.bad_conds.get(k, []) or []) if self.bad_conds else []
                )
            comfy.sampler_helpers.cleanup_models(cleanup_conds, self.loaded_models)

        del self.inner_model
        del self.inner_bad_model
        del self.conds
        del self.bad_conds
        del self.loaded_models

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

        # Keep a reference so sample() can reuse it if needed
        self.model_options = model_options

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        if math.isclose(self.cfg, 1.0) and model_options.get("disable_cfg1_optimization", False) is False:
            uncond_for_good = None
        else:
            uncond_for_good = negative_cond

        conds_good: List[Any] = [positive_cond, uncond_for_good]
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

        if uncond_for_good is None:
            uncond_pred = cond_pred

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
            cfg_out = cond_pred if uncond_for_good is None else (uncond_pred + (cond_pred - uncond_pred) * self.cfg)

        pos_bad_cond = (self.bad_conds.get("positive", None) if self.bad_conds else None) or positive_cond
        conds_bad: List[Any] = [pos_bad_cond, None]
        out_bad = comfy.samplers.calc_cond_batch(
            self.inner_bad_model,
            conds_bad,
            x,
            timestep,
            model_options,
        )
        for fn in model_options.get("sampler_pre_cfg_function", []):
            args = {
                "conds": conds_bad,
                "conds_out": out_bad,
                "cond_scale": self.cfg,
                "timestep": timestep,
                "input": x,
                "sigma": timestep,
                "model": self.inner_bad_model,
                "model_options": model_options,
            }
            out_bad = fn(args)

        bad_cond_pred = out_bad[0]

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
