from __future__ import annotations

import math
import inspect
import os
from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, List

import torch
import comfy.sampler_helpers
import comfy.samplers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks


def _clone_cond_metadata(cond):
    """
    Clone only the conditioning metadata containers, not the tensors.
    Comfy cond items are typically [tensor, {metadata...}, ...].
    """
    if isinstance(cond, Mapping):
        return {
            k: _clone_cond_metadata(v) if isinstance(v, (Mapping, list, tuple)) else v for k, v in cond.items()
        }
    if isinstance(cond, list):
        return [_clone_cond_metadata(v) if isinstance(v, (Mapping, list, tuple)) else v for v in cond]
    if isinstance(cond, tuple):
        return tuple(_clone_cond_metadata(v) if isinstance(v, (Mapping, list, tuple)) else v for v in cond)
    return cond


def _clone_container(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clone_container(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_container(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_container(v) for v in x)
    return x


def _clone_transformer_options(to: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clone transformer_options containers so the bad-model pass can't mutate the
    good-model pass state across timesteps.
    """
    return _clone_container(to or {})


def _clone_model_options_for_bad(model_options: Dict[str, Any]) -> Dict[str, Any]:
    bad_opts = dict(model_options)
    to = model_options.get("transformer_options") or {}
    bad_opts["transformer_options"] = _clone_transformer_options(to)
    return bad_opts


AG_SWAP_MODE_SHARED_SAFE = "shared_safe_low_vram"
AG_SWAP_MODE_SHARED_FAST = "shared_fast_extra_vram"
AG_SWAP_MODE_DUAL_MODELS = "dual_models_2x_vram"

AG_SWAP_MODE_CHOICES = [
    AG_SWAP_MODE_SHARED_SAFE,
    AG_SWAP_MODE_SHARED_FAST,
    AG_SWAP_MODE_DUAL_MODELS,
]

AG_DELTA_MODE_RAW = "raw_delta"
AG_DELTA_MODE_PROJECT_CFG = "project_cfg"
AG_DELTA_MODE_BAD_CONDITIONAL = "bad_conditional"

AG_DELTA_MODE_CHOICES = [
    AG_DELTA_MODE_BAD_CONDITIONAL,
    AG_DELTA_MODE_RAW,
    AG_DELTA_MODE_PROJECT_CFG,
]

# Default cap for AG magnitude relative to CFG-direction magnitude.
# Old code used 0.10 which is often visually near-invisible.
AG_DEFAULT_MAX_RATIO = 0.35


def _patcher_has_active_patches(patcher) -> bool:
    backup = getattr(patcher, "backup", None)
    if isinstance(backup, dict) and len(backup) > 0:
        return True

    obj_backup = getattr(patcher, "object_patches_backup", None)
    if isinstance(obj_backup, dict) and len(obj_backup) > 0:
        return True

    return False


def _infer_single_active_owner(patchers):
    active_patchers = [candidate for candidate in patchers if _patcher_has_active_patches(candidate)]
    if len(active_patchers) == 1:
        return active_patchers[0], False
    if len(active_patchers) > 1:
        return None, True
    return None, False


def _debug_print_weight_signature_once(model, patcher, *, tag: str, enabled: bool | None = None) -> None:
    if enabled is None:
        enabled = bool(getattr(model, "_ag_debug_swap", False)) or os.environ.get("AG_DEBUG_SWAP", "0") == "1"
    if not enabled:
        return

    key = f"_ag_dbg_sig_once_{tag}_{id(patcher)}"
    if hasattr(model, key):
        return

    try:
        for _, param in model.named_parameters():
            sig = float(param.detach().flatten()[:2048].float().sum().cpu())
            print("[AutoGuidance] active patcher", tag, "sig", sig)
            break
        setattr(model, key, True)
    except Exception as e:
        print(f"[AutoGuidance] signature debug failed for {tag}: {e!r}")


def _prepare_fast_weight_swap(model, patcher, *, peer_patchers, device) -> None:
    """
    Configure patchers for faster swapping.

    IMPORTANT:
    Do NOT share `backup` dicts across patchers. Many Comfy builds clear/replace
    `backup` inside unpatch_model(), and sharing it causes weight corruption.
    We still get speed by keeping backups on the target device and enabling
    inplace updates.
    """
    for p in (patcher, *peer_patchers):
        if device is not None:
            try:
                p.offload_device = device
            except Exception:
                pass

        try:
            p.weight_inplace_update = True
        except Exception:
            pass


def _call_unpatch_model_compat(patcher, *, device, prefer_device: bool = False) -> None:
    """
    Unpatch compat across Comfy variants.

    When prefer_device=True, try device-targeted unpatch first to better preserve
    on-device backup behavior on builds that support it.
    """
    if prefer_device and device is not None:
        try:
            patcher.unpatch_model(device_to=device, unpatch_weights=True)
            return
        except TypeError:
            pass
        except Exception:
            pass

    try:
        patcher.unpatch_model(unpatch_weights=True)
        return
    except TypeError:
        pass

    try:
        if device is not None:
            patcher.unpatch_model(device_to=device, unpatch_weights=True)
        else:
            patcher.unpatch_model(unpatch_weights=True)
        return
    except TypeError:
        pass

    try:
        if device is not None:
            patcher.unpatch_model(device, True)
        else:
            patcher.unpatch_model()
        return
    except Exception:
        patcher.unpatch_model()


def _activate_patcher_for_forward(
    patcher: comfy.model_patcher.ModelPatcher,
    *,
    peer_patchers: tuple[comfy.model_patcher.ModelPatcher, ...] = (),
    fast_weight_swap: bool = False,
    safe_force_clean_swap: bool | None = None,
    uuid_only_noop: bool | None = None,
    debug_swap: bool | None = None,
) -> None:
    m = getattr(patcher, "model", None)
    if m is None:
        return

    if not hasattr(m, "current_weight_patches_uuid"):
        m.current_weight_patches_uuid = None

    # Opt-in performance mode for shared-model LoRA swapping (costs extra VRAM).
    # IMPORTANT: only enabled via node/UI.
    fast_mode = bool(fast_weight_swap)

    # In shared_safe mode (fast_mode=False), default to a forced clean swap.
    # (Prevents state leak across wrapper stacks on some Comfy builds.)
    if safe_force_clean_swap is None:
        safe_force_clean_swap = (os.environ.get("AG_SAFE_FORCE_CLEAN_SWAP", "1") == "1")
    safe_force_clean = (not fast_mode) and bool(safe_force_clean_swap)

    if uuid_only_noop is None:
        uuid_only_noop = (os.environ.get("AG_UUID_ONLY_NOOP", "0") == "1")
    want_uuid = getattr(patcher, "patches_uuid", None)
    cur_uuid = getattr(m, "current_weight_patches_uuid", None)
    cur_owner = getattr(m, "current_patcher", None)
    all_patchers = (patcher, *peer_patchers)

    device = getattr(patcher, "load_device", None) or getattr(m, "device", None)

    if safe_force_clean:
        # Avoid doing the expensive work repeatedly if we already activated this patcher last time.
        last_id = getattr(m, "_ag_last_patcher_id", None)
        if last_id == id(patcher):
            m.current_patcher = patcher
            m.current_weight_patches_uuid = getattr(patcher, "patches_uuid", None)
            _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)
            return

        # Force-remove any lingering peer patches, then apply the requested patcher.
        for p in (*peer_patchers, patcher):
            try:
                _call_unpatch_model_compat(p, device=device)
            except Exception:
                pass
        m.current_weight_patches_uuid = None
        m.current_patcher = None

        _call_patch_model_safe(patcher, device=device)
        m.current_patcher = patcher
        m.current_weight_patches_uuid = getattr(patcher, "patches_uuid", None)
        m._ag_last_patcher_id = id(patcher)
        _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)
        return

    # UUIDs are not always reliable on all Comfy builds. Only use UUID ownership
    # resolution when both sides are concrete. Otherwise infer from active backups.
    can_use_uuid_resolution = (want_uuid is not None and cur_uuid is not None)
    if can_use_uuid_resolution and (cur_owner is None or getattr(cur_owner, "patches_uuid", None) != cur_uuid):
        resolved = None
        for candidate in all_patchers:
            if getattr(candidate, "patches_uuid", None) == cur_uuid:
                resolved = candidate
                break
        if resolved is not None:
            cur_owner = resolved

    if cur_owner is None:
        inferred_owner, _ = _infer_single_active_owner(all_patchers)
        if inferred_owner is not None:
            cur_owner = inferred_owner

    real_owner = getattr(m, "current_patcher", None)
    can_noop_by_uuid = (want_uuid is not None and cur_uuid is not None)
    if can_noop_by_uuid and cur_uuid == want_uuid and (uuid_only_noop or real_owner is patcher):
        m.current_patcher = patcher
        _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)
        return

    # (device already computed above)

    def _unpatch_owner_fast(owner) -> None:
        if owner is None:
            return
        _call_unpatch_model_compat(owner, device=device, prefer_device=True)
        m.current_weight_patches_uuid = None
        if getattr(m, "current_patcher", None) is owner:
            m.current_patcher = None

    def _unpatch_active_peers() -> None:
        cleaned = False
        for candidate in peer_patchers:
            if _patcher_has_active_patches(candidate):
                _call_unpatch_model_compat(candidate, device=device, prefer_device=fast_mode)
                cleaned = True
        if cleaned:
            m.current_weight_patches_uuid = None
            if getattr(m, "current_patcher", None) in peer_patchers:
                m.current_patcher = None

    # Fast mode keeps backups on target device and uses inplace updates, but
    # still performs correctness-first unpatch/patch calls across Comfy variants.
    if fast_mode:
        _prepare_fast_weight_swap(m, patcher, peer_patchers=peer_patchers, device=device)
        fast_force_clean = bool(safe_force_clean_swap)

        if fast_force_clean:
            last_id = getattr(m, "_ag_last_patcher_id_fast", None)
            if last_id == id(patcher):
                m.current_patcher = patcher
                m.current_weight_patches_uuid = getattr(patcher, "patches_uuid", None)
                _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)
                return

            for p in (*peer_patchers, patcher):
                try:
                    _call_unpatch_model_compat(p, device=device, prefer_device=True)
                except Exception:
                    pass
            m.current_weight_patches_uuid = None
            m.current_patcher = None

            _call_patch_model_safe(patcher, device=device)
            m.current_patcher = patcher
            m.current_weight_patches_uuid = getattr(patcher, "patches_uuid", None)
            m._ag_last_patcher_id_fast = id(patcher)
            _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)
            return

        # Less strict: unpatch inferred owner only, still using safe patch_model.
        owner = getattr(m, "current_patcher", None) or cur_owner
        if owner is None:
            inferred_owner, ambiguous = _infer_single_active_owner(all_patchers)
            if inferred_owner is not None:
                owner = inferred_owner
            elif ambiguous:
                _unpatch_active_peers()
        if owner is not None and owner is not patcher:
            _unpatch_owner_fast(owner)
        elif owner is patcher and cur_uuid != want_uuid:
            _unpatch_owner_fast(patcher)

        _call_patch_model_safe(patcher, device=device)
        m.current_patcher = patcher
        m.current_weight_patches_uuid = getattr(patcher, "patches_uuid", None)
        _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)
        return

    def _unpatch_owner(owner) -> None:
        if owner is None:
            return
        try:
            if device is not None:
                owner.unpatch_model(device_to=device, unpatch_weights=True)
            else:
                owner.unpatch_model(unpatch_weights=True)
        except TypeError:
            try:
                if device is not None:
                    owner.unpatch_model(device, True)
                else:
                    owner.unpatch_model()
            except Exception:
                owner.unpatch_model()

        m.current_weight_patches_uuid = None
        if getattr(m, "current_patcher", None) is owner:
            m.current_patcher = None

    if cur_owner is None:
        inferred_owner, ambiguous = _infer_single_active_owner(all_patchers)
        if inferred_owner is not None:
            cur_owner = inferred_owner
        elif ambiguous:
            _unpatch_active_peers()

    if cur_owner is not None and cur_owner is not patcher:
        _unpatch_owner(cur_owner)
    elif cur_owner is patcher and cur_uuid != want_uuid:
        _unpatch_owner(patcher)

    cur_uuid = getattr(m, "current_weight_patches_uuid", None)
    cur_owner = getattr(m, "current_patcher", None)

    real_owner = getattr(m, "current_patcher", None)
    can_noop_by_uuid = (want_uuid is not None and cur_uuid is not None)
    if can_noop_by_uuid and cur_uuid == want_uuid and (uuid_only_noop or real_owner is patcher):
        m.current_patcher = patcher
        _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)
        return

    _call_patch_model_safe(patcher, device=device)

    m.current_patcher = patcher
    m.current_weight_patches_uuid = getattr(patcher, "patches_uuid", None)
    _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)


def _sig_accepts_kw(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return True

    if name in sig.parameters:
        return True

    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _call_patch_model_safe(patcher, *, device):
    """
    Correctness-first patch path (low VRAM, slower).
    """
    fn = patcher.patch_model

    def _is_sig_typeerror(e: TypeError) -> bool:
        msg = str(e)
        return (
            "unexpected keyword argument" in msg
            or "got an unexpected keyword argument" in msg
            or "positional argument" in msg
            or "required positional argument" in msg
            or "missing a required keyword-only argument" in msg
            or "missing required keyword-only argument" in msg
            or "required keyword-only argument" in msg
            or "got multiple values for argument" in msg
            or "multiple values for keyword argument" in msg
            or ("takes" in msg and "positional" in msg)
        )

    def _try(load_weights: bool, force: bool, *, minimal: bool) -> bool:
        kw = {}

        if not minimal:
            if device is not None and _sig_accepts_kw(fn, "device_to"):
                kw["device_to"] = device
            if _sig_accepts_kw(fn, "lowvram_model_memory"):
                kw["lowvram_model_memory"] = 0

        if _sig_accepts_kw(fn, "load_weights"):
            kw["load_weights"] = load_weights
        if force and _sig_accepts_kw(fn, "force_patch_weights"):
            kw["force_patch_weights"] = True
        try:
            fn(**kw)
            return True
        except TypeError as e:
            if _is_sig_typeerror(e):
                return False
            raise

    if _try(load_weights=True, force=True, minimal=False):
        return
    if _try(load_weights=True, force=False, minimal=False):
        return

    fn()


def _find_first_y(cond: Any):
    if isinstance(cond, Mapping):
        y = cond.get("y", None)
        if y is not None:
            return y
        for v in cond.values():
            if isinstance(v, (Mapping, list, tuple)):
                y2 = _find_first_y(v)
                if y2 is not None:
                    return y2
        return None
    if isinstance(cond, (list, tuple)):
        for v in cond:
            if isinstance(v, (Mapping, list, tuple)):
                y2 = _find_first_y(v)
                if y2 is not None:
                    return y2
    return None


def _ensure_y_everywhere(cond: Any, y_ref) -> None:
    if y_ref is None:
        return
    if isinstance(cond, Mapping):
        try:
            if cond.get("y", None) is None:
                cond["y"] = y_ref
        except Exception:
            pass
        for v in cond.values():
            if isinstance(v, (Mapping, list, tuple)):
                _ensure_y_everywhere(v, y_ref)
        return
    if isinstance(cond, (list, tuple)):
        for v in cond:
            if isinstance(v, (Mapping, list, tuple)):
                _ensure_y_everywhere(v, y_ref)


def _restore_y_per_entry(dst_cond: Any, src_pos: Any, src_neg: Any = None) -> Any:
    """
    Restore SDXL class label 'y' WITHOUT collapsing multi-entry conditionings.
    For list conditionings, copy per-index y from src_pos/src_neg into dst_cond.
    """
    is_tuple = isinstance(dst_cond, tuple)
    if isinstance(dst_cond, (list, tuple)):
        dst_list = list(dst_cond)
        src_pos_list = list(src_pos) if isinstance(src_pos, (list, tuple)) else None
        src_neg_list = list(src_neg) if isinstance(src_neg, (list, tuple)) else None

        y_default = _find_first_y(src_pos) or _find_first_y(src_neg)

        for i in range(len(dst_list)):
            y_ref = None
            if src_pos_list is not None and i < len(src_pos_list):
                y_ref = _find_first_y(src_pos_list[i])
            if y_ref is None and src_neg_list is not None and i < len(src_neg_list):
                y_ref = _find_first_y(src_neg_list[i])
            if y_ref is None:
                y_ref = y_default

            if y_ref is not None:
                _ensure_y_everywhere(dst_list[i], y_ref)

        return tuple(dst_list) if is_tuple else dst_list

    y_ref = _find_first_y(src_pos) or _find_first_y(src_neg)
    if y_ref is not None:
        _ensure_y_everywhere(dst_cond, y_ref)
    return dst_cond


def _looks_like_tensor(v: Any) -> bool:
    return torch.is_tensor(v)


def _looks_like_tensorish(v: Any) -> bool:
    if _looks_like_tensor(v):
        return True
    if isinstance(v, (list, tuple)) and v and all(_looks_like_tensor(x) for x in v):
        return True
    return False


def _tensorish_width(v: Any) -> int | None:
    if _looks_like_tensor(v):
        return int(v.shape[-1])
    if isinstance(v, (list, tuple)) and v and all(_looks_like_tensor(x) for x in v):
        return sum(int(x.shape[-1]) for x in v)
    return None


def _coerce_tensorish_to_tensor(v: Any):
    if _looks_like_tensor(v):
        return v
    if isinstance(v, (list, tuple)) and v and all(_looks_like_tensor(x) for x in v):
        try:
            return torch.cat(list(v), dim=-1)
        except Exception:
            return v
    return v


def _ensure_primary_text_cond(dst_cond: Any, src_cond: Any) -> Any:
    """
    Ensure the primary conditioning tensor (the first element of each cond item)
    exists and matches src's embedding width. This prevents SDXL cross-attn from
    seeing context=None and falling back to self-attn (640-dim).
    """
    is_tuple = isinstance(dst_cond, tuple)
    if isinstance(dst_cond, (list, tuple)) and isinstance(src_cond, (list, tuple)):
        dst_list = list(dst_cond)
        src_list = list(src_cond)
        n = min(len(dst_list), len(src_list))
        for i in range(n):
            d = dst_list[i]
            s = src_list[i]

            d0 = d[0] if isinstance(d, (list, tuple)) and len(d) >= 1 else None
            s0 = s[0] if isinstance(s, (list, tuple)) and len(s) >= 1 else None

            if not _looks_like_tensorish(s0):
                continue

            s0_t = _coerce_tensorish_to_tensor(s0)
            if not _looks_like_tensor(s0_t):
                continue
            d0_t = _coerce_tensorish_to_tensor(d0)

            dw = _tensorish_width(d0_t)
            sw = _tensorish_width(s0_t)

            if sw is None:
                continue

            if isinstance(d, tuple):
                needs_replace = (dw is None) or (dw != sw) or (not _looks_like_tensor(d0_t))
                if needs_replace:
                    dst_list[i] = (s0_t, *d[1:])
                elif d0 is not d0_t:
                    dst_list[i] = (d0_t, *d[1:])
            elif isinstance(d, list):
                if len(d) >= 1:
                    needs_replace = (dw is None) or (dw != sw) or (not _looks_like_tensor(d0_t))
                    if needs_replace:
                        d[0] = s0_t
                    elif d0 is not d0_t:
                        d[0] = d0_t
                else:
                    dst_list[i] = [s0_t]
        return tuple(dst_list) if is_tuple else dst_list

    return dst_cond


class Guider_AutoGuidanceCFG(comfy.samplers.CFGGuider):
    """
    AutoGuidance + CFG guider.

    - "good" model: used for CFG (positive + negative)
    - "bad" model: used as the autoguide reference (positive only)

    Output matches what ComfyUI guiders return from predict_noise(): "denoised" prediction
    (same convention used by other custom guiders that subclass CFGGuider).
    """

    def __init__(
        self,
        good_model,
        bad_model,
        *,
        swap_mode: str = AG_SWAP_MODE_SHARED_SAFE,
        ag_delta_mode: str = AG_DELTA_MODE_BAD_CONDITIONAL,
        ag_max_ratio: float = AG_DEFAULT_MAX_RATIO,
        ag_allow_negative: bool = True,
        safe_force_clean_swap: bool = True,
        uuid_only_noop: bool = False,
        debug_swap: bool = False,
        debug_metrics: bool = False,
    ):
        super().__init__(good_model)
        self.bad_model_patcher = bad_model
        self.inner_bad_model = None
        self.bad_conds = None
        self.w_ag: float = 1.0  # paper-style weight; (w_ag - 1) is the delta scale
        self.swap_mode: str = str(swap_mode)
        self.ag_delta_mode: str = str(ag_delta_mode)
        self.ag_max_ratio: float = float(ag_max_ratio)
        self.ag_allow_negative: bool = bool(ag_allow_negative)
        self.safe_force_clean_swap: bool = bool(safe_force_clean_swap)
        self.uuid_only_noop: bool = bool(uuid_only_noop)
        self.debug_swap: bool = bool(debug_swap)
        self.debug_metrics: bool = bool(debug_metrics)

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
        # SDXL class-conditional models require 'y'. Some hook/filter stacks drop it for the bad model.
        # Restore 'y' per conditioning entry (do NOT merge other tensors like embeddings).
        if self.bad_conds and self.conds:
            bad_positive = self.bad_conds.get("positive", None)
            good_positive = self.conds.get("positive", None)
            good_negative = self.conds.get("negative", None)

            if bad_positive is not None and (good_positive is not None or good_negative is not None):
                bad_positive = _clone_cond_metadata(bad_positive)
                bad_positive = _restore_y_per_entry(bad_positive, good_positive, good_negative)
                self.bad_conds["positive"] = bad_positive

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

        # IMPORTANT: do NOT let the bad-model pass mutate shared transformer_options
        # that the good-model pass relies on across timesteps.
        bad_model_options = _clone_model_options_for_bad(model_options)
        shared_model = (getattr(self.model_patcher, "model", None) is getattr(self.bad_model_patcher, "model", None))

        def _patch_info(p):
            patches = getattr(p, "patches", None)
            if not isinstance(patches, dict):
                return {"uuid": getattr(p, "patches_uuid", None), "count": None, "top": []}
            c = Counter(k.split(".", 1)[0] for k in patches.keys())
            keys = list(patches.keys())
            return {
                "uuid": getattr(p, "patches_uuid", None),
                "count": len(patches),
                "top": c.most_common(10),
                "sample_keys": keys[:8],
            }

        dbg = bool(getattr(self, "debug_swap", False)) or (os.environ.get("AG_DEBUG_SWAP", "0") == "1")
        patch_dbg_key = f"_ag_dbg_patches_once_{getattr(self.bad_model_patcher, 'patches_uuid', None)}"
        if dbg and not hasattr(self, patch_dbg_key):
            print(
                "[AutoGuidance] patch_info",
                {
                    "good": _patch_info(self.model_patcher),
                    "bad": _patch_info(self.bad_model_patcher),
                    "shared": shared_model,
                },
            )
            setattr(self, patch_dbg_key, True)

        # 3-mode policy:
        # - shared_safe_low_vram: always use slow correctness-first patching
        # - shared_fast_extra_vram: use fast weightsafe swap (extra VRAM)
        # - dual_models_2x_vram: requires truly distinct model objects; otherwise fall back safely
        swap_mode = getattr(self, "swap_mode", AG_SWAP_MODE_SHARED_SAFE)
        if swap_mode == AG_SWAP_MODE_DUAL_MODELS and shared_model:
            if not hasattr(self, "_ag_warn_dual_shared_once"):
                print(
                    "[AutoGuidance] swap_mode=dual_models_2x_vram requested but good/bad share the same base model object; "
                    "falling back to shared_safe_low_vram to avoid washed-out output. "
                    "To actually use dual models, load the checkpoint twice as two distinct instances (e.g., via a copied/symlinked file path)."
                )
                self._ag_warn_dual_shared_once = True
            swap_mode = AG_SWAP_MODE_SHARED_SAFE

        fast_swap = (swap_mode == AG_SWAP_MODE_SHARED_FAST)


        def _activate(p, peers: tuple[comfy.model_patcher.ModelPatcher, ...]) -> None:
            if not shared_model:
                return

            # Allow environment variables to force-enable/disable debug/safety even when the node
            # is used with default UI values.
            safe_force_clean = bool(getattr(self, "safe_force_clean_swap", True))
            env_safe = os.environ.get("AG_SAFE_FORCE_CLEAN_SWAP", None)
            if env_safe in ("0", "1"):
                safe_force_clean = (env_safe == "1")

            uuid_noop = bool(getattr(self, "uuid_only_noop", False)) or (os.environ.get("AG_UUID_ONLY_NOOP", "0") == "1")
            dbg_swap = bool(getattr(self, "debug_swap", False)) or (os.environ.get("AG_DEBUG_SWAP", "0") == "1")

            _activate_patcher_for_forward(
                p,
                peer_patchers=peers,
                fast_weight_swap=fast_swap,
                safe_force_clean_swap=safe_force_clean,
                uuid_only_noop=uuid_noop,
                debug_swap=dbg_swap,
            )
        if shared_model:
            _activate(self.model_patcher, (self.bad_model_patcher,))

        if not hasattr(self, "_ag_dbg_once"):
            try:
                if shared_model:
                    print(
                        f"[AutoGuidance] shared_model={shared_model} "
                        f"good_uuid={getattr(self.model_patcher, 'patches_uuid', None)} "
                        f"bad_uuid={getattr(self.bad_model_patcher, 'patches_uuid', None)} "
                        f"current_uuid={getattr(getattr(self.model_patcher, 'model', None), 'current_weight_patches_uuid', None)}"
                    )
            except Exception as e:
                print(f"[AutoGuidance] dbg failed: {e!r}")
            self._ag_dbg_once = True

        try:
            positive_cond = self.conds.get("positive", None)
            negative_cond = self.conds.get("negative", None)

            # SDXL needs the extra y/adm inputs; do not use None-uncond optimization.
            uncond_for_good = negative_cond

            conds_good: List[Any] = [positive_cond, uncond_for_good]

            def _run_good(conds_list: List[Any]) -> List[Any]:
                if shared_model:
                    _activate(self.model_patcher, (self.bad_model_patcher,))
                    if not hasattr(self, "_ag_dbg_flip_good_once"):
                        m = getattr(self.model_patcher, "model", None)
                        print(
                            "[AutoGuidance] good_active?",
                            getattr(m, "current_weight_patches_uuid", None),
                            "want",
                            getattr(self.model_patcher, "patches_uuid", None),
                        )
                        self._ag_dbg_flip_good_once = True
                if shared_model and not hasattr(self, "_ag_uuid_check_good_once"):
                    m = getattr(self.model_patcher, "model", None)
                    print("[AutoGuidance] after_good_activate current_uuid=", getattr(m, "current_weight_patches_uuid", None))
                    self._ag_uuid_check_good_once = True
                # Keep hooks/wrappers that consult current_patcher aligned.
                self.inner_model.current_patcher = self.model_patcher
                return comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    conds_list,
                    x,
                    timestep,
                    model_options,
                )

            out_good = _run_good(conds_good)

            for fn in model_options.get("sampler_pre_cfg_function", []):
                if shared_model:
                    _activate(self.model_patcher, (self.bad_model_patcher,))
                self.inner_model.current_patcher = self.model_patcher
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

            # IMPORTANT: protect good outputs from being overwritten by the later bad-model pass.
            if torch.is_tensor(cond_pred):
                cond_pred_good = cond_pred.clone()
            else:
                cond_pred_good = cond_pred
            if torch.is_tensor(uncond_pred):
                uncond_pred_good = uncond_pred.clone()
            else:
                uncond_pred_good = uncond_pred

            if "sampler_cfg_function" in model_options:
                if shared_model:
                    _activate(self.model_patcher, (self.bad_model_patcher,))
                self.inner_model.current_patcher = self.model_patcher
                args = {
                    "cond": x - cond_pred,
                    "uncond": x - uncond_pred,
                    "cond_scale": self.cfg,
                    "timestep": timestep,
                    "input": x,
                    "sigma": timestep,
                    "cond_denoised": cond_pred_good,
                    "uncond_denoised": uncond_pred_good,
                    "model": self.inner_model,
                    "model_options": model_options,
                }
                cfg_out = x - model_options["sampler_cfg_function"](args)
            else:
                cfg_out = uncond_pred + (cond_pred - uncond_pred) * self.cfg

            if self.w_ag <= 1.0 + 1e-6:
                denoised = cfg_out
                for fn in model_options.get("sampler_post_cfg_function", []):
                    if shared_model:
                        _activate(self.model_patcher, (self.bad_model_patcher,))
                    self.inner_model.current_patcher = self.model_patcher
                    denoised = fn(
                        {
                            "denoised": denoised,
                            "cond": positive_cond,
                            "uncond": uncond_for_good,
                            "model": self.inner_model,
                            "uncond_denoised": uncond_pred_good,
                            "cond_denoised": cond_pred_good,
                            "sigma": timestep,
                            "model_options": model_options,
                            "input": x,
                            "cond_scale": self.cfg,
                            "bad_model": self.inner_bad_model,
                            "bad_cond_denoised": None,
                            "autoguidance_w": self.w_ag,
                        }
                    )
                if shared_model:
                    _activate(self.model_patcher, (self.bad_model_patcher,))
                return denoised

            # Prefer bad-model filtered conds for hook compatibility, but restore 'y' if required.
            pos_bad_cond = self.bad_conds["positive"] if self.bad_conds and "positive" in self.bad_conds else positive_cond
            pos_bad_cond = _clone_cond_metadata(pos_bad_cond)

            pos_bad_cond = _restore_y_per_entry(pos_bad_cond, positive_cond, negative_cond)
            pos_bad_cond = _ensure_primary_text_cond(pos_bad_cond, positive_cond)
            if self.bad_conds is not None:
                self.bad_conds["positive"] = pos_bad_cond

            # Prefer running full bad-model CFG so AG compares guided predictions
            # (cfg_good - cfg_bad), which is significantly more stable for LCM/DMD2.
            neg_src = None
            if self.bad_conds is not None:
                neg_src = self.bad_conds.get("negative", None)
            neg_bad = _clone_cond_metadata(neg_src if neg_src is not None else uncond_for_good)
            neg_bad = _restore_y_per_entry(neg_bad, positive_cond, negative_cond)
            neg_bad = _ensure_primary_text_cond(neg_bad, uncond_for_good)
            if self.bad_conds is not None:
                self.bad_conds["negative"] = neg_bad
            conds_bad: List[Any] = [pos_bad_cond, neg_bad]

            def _run_bad(conds_list: List[Any]) -> List[Any]:
                if shared_model:
                    _activate(self.bad_model_patcher, (self.model_patcher,))
                    if not hasattr(self, "_ag_dbg_flip_bad_once"):
                        m = getattr(self.bad_model_patcher, "model", None)
                        print(
                            "[AutoGuidance] bad_active?",
                            getattr(m, "current_weight_patches_uuid", None),
                            "want",
                            getattr(self.bad_model_patcher, "patches_uuid", None),
                        )
                        self._ag_dbg_flip_bad_once = True
                if shared_model and not hasattr(self, "_ag_uuid_check_bad_once"):
                    m = getattr(self.bad_model_patcher, "model", None)
                    print("[AutoGuidance] after_bad_activate current_uuid=", getattr(m, "current_weight_patches_uuid", None))
                    self._ag_uuid_check_bad_once = True
                self.inner_bad_model.current_patcher = self.bad_model_patcher
                return comfy.samplers.calc_cond_batch(
                    self.inner_bad_model,
                    conds_list,
                    x,
                    timestep,
                    bad_model_options,
                )

            try:
                out_bad = _run_bad(conds_bad)
            except AssertionError as e:
                if "must specify y if and only if the model is class-conditional" not in str(e):
                    raise
                pos_bad_cond = _clone_cond_metadata(pos_bad_cond)
                pos_bad_cond = _restore_y_per_entry(pos_bad_cond, positive_cond, negative_cond)
                pos_bad_cond = _ensure_primary_text_cond(pos_bad_cond, positive_cond)
                neg_bad = _clone_cond_metadata(uncond_for_good)
                neg_bad = _restore_y_per_entry(neg_bad, positive_cond, negative_cond)
                neg_bad = _ensure_primary_text_cond(neg_bad, uncond_for_good)
                if self.bad_conds is not None:
                    self.bad_conds["positive"] = pos_bad_cond
                    self.bad_conds["negative"] = neg_bad
                conds_bad = [pos_bad_cond, neg_bad]
                try:
                    out_bad = _run_bad(conds_bad)
                except RuntimeError as e2:
                    if "mat1 and mat2 shapes cannot be multiplied" not in str(e2):
                        raise
                    pos_bad_cond = _clone_cond_metadata(positive_cond)
                    pos_bad_cond = _restore_y_per_entry(pos_bad_cond, positive_cond, negative_cond)
                    pos_bad_cond = _ensure_primary_text_cond(pos_bad_cond, positive_cond)
                    neg_bad = _clone_cond_metadata(uncond_for_good)
                    neg_bad = _restore_y_per_entry(neg_bad, positive_cond, negative_cond)
                    neg_bad = _ensure_primary_text_cond(neg_bad, uncond_for_good)
                    if self.bad_conds is not None:
                        self.bad_conds["positive"] = pos_bad_cond
                        self.bad_conds["negative"] = neg_bad
                    conds_bad = [pos_bad_cond, neg_bad]
                    out_bad = _run_bad(conds_bad)
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" not in str(e):
                    raise
                pos_bad_cond = _clone_cond_metadata(positive_cond)
                pos_bad_cond = _restore_y_per_entry(pos_bad_cond, positive_cond, negative_cond)
                pos_bad_cond = _ensure_primary_text_cond(pos_bad_cond, positive_cond)
                neg_bad = _clone_cond_metadata(uncond_for_good)
                neg_bad = _restore_y_per_entry(neg_bad, positive_cond, negative_cond)
                neg_bad = _ensure_primary_text_cond(neg_bad, uncond_for_good)
                if self.bad_conds is not None:
                    self.bad_conds["positive"] = pos_bad_cond
                    self.bad_conds["negative"] = neg_bad
                conds_bad = [pos_bad_cond, neg_bad]
                out_bad = _run_bad(conds_bad)
            bad_cond_pred = out_bad[0]
            bad_uncond_pred = out_bad[1]

            dbg = bool(getattr(self, "debug_metrics", False)) or (os.environ.get("AG_DEBUG_METRICS", "0") == "1")
            bad_sig_key = f"_ag_dbg_badcond_sig_once_{getattr(self.bad_model_patcher, 'patches_uuid', None)}"
            if dbg and not hasattr(self, bad_sig_key):
                try:
                    sig = float(bad_cond_pred.detach().float().flatten()[:8192].sum().cpu())
                    print("[AutoGuidance] bad_cond_pred_sig", sig)
                except Exception as e:
                    print("[AutoGuidance] bad_cond_pred_sig failed", repr(e))
                setattr(self, bad_sig_key, True)

            # Keep bad CFG computation consistent with the good path.
            # Some workflows install a sampler_cfg_function (e.g. cfg-rescale, custom thresholding).
            if "sampler_cfg_function" in model_options:
                if shared_model:
                    _activate(self.bad_model_patcher, (self.model_patcher,))
                self.inner_bad_model.current_patcher = self.bad_model_patcher
                args_bad = {
                    "cond": x - bad_cond_pred,
                    "uncond": x - bad_uncond_pred,
                    "cond_scale": self.cfg,
                    "timestep": timestep,
                    "input": x,
                    "sigma": timestep,
                    "cond_denoised": bad_cond_pred,
                    "uncond_denoised": bad_uncond_pred,
                    "model": self.inner_bad_model,
                    "model_options": bad_model_options,
                }
                bad_cfg_out = x - model_options["sampler_cfg_function"](args_bad)
            else:
                bad_cfg_out = bad_uncond_pred + (bad_cond_pred - bad_uncond_pred) * self.cfg

            # Leave both models in a known patcher state to reduce cross-talk across wrappers/cached objects.
            self.inner_bad_model.current_patcher = self.bad_model_patcher
            self.inner_model.current_patcher = self.model_patcher

            d_cfg = cond_pred_good - uncond_pred_good
            # Two useful deltas:
            # - d_ag_cfg: delta vs bad model run with the same CFG pipeline
            # - d_ag_ref: delta vs bad model conditional-only (paper-style "bad" reference)
            d_ag_cfg = cfg_out - bad_cfg_out
            d_ag_ref = cfg_out - bad_cond_pred
            w = max(float(self.w_ag - 1.0), 0.0)
            if w <= 0.0:
                denoised = cfg_out
            else:
                mode = getattr(self, "ag_delta_mode", AG_DELTA_MODE_BAD_CONDITIONAL)
                allow_neg = bool(getattr(self, "ag_allow_negative", True))

                # Choose AG direction.
                if mode == AG_DELTA_MODE_PROJECT_CFG:
                    cfg_denom = (d_cfg.float() * d_cfg.float()).sum() + 1e-8
                    # Project the "push away from bad" direction onto the CFG direction.
                    alpha = (d_ag_ref.float() * d_cfg.float()).sum() / cfg_denom
                    if not allow_neg:
                        alpha = torch.clamp(alpha, min=0.0)
                    d_ag_dir = d_cfg * alpha.to(d_cfg.dtype)
                elif mode == AG_DELTA_MODE_RAW:
                    # Delta between good- and bad-guided outputs.
                    d_ag_dir = d_ag_cfg
                else:
                    # Strongest + most LoRA-sensitive: compare good CFG output to bad conditional-only output.
                    d_ag_dir = d_ag_ref

                ag_delta = w * d_ag_dir

                # Cap magnitude relative to CFG-direction magnitude.
                max_ratio = float(getattr(self, "ag_max_ratio", AG_DEFAULT_MAX_RATIO))
                if max_ratio > 0.0:
                    n_cfg = d_cfg.float().pow(2).sum().sqrt() + 1e-8
                    n_delta = ag_delta.float().pow(2).sum().sqrt() + 1e-8
                    limit = max_ratio * n_cfg
                    scale = torch.clamp(limit / n_delta, max=1.0)
                    ag_delta = ag_delta * scale.to(ag_delta.dtype)

                    debug_metrics = bool(getattr(self, "debug_metrics", False)) or (os.environ.get("AG_DEBUG_METRICS", "0") == "1")
                    if debug_metrics and not hasattr(self, "_ag_dbg_dir_once"):
                        d1 = d_ag_dir.detach().float()
                        d2 = d_cfg.detach().float()
                        n1 = d1.pow(2).sum().sqrt() + 1e-8
                        n2 = d2.pow(2).sum().sqrt() + 1e-8
                        cos = float((d1 * d2).sum().cpu() / (n1 * n2))
                        sig = float(d1.flatten()[:8192].sum().cpu())
                        print("[AutoGuidance] dir", {"cos_ag_vs_cfg": cos, "ag_dir_sig": sig})
                        self._ag_dbg_dir_once = True

                    if debug_metrics and not hasattr(self, "_ag_dbg_metrics_once"):
                        # Useful sanity checks: if these are ~0, your bad model path isn't actually different.
                        d_cond = (cond_pred_good - bad_cond_pred).float()
                        n_cond = d_cond.pow(2).sum().sqrt() + 1e-8
                        print(
                            "[AutoGuidance] metrics",
                            {
                                "mode": mode,
                                "w_ag": float(self.w_ag),
                                "w": float(w),
                                "ag_max_ratio": float(max_ratio),
                                "n_cfg": float(n_cfg.detach().cpu()),
                                "n_delta": float(n_delta.detach().cpu()),
                                "scale": float(scale.detach().cpu()),
                                "n_good_minus_bad_cond": float(n_cond.detach().cpu()),
                            },
                        )
                        self._ag_dbg_metrics_once = True

                denoised = cfg_out + ag_delta

            for fn in model_options.get("sampler_post_cfg_function", []):
                if shared_model:
                    _activate(self.model_patcher, (self.bad_model_patcher,))
                self.inner_model.current_patcher = self.model_patcher
                args = {
                    "denoised": denoised,
                    "cond": positive_cond,
                    "uncond": uncond_for_good,
                    "model": self.inner_model,
                    "uncond_denoised": uncond_pred_good,
                    "cond_denoised": cond_pred_good,
                    "sigma": timestep,
                    "model_options": model_options,
                    "input": x,
                    "cond_scale": self.cfg,
                    "bad_model": self.inner_bad_model,
                    "bad_cond_denoised": bad_cond_pred,
                    "autoguidance_w": self.w_ag,
                }
                denoised = fn(args)

            if shared_model:
                _activate(self.model_patcher, (self.bad_model_patcher,))
            return denoised
        finally:
            try:
                if shared_model:
                    _activate(self.model_patcher, (self.bad_model_patcher,))
            except Exception:
                pass
            if self.inner_bad_model is not None:
                self.inner_bad_model.current_patcher = self.bad_model_patcher
            if self.inner_model is not None:
                self.inner_model.current_patcher = self.model_patcher


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
                # 3-mode behavior selection:
                # - shared_safe_low_vram: correctness-first, slow, minimal extra VRAM (default)
                # - shared_fast_extra_vram: faster shared-model swapping, costs extra VRAM
                # - dual_models_2x_vram: fastest, but requires truly distinct model instances (~2x VRAM)
                "swap_mode": (AG_SWAP_MODE_CHOICES, {"default": AG_SWAP_MODE_SHARED_SAFE}),
            },
            "optional": {
                # If AG looks too subtle, increase ag_max_ratio first (0.35 -> 0.75).
                "ag_delta_mode": (AG_DELTA_MODE_CHOICES, {"default": AG_DELTA_MODE_BAD_CONDITIONAL}),
                "ag_max_ratio": (
                    "FLOAT",
                    {"default": AG_DEFAULT_MAX_RATIO, "min": 0.0, "max": 2.0, "step": 0.05, "round": 0.01},
                ),
                # Only affects project_cfg mode; when False, opposite-direction AG becomes 0.
                "ag_allow_negative": ("BOOLEAN", {"default": True}),

                # === Swap/ownership safety tuning ===
                # In shared_safe_low_vram mode, force a clean swap every time we switch between good/bad.
                # This is slower but fixes "washed out" or "same output" issues on some ComfyUI builds.
                "safe_force_clean_swap": ("BOOLEAN", {"default": True}),
                # If enabled, treat "same UUID" as a no-op without further ownership validation.
                # Useful for debugging, but can hide broken swaps.
                "uuid_only_noop": ("BOOLEAN", {"default": False}),

                # === Debug ===
                # Print a one-time lightweight weight signature per active patcher swap.
                "debug_swap": ("BOOLEAN", {"default": True}),
                # Print one-time AG magnitude metrics (including good-vs-bad-cond distance).
                "debug_metrics": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/guiders"

    def get_guider(
        self,
        good_model,
        bad_model,
        positive,
        negative,
        cfg: float,
        w_autoguide: float,
        swap_mode=AG_SWAP_MODE_SHARED_SAFE,
        ag_delta_mode=AG_DELTA_MODE_BAD_CONDITIONAL,
        ag_max_ratio=AG_DEFAULT_MAX_RATIO,
        ag_allow_negative=True,
        safe_force_clean_swap=True,
        uuid_only_noop=False,
        debug_swap=True,
        debug_metrics=True,
    ):
        guider = Guider_AutoGuidanceCFG(
            good_model,
            bad_model,
            swap_mode=swap_mode,
            ag_delta_mode=ag_delta_mode,
            ag_max_ratio=ag_max_ratio,
            ag_allow_negative=ag_allow_negative,
            safe_force_clean_swap=safe_force_clean_swap,
            uuid_only_noop=uuid_only_noop,
            debug_swap=debug_swap,
            debug_metrics=debug_metrics,
        )
        # Make swap debugging available inside patcher-level helpers.
        for _p in (good_model, bad_model):
            _m = getattr(_p, "model", None)
            if _m is not None:
                try:
                    setattr(_m, "_ag_debug_swap", bool(debug_swap))
                except Exception:
                    pass
        guider.set_conds(positive, negative)
        guider.set_scales(cfg=cfg, w_ag=w_autoguide)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "AutoGuidanceCFGGuider": AutoGuidanceCFGGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoGuidanceCFGGuider": "AutoGuidance CFG Guider (good+bad)",
}
