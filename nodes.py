from __future__ import annotations

import math
import inspect
import os
from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch
import comfy.sampler_helpers
import comfy.samplers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks


def _to_sigma_scalar(v) -> float | None:
    """Return scalar sigma even if passed as a batch vector (shape [B])."""
    if v is None:
        return None
    if torch.is_tensor(v):
        t = v.detach()
        if t.numel() == 0:
            return None
        t = t.reshape(-1)
        if t.numel() == 1:
            return float(t.item())
        # Usually all batch sigmas are identical; if not, be conservative.
        if not torch.allclose(t, t[0], rtol=1e-3, atol=1e-5):
            return float(t.max().item())
        return float(t[0].item())
    try:
        return float(v)
    except Exception:
        return None


def _flatten_per_sample(t: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dimensions into one vector per batch item."""
    if t.ndim == 0:
        return t.reshape(1, 1)
    if t.ndim == 1:
        return t.reshape(t.shape[0], 1)
    if t.ndim == 3:
        return t.reshape(1, -1)
    return t.reshape(t.shape[0], -1)


def _dot_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample dot product over non-batch dimensions."""
    af = _flatten_per_sample(a.float())
    bf = _flatten_per_sample(b.float())
    return (af * bf).sum(dim=1)


def _norm_per_sample(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample L2 norm over non-batch dimensions."""
    return _dot_per_sample(t, t).sqrt() + eps


def _expand_batch_scale(scale: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Expand [B] scale to broadcast with ref tensor shape [B,...]."""
    if ref.ndim <= 1:
        return scale
    return scale.reshape(scale.shape[0], *([1] * (ref.ndim - 1)))


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
AG_DELTA_MODE_REJECT_CFG = "reject_cfg"

AG_DELTA_MODE_CHOICES = [
    AG_DELTA_MODE_BAD_CONDITIONAL,
    AG_DELTA_MODE_RAW,
    AG_DELTA_MODE_PROJECT_CFG,
    AG_DELTA_MODE_REJECT_CFG,
]

AG_POST_CFG_KEEP = "keep"
AG_POST_CFG_APPLY_AFTER = "apply_after"
AG_POST_CFG_SKIP = "skip"

AG_POST_CFG_MODE_CHOICES = [
    AG_POST_CFG_KEEP,
    AG_POST_CFG_APPLY_AFTER,
    AG_POST_CFG_SKIP,
]

AG_COMBINE_MODE_SEQUENTIAL_DELTA = "sequential_delta"
AG_COMBINE_MODE_MULTI_GUIDANCE_PAPER = "multi_guidance_paper"

AG_COMBINE_MODE_CHOICES = [
    AG_COMBINE_MODE_SEQUENTIAL_DELTA,
    AG_COMBINE_MODE_MULTI_GUIDANCE_PAPER,
]

AG_RAMP_FLAT = "flat"
AG_RAMP_DETAIL_LATE = "detail_late"
AG_RAMP_COMPOSE_EARLY = "compose_early"
AG_RAMP_MID_PEAK = "mid_peak"

AG_RAMP_MODE_CHOICES = [
    AG_RAMP_FLAT,
    AG_RAMP_DETAIL_LATE,
    AG_RAMP_COMPOSE_EARLY,
    AG_RAMP_MID_PEAK,
]

# Default cap for AG magnitude relative to the actually applied CFG update magnitude.
# Old code used 0.10 which is often visually near-invisible.
AG_DEFAULT_MAX_RATIO = 0.35


def _ag_ramp_factor(prog01: float, *, mode: str, power: float) -> float:
    """
    prog01: 0 early (high sigma) -> 1 late (low sigma)
    Returns factor in [0,1] to scale max_ratio.
    """
    p = max(1e-6, float(power))
    x = max(0.0, min(1.0, float(prog01)))

    if mode == AG_RAMP_FLAT:
        f = 1.0
    elif mode == AG_RAMP_DETAIL_LATE:
        f = x ** p
    elif mode == AG_RAMP_COMPOSE_EARLY:
        f = (1.0 - x) ** p
    elif mode == AG_RAMP_MID_PEAK:
        bell = 4.0 * x * (1.0 - x)
        f = max(0.0, bell) ** p
    else:
        f = 1.0

    return max(0.0, min(1.0, float(f)))


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


def _layer_digest(model, name: str):
    try:
        for param_name, param in model.named_parameters():
            if param_name == name:
                t = param.detach().float().flatten()
                return float(t[:4096].sum().cpu()), float(t[:4096].abs().sum().cpu())
    except Exception:
        pass
    return None


def _debug_print_layer_digest_once(
    model,
    patcher,
    *,
    enabled: bool | None = None,
    layer_name: str = "diffusion_model.input_blocks.1.0.in_layers.2.weight",
) -> None:
    if enabled is None:
        enabled = bool(getattr(model, "_ag_debug_swap", False)) or os.environ.get("AG_DEBUG_SWAP", "0") == "1"
    if not enabled:
        return

    uuid = getattr(patcher, "patches_uuid", None)
    key = f"_ag_dbg_layer_once_{uuid}_{id(patcher)}"
    if hasattr(model, key):
        return

    digest = _layer_digest(model, layer_name)
    print("[AutoGuidance] layer_digest", {"layer": layer_name, "uuid": uuid, "digest": digest})
    setattr(model, key, True)


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

    def _debug_after_activate() -> None:
        _debug_print_weight_signature_once(m, patcher, tag=str(getattr(patcher, "patches_uuid", "none")), enabled=debug_swap)
        _debug_print_layer_digest_once(m, patcher, enabled=debug_swap)

    if safe_force_clean:
        # Avoid doing the expensive work repeatedly if we already activated this patcher last time.
        last_id = getattr(m, "_ag_last_patcher_id", None)
        if last_id == id(patcher):
            m.current_patcher = patcher
            m.current_weight_patches_uuid = getattr(patcher, "patches_uuid", None)
            _debug_after_activate()
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
        _debug_after_activate()
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
        _debug_after_activate()
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
                _debug_after_activate()
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
            _debug_after_activate()
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
        _debug_after_activate()
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
        _debug_after_activate()
        return

    _call_patch_model_safe(patcher, device=device)

    m.current_patcher = patcher
    m.current_weight_patches_uuid = getattr(patcher, "patches_uuid", None)
    _debug_after_activate()


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


def _sig_has_param(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    return name in sig.parameters


def _find_first_key(cond: Any, key: str):
    if isinstance(cond, Mapping):
        if key in cond:
            return cond.get(key)
        for v in cond.values():
            if isinstance(v, (Mapping, list, tuple)):
                out = _find_first_key(v, key)
                if out is not None:
                    return out
        return None
    if isinstance(cond, (list, tuple)):
        for v in cond:
            if isinstance(v, (Mapping, list, tuple)):
                out = _find_first_key(v, key)
                if out is not None:
                    return out
    return None


def _find_first_3d_tensor(cond: Any):
    if torch.is_tensor(cond) and cond.ndim == 3:
        return cond
    if isinstance(cond, Mapping):
        for v in cond.values():
            if isinstance(v, (Mapping, list, tuple)) or torch.is_tensor(v):
                t = _find_first_3d_tensor(v)
                if t is not None:
                    return t
        return None
    if isinstance(cond, (list, tuple)):
        for v in cond:
            if isinstance(v, (Mapping, list, tuple)) or torch.is_tensor(v):
                t = _find_first_3d_tensor(v)
                if t is not None:
                    return t
    return None


def _infer_num_tokens(cond: Any) -> int | None:
    t = _find_first_3d_tensor(cond)
    if t is None:
        return None
    try:
        return int(t.shape[1])
    except Exception:
        return None


def _ensure_key_everywhere(cond: Any, key: str, value) -> None:
    if value is None:
        return
    if isinstance(cond, Mapping):
        try:
            if key not in cond:
                cond[key] = value
        except Exception:
            pass
        for v in cond.values():
            if isinstance(v, (Mapping, list, tuple)):
                _ensure_key_everywhere(v, key, value)
        return
    if isinstance(cond, (list, tuple)):
        for v in cond:
            if isinstance(v, (Mapping, list, tuple)):
                _ensure_key_everywhere(v, key, value)


def _restore_key_per_entry(dst_cond: Any, src_pos: Any, src_neg: Any = None, *, key: str) -> Any:
    """
    Restore extra conditioning fields (e.g. num_tokens) per entry without collapsing structures.
    If src doesn't provide it, infer from a 3D context tensor (shape [B, T, D] -> T).
    """
    is_tuple = isinstance(dst_cond, tuple)
    if isinstance(dst_cond, (list, tuple)):
        dst_list = list(dst_cond)
        src_pos_list = list(src_pos) if isinstance(src_pos, (list, tuple)) else None
        src_neg_list = list(src_neg) if isinstance(src_neg, (list, tuple)) else None

        default_val = _find_first_key(src_pos, key) or _find_first_key(src_neg, key)
        if default_val is None:
            default_val = _infer_num_tokens(src_pos) or _infer_num_tokens(src_neg)

        for i in range(len(dst_list)):
            val = None
            if src_pos_list is not None and i < len(src_pos_list):
                val = _find_first_key(src_pos_list[i], key) or _infer_num_tokens(src_pos_list[i])
            if val is None and src_neg_list is not None and i < len(src_neg_list):
                val = _find_first_key(src_neg_list[i], key) or _infer_num_tokens(src_neg_list[i])
            if val is None:
                val = _infer_num_tokens(dst_list[i]) or default_val
            if val is not None:
                _ensure_key_everywhere(dst_list[i], key, val)
        return tuple(dst_list) if is_tuple else dst_list

    val = _find_first_key(src_pos, key) or _find_first_key(src_neg, key)
    if val is None:
        val = _infer_num_tokens(src_pos) or _infer_num_tokens(src_neg) or _infer_num_tokens(dst_cond)
    if val is not None:
        _ensure_key_everywhere(dst_cond, key, val)
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
        ag_ramp_mode: str = AG_RAMP_FLAT,
        ag_ramp_power: float = 2.0,
        ag_ramp_floor: float = 0.0,
        ag_post_cfg_mode: str = AG_POST_CFG_KEEP,
        ag_combine_mode: str = AG_COMBINE_MODE_SEQUENTIAL_DELTA,
        safe_force_clean_swap: bool = True,
        uuid_only_noop: bool = False,
        debug_swap: bool = False,
        debug_metrics: bool = False,
        debug_metrics_all: bool = False,
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
        self.ag_ramp_mode: str = str(ag_ramp_mode)
        self.ag_ramp_power: float = float(ag_ramp_power)
        self.ag_ramp_floor: float = float(ag_ramp_floor)
        self.ag_post_cfg_mode: str = str(ag_post_cfg_mode)
        self.ag_combine_mode: str = str(ag_combine_mode)
        self.safe_force_clean_swap: bool = bool(safe_force_clean_swap)
        self.uuid_only_noop: bool = bool(uuid_only_noop)
        self.debug_swap: bool = bool(debug_swap)
        self.debug_metrics: bool = bool(debug_metrics)
        self.debug_metrics_all: bool = bool(debug_metrics_all)

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
        try:
            v = sigmas[0] if sigmas is not None and len(sigmas) > 0 else None
            if v is None:
                self._ag_sigma_max = None
            elif torch.is_tensor(v):
                self._ag_sigma_max = float(v.detach().cpu().item())
            else:
                self._ag_sigma_max = float(v)
        except Exception:
            self._ag_sigma_max = None
        sigmas = sigmas.to(device)
        self._ag_step = 0
        try:
            self._ag_steps_total = int(sigmas.shape[0])
        except Exception:
            self._ag_steps_total = None

        comfy.samplers.cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            self.bad_model_patcher.pre_run()

            # Dual-model sanity: confirm good/bad model weights differ when swap activation is not used.
            if getattr(self, "swap_mode", None) == AG_SWAP_MODE_DUAL_MODELS:
                dbg = bool(getattr(self, "debug_swap", False)) or (os.environ.get("AG_DEBUG_SWAP", "0") == "1")
                if dbg and not hasattr(self, "_ag_dbg_dual_digest_once"):
                    layer = "diffusion_model.input_blocks.1.0.in_layers.2.weight"
                    gm = getattr(self.model_patcher, "model", None)
                    bm = getattr(self.bad_model_patcher, "model", None)
                    print("[AutoGuidance] dual_layer_digest", {"good": _layer_digest(gm, layer), "bad": _layer_digest(bm, layer)})
                    self._ag_dbg_dual_digest_once = True

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
        Mirrors comfy.samplers.sampling_function + cfg_function, but injects AutoGuidance.
        NOTE: Comfy may also apply sampler_post_cfg_function outside the guider. If we handle
        post-CFG here, we strip the list in-place to avoid double-application.
        """
        if model_options is None:
            model_options = {}

        # IMPORTANT: model_options may be reused by wrappers/caches across calls.
        # Copy before mutating (we strip post-cfg hooks below).
        model_options = dict(model_options)

        # Ensure wrapper-related options always exist
        if model_options.get("transformer_options", None) is None:
            model_options["transformer_options"] = {}

        step = int(getattr(self, "_ag_step", 0))
        self._ag_step = step + 1
        total_steps = getattr(self, "_ag_steps_total", None)
        last = (total_steps is not None and step == total_steps - 1)

        post_cfg_mode = str(getattr(self, "ag_post_cfg_mode", AG_POST_CFG_KEEP))
        # Copy the hook list for our internal use, but IMPORTANT: strip it from model_options
        # so Comfy doesn't apply it again outside this guider.
        orig_post_cfg_fns = model_options.get("sampler_post_cfg_function", []) or []
        post_cfg_fns = list(orig_post_cfg_fns)
        if post_cfg_mode == AG_POST_CFG_SKIP:
            post_cfg_fns = []
        # If there were any hooks, strip them in-place to prevent double application downstream.
        # This is intentional in all modes (including keep) to enforce exactly-once semantics.
        # (We either run them ourselves, or intentionally skip them.)
        if orig_post_cfg_fns:
            try:
                model_options["sampler_post_cfg_function"] = []
            except Exception:
                pass
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
            ag_combine_mode = str(getattr(self, "ag_combine_mode", AG_COMBINE_MODE_SEQUENTIAL_DELTA))
            # Paper-style multi-guidance (Karras et al.) is typically parameterized with
            # non-negative weights w_cfg,w_ag and uses cfg>=1 as the "no-guidance" baseline.
            #
            # In practice it is also useful to run CFG < 1.0 (interpolating towards uncond),
            # while still using multi-guidance.
            #
            # Allow cfg < 1.0 by not clamping (cfg=1.0 -> w_cfg=0, cfg<1.0 -> w_cfg<0).
            w_cfg_paper = float(self.cfg) - 1.0
            w_ag_paper = max(float(self.w_ag) - 1.0, 0.0)
            cfg_effective_paper = 1.0 + w_cfg_paper + w_ag_paper
            cond_scale_for_pre_cfg = cfg_effective_paper if ag_combine_mode == AG_COMBINE_MODE_MULTI_GUIDANCE_PAPER else float(self.cfg)

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

            def _apply_pre_cfg_hooks(
                *,
                conds_list: List[Any],
                out_list: List[Any],
                run_model,
                active_patcher,
                other_patcher,
                options: Dict[str, Any],
                cond_scale_override: Optional[float] = None,
            ) -> List[Any]:
                out_cur = out_list
                for fn in options.get("sampler_pre_cfg_function", []):
                    if shared_model:
                        _activate(active_patcher, (other_patcher,))
                    run_model.current_patcher = active_patcher
                    args = {
                        "conds": conds_list,
                        "conds_out": out_cur,
                        "cond_scale": cond_scale_override if cond_scale_override is not None else self.cfg,
                        "timestep": timestep,
                        "input": x,
                        "sigma": timestep,
                        "model": run_model,
                        "model_options": options,
                    }
                    out_cur = fn(args)
                return out_cur

            out_good = _apply_pre_cfg_hooks(
                conds_list=conds_good,
                out_list=out_good,
                run_model=self.inner_model,
                active_patcher=self.model_patcher,
                other_patcher=self.bad_model_patcher,
                options=model_options,
                cond_scale_override=cond_scale_for_pre_cfg,
            )

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

            cfg_out = None
            if ag_combine_mode != AG_COMBINE_MODE_MULTI_GUIDANCE_PAPER:
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

            # Fast path: AG disabled.
            if self.w_ag <= 1.0 + 1e-6:
                if ag_combine_mode == AG_COMBINE_MODE_MULTI_GUIDANCE_PAPER:
                    cfg_effective = 1.0 + w_cfg_paper
                    origin = uncond_pred_good
                    if "sampler_cfg_function" in model_options:
                        if shared_model:
                            _activate(self.model_patcher, (self.bad_model_patcher,))
                        self.inner_model.current_patcher = self.model_patcher
                        args = {
                            "cond": x - cond_pred_good,
                            "uncond": x - origin,
                            "cond_scale": cfg_effective,
                            "timestep": timestep,
                            "input": x,
                            "sigma": timestep,
                            "cond_denoised": cond_pred_good,
                            "uncond_denoised": origin,
                            "model": self.inner_model,
                            "model_options": model_options,
                        }
                        denoised = x - model_options["sampler_cfg_function"](args)
                    else:
                        denoised = origin + (cond_pred_good - origin) * cfg_effective
                    cond_scale_for_hooks = cfg_effective
                    uncond_denoised_for_hooks = origin
                else:
                    denoised = cfg_out
                    cond_scale_for_hooks = self.cfg
                    uncond_denoised_for_hooks = uncond_pred_good

                for fn in post_cfg_fns:
                    if shared_model:
                        _activate(self.model_patcher, (self.bad_model_patcher,))
                    self.inner_model.current_patcher = self.model_patcher
                    denoised = fn(
                        {
                            "denoised": denoised,
                            "cond": positive_cond,
                            "uncond": uncond_for_good,
                            "model": self.inner_model,
                            "uncond_denoised": uncond_denoised_for_hooks,
                            "cond_denoised": cond_pred_good,
                            "sigma": timestep,
                            "model_options": model_options,
                            "input": x,
                            "cond_scale": cond_scale_for_hooks,
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

            # Z-Image / NextDiT requires num_tokens; ensure it exists on bad pass.
            key = type(getattr(self.inner_bad_model, "diffusion_model", None))
            if getattr(self, "_ag_needs_num_tokens_key", None) != key:
                needs = (_find_first_key(positive_cond, "num_tokens") is not None) or (_find_first_key(negative_cond, "num_tokens") is not None)
                if not needs:
                    try:
                        dm = getattr(self.inner_bad_model, "diffusion_model", None)
                        fn = getattr(dm, "forward", None) if dm is not None else None
                        needs = bool(fn is not None and _sig_has_param(fn, "num_tokens"))
                    except Exception:
                        needs = False
                self._ag_needs_num_tokens = needs
                self._ag_needs_num_tokens_key = key

            if self._ag_needs_num_tokens:
                pos_bad_cond = _restore_key_per_entry(pos_bad_cond, positive_cond, negative_cond, key="num_tokens")
                neg_bad = _restore_key_per_entry(neg_bad, negative_cond, positive_cond, key="num_tokens")
                if self.bad_conds is not None:
                    self.bad_conds["positive"] = pos_bad_cond
                    self.bad_conds["negative"] = neg_bad

                dbg_nt = bool(getattr(self, "debug_metrics", False)) or (os.environ.get("AG_DEBUG_METRICS", "0") == "1")
                if dbg_nt and step == 0 and not hasattr(self, "_ag_dbg_num_tokens_once"):
                    print("[AG] good num_tokens", _find_first_key(positive_cond, "num_tokens"), _find_first_key(negative_cond, "num_tokens"))
                    print("[AG] bad  num_tokens", _find_first_key(pos_bad_cond, "num_tokens"), _find_first_key(neg_bad, "num_tokens"))
                    self._ag_dbg_num_tokens_once = True
            conds_bad: List[Any] = [pos_bad_cond, neg_bad]

            def _run_bad(conds_list: List[Any]) -> tuple[List[Any], Dict[str, Any]]:
                # IMPORTANT: build bad_model_options *after* the good pass and any sampler_pre_cfg hooks.
                # Some model families (e.g. NextDiT used by Z-Image) inject per-step fields (like num_tokens)
                # into model_options/transformer_options during the good pass. If we clone too early, the bad
                # pass can miss those required fields and crash.
                #
                # Keep this clone inside _run_bad so every bad-pass invocation gets a fresh, isolated copy,
                # including retry paths in the exception handlers below.
                bad_model_options = _clone_model_options_for_bad(model_options)
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
                ), bad_model_options

            def _rebuild_bad_pos_from_good() -> Any:
                rebuilt = _clone_cond_metadata(positive_cond)
                rebuilt = _restore_y_per_entry(rebuilt, positive_cond, negative_cond)
                rebuilt = _ensure_primary_text_cond(rebuilt, positive_cond)
                if self.bad_conds is not None:
                    self.bad_conds["positive"] = rebuilt
                return rebuilt

            def _rebuild_bad_neg_from_good() -> Any:
                rebuilt = _clone_cond_metadata(uncond_for_good)
                rebuilt = _restore_y_per_entry(rebuilt, positive_cond, negative_cond)
                rebuilt = _ensure_primary_text_cond(rebuilt, uncond_for_good)
                if self.bad_conds is not None:
                    self.bad_conds["negative"] = rebuilt
                return rebuilt

            def _run_bad_full_with_fallback(pos_in: Any, neg_in: Any):
                pos_local = pos_in
                neg_local = neg_in
                conds_local = [pos_local, neg_local]
                try:
                    return _run_bad(conds_local), conds_local
                except AssertionError as e:
                    if "must specify y if and only if the model is class-conditional" not in str(e):
                        raise
                    pos_local = _clone_cond_metadata(pos_local)
                    pos_local = _restore_y_per_entry(pos_local, positive_cond, negative_cond)
                    pos_local = _ensure_primary_text_cond(pos_local, positive_cond)
                    neg_local = _rebuild_bad_neg_from_good()
                    if self.bad_conds is not None:
                        self.bad_conds["positive"] = pos_local
                    conds_local = [pos_local, neg_local]
                    try:
                        return _run_bad(conds_local), conds_local
                    except RuntimeError as e2:
                        if "mat1 and mat2 shapes cannot be multiplied" not in str(e2):
                            raise
                        pos_local = _rebuild_bad_pos_from_good()
                        neg_local = _rebuild_bad_neg_from_good()
                        conds_local = [pos_local, neg_local]
                        return _run_bad(conds_local), conds_local
                except RuntimeError as e:
                    if "mat1 and mat2 shapes cannot be multiplied" not in str(e):
                        raise
                    pos_local = _rebuild_bad_pos_from_good()
                    neg_local = _rebuild_bad_neg_from_good()
                    conds_local = [pos_local, neg_local]
                    return _run_bad(conds_local), conds_local

            if ag_combine_mode == AG_COMBINE_MODE_MULTI_GUIDANCE_PAPER:
                has_pre_cfg = bool(model_options.get("sampler_pre_cfg_function", []))
                if has_pre_cfg:
                    (out_bad, bad_opts_used), conds_bad = _run_bad_full_with_fallback(pos_bad_cond, neg_bad)
                    out_bad = _apply_pre_cfg_hooks(
                        conds_list=conds_bad,
                        out_list=out_bad,
                        run_model=self.inner_bad_model,
                        active_patcher=self.bad_model_patcher,
                        other_patcher=self.model_patcher,
                        options=bad_opts_used,
                        cond_scale_override=cond_scale_for_pre_cfg,
                    )
                    bad_cond_pred = out_bad[0]
                    bad_uncond_pred = out_bad[1]
                else:
                    conds_bad = [pos_bad_cond]
                    try:
                        out_bad, bad_opts_used = _run_bad(conds_bad)
                    except AssertionError as e:
                        if "must specify y if and only if the model is class-conditional" not in str(e):
                            raise
                        pos_bad_cond = _rebuild_bad_pos_from_good()
                        conds_bad = [pos_bad_cond]
                        try:
                            out_bad, bad_opts_used = _run_bad(conds_bad)
                        except RuntimeError as e2:
                            if "mat1 and mat2 shapes cannot be multiplied" not in str(e2):
                                raise
                            pos_bad_cond = _rebuild_bad_pos_from_good()
                            conds_bad = [pos_bad_cond]
                            out_bad, bad_opts_used = _run_bad(conds_bad)
                    except RuntimeError as e:
                        if "mat1 and mat2 shapes cannot be multiplied" not in str(e):
                            raise
                        pos_bad_cond = _rebuild_bad_pos_from_good()
                        conds_bad = [pos_bad_cond]
                        out_bad, bad_opts_used = _run_bad(conds_bad)

                    out_bad = _apply_pre_cfg_hooks(
                        conds_list=conds_bad,
                        out_list=out_bad,
                        run_model=self.inner_bad_model,
                        active_patcher=self.bad_model_patcher,
                        other_patcher=self.model_patcher,
                        options=bad_opts_used,
                        cond_scale_override=cond_scale_for_pre_cfg,
                    )
                    bad_cond_pred = out_bad[0]
                    bad_uncond_pred = None
            else:
                (out_bad, bad_opts_used), conds_bad = _run_bad_full_with_fallback(pos_bad_cond, neg_bad)
                bad_cond_pred = out_bad[0]
                bad_uncond_pred = out_bad[1]

                out_bad = _apply_pre_cfg_hooks(
                    conds_list=conds_bad,
                    out_list=out_bad,
                    run_model=self.inner_bad_model,
                    active_patcher=self.bad_model_patcher,
                    other_patcher=self.model_patcher,
                    options=bad_opts_used,
                    cond_scale_override=cond_scale_for_pre_cfg,
                )
                bad_cond_pred = out_bad[0]
                bad_uncond_pred = out_bad[1]

            if ag_combine_mode == AG_COMBINE_MODE_MULTI_GUIDANCE_PAPER:
                w_cfg = w_cfg_paper
                w_ag = w_ag_paper
                w_sum = w_cfg + w_ag
                cfg_effective = 1.0 + w_sum

                # For w_cfg,w_ag>=0, we can express the paper formula via an "origin" and a
                # standard CFG step (optionally passing through sampler_cfg_function). When
                # cfg < 1.0, w_cfg becomes negative; the algebra still holds for w_sum!=0 but
                # the previous implementation short-circuited to cond_pred_good for w_sum<=0.
                #
                # Handle the w_sum≈0 edge case with the direct closed-form instead to avoid
                # division by ~0. For hooks, we still provide a reasonable origin (good uncond).
                if abs(w_sum) <= 1e-8:
                    origin = uncond_pred_good
                    denoised = (1.0 + w_sum) * cond_pred_good - (w_cfg * uncond_pred_good) - (w_ag * bad_cond_pred)
                else:
                    origin = ((w_cfg * uncond_pred_good) + (w_ag * bad_cond_pred)) / w_sum
                    if "sampler_cfg_function" in model_options:
                        if shared_model:
                            _activate(self.model_patcher, (self.bad_model_patcher,))
                        self.inner_model.current_patcher = self.model_patcher
                        args = {
                            "cond": x - cond_pred_good,
                            "uncond": x - origin,
                            "cond_scale": cfg_effective,
                            "timestep": timestep,
                            "input": x,
                            "sigma": timestep,
                            "cond_denoised": cond_pred_good,
                            "uncond_denoised": origin,
                            "model": self.inner_model,
                            "model_options": model_options,
                        }
                        denoised = x - model_options["sampler_cfg_function"](args)
                    else:
                        denoised = origin + (cond_pred_good - origin) * cfg_effective

                denoised_before_post = denoised
                for fn in post_cfg_fns:
                    if shared_model:
                        _activate(self.model_patcher, (self.bad_model_patcher,))
                    self.inner_model.current_patcher = self.model_patcher
                    args = {
                        "denoised": denoised,
                        "cond": positive_cond,
                        "uncond": uncond_for_good,
                        "model": self.inner_model,
                        "uncond_denoised": origin,
                        "cond_denoised": cond_pred_good,
                        "sigma": timestep,
                        "model_options": model_options,
                        "input": x,
                        "cond_scale": cfg_effective,
                        "bad_model": self.inner_bad_model,
                        "bad_cond_denoised": bad_cond_pred,
                        "autoguidance_w": self.w_ag,
                    }
                    denoised = fn(args)

                denoised_after_post = denoised
                debug_metrics = bool(getattr(self, "debug_metrics", False)) or (os.environ.get("AG_DEBUG_METRICS", "0") == "1")
                debug_all = bool(getattr(self, "debug_metrics_all", False)) or (os.environ.get("AG_DEBUG_METRICS_ALL", "0") == "1")
                if debug_metrics and (debug_all or step == 0 or last):
                    try:
                        def _l2(t):
                            return float(t.detach().float().pow(2).sum().sqrt().cpu())

                        n_pre = _l2(denoised_before_post - cond_pred_good)
                        n_post = _l2(denoised_after_post - cond_pred_good)
                        n_change = _l2(denoised_after_post - denoised_before_post)
                        print(
                            "[AutoGuidance] paper_multi_post_cfg_effect_step",
                            step,
                            {
                                "cond_scale_effective": float(cfg_effective),
                                "w_cfg": float(w_cfg),
                                "w_ag": float(w_ag),
                                "n_pre_minus_cond": n_pre,
                                "n_post_minus_cond": n_post,
                                "n_post_change": n_change,
                            },
                        )
                    except Exception as e:
                        print("[AutoGuidance] paper_multi_post_cfg_effect failed", repr(e))

                if shared_model:
                    _activate(self.model_patcher, (self.bad_model_patcher,))
                return denoised

            dbg = bool(getattr(self, "debug_metrics", False)) or (os.environ.get("AG_DEBUG_METRICS", "0") == "1")
            if dbg and step == 0 and not hasattr(self, "_ag_dbg_post_cfg_once"):
                try:
                    print("[AutoGuidance] post_cfg_functions", [getattr(fn, "__name__", type(fn).__name__) for fn in post_cfg_fns])
                    print("[AutoGuidance] post_cfg_mode", post_cfg_mode)
                except Exception as e:
                    print("[AutoGuidance] post_cfg_functions failed", repr(e))
                self._ag_dbg_post_cfg_once = True
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
                    "model_options": bad_opts_used,
                }
                bad_cfg_out = x - model_options["sampler_cfg_function"](args_bad)
            else:
                bad_cfg_out = bad_uncond_pred + (bad_cond_pred - bad_uncond_pred) * self.cfg

            # Leave both models in a known patcher state to reduce cross-talk across wrappers/cached objects.
            self.inner_bad_model.current_patcher = self.bad_model_patcher
            self.inner_model.current_patcher = self.model_patcher

            d_cfg = cond_pred_good - uncond_pred_good
            cfg_update_dir = cfg_out - uncond_pred_good  # actual applied CFG update direction
            # Two useful AG deltas:
            # - d_ag_cond: conditional good-vs-bad direction (paper-style push-away-from-bad)
            # - d_ag_cfg:  CFG output good-vs-bad direction (optional/raw mode)
            d_ag_cond = cond_pred_good - bad_cond_pred
            d_ag_cfg = cfg_out - bad_cfg_out
            w = max(float(self.w_ag - 1.0), 0.0)
            if w <= 0.0:
                ag_delta = None
                denoised = cfg_out
            else:
                mode = getattr(self, "ag_delta_mode", AG_DELTA_MODE_BAD_CONDITIONAL)
                allow_neg = bool(getattr(self, "ag_allow_negative", True))

                # Choose AG direction.
                if mode == AG_DELTA_MODE_PROJECT_CFG:
                    cfg_denom = _dot_per_sample(cfg_update_dir, cfg_update_dir)
                    # Project the conditional push-away-from-bad direction onto the actual CFG update direction.
                    alpha = _dot_per_sample(d_ag_cond, cfg_update_dir) / (cfg_denom + 1e-8)
                    if not allow_neg:
                        alpha = torch.clamp(alpha, min=0.0)
                    d_ag_dir = cfg_update_dir * _expand_batch_scale(alpha.to(cfg_update_dir.dtype), cfg_update_dir)
                elif mode == AG_DELTA_MODE_REJECT_CFG:
                    # Remove CFG-parallel component (w.r.t. actual CFG update) so AG can drive non-CFG structure changes.
                    cfg_denom = _dot_per_sample(cfg_update_dir, cfg_update_dir)
                    alpha = _dot_per_sample(d_ag_cond, cfg_update_dir) / (cfg_denom + 1e-8)
                    if not allow_neg:
                        alpha = torch.clamp(alpha, min=0.0)
                    d_ag_dir = d_ag_cond - cfg_update_dir * _expand_batch_scale(alpha.to(cfg_update_dir.dtype), cfg_update_dir)
                elif mode == AG_DELTA_MODE_RAW:
                    # Delta between good- and bad-guided outputs.
                    d_ag_dir = d_ag_cfg
                else:
                    # Default: conditional good-vs-bad direction.
                    d_ag_dir = d_ag_cond

                ag_delta = w * d_ag_dir

                # Cap magnitude relative to the actual applied CFG update.
                max_ratio = float(getattr(self, "ag_max_ratio", AG_DEFAULT_MAX_RATIO))
                if max_ratio > 0.0:
                    n_cfg = _norm_per_sample(cfg_update_dir)
                    n_delta = _norm_per_sample(ag_delta)
                    ratio = max_ratio
                    sigma_max = getattr(self, "_ag_sigma_max", None)
                    sigma_cur = None
                    prog = None
                    ramp_factor = 1.0
                    total_steps = getattr(self, "_ag_steps_total", None)
                    if total_steps is not None and int(total_steps) > 1:
                        prog = float(step) / float(int(total_steps) - 1)  # 0 early -> 1 late
                        prog = max(0.0, min(1.0, prog))
                        ramp_mode = str(getattr(self, "ag_ramp_mode", AG_RAMP_FLAT))
                        ramp_power = float(getattr(self, "ag_ramp_power", 2.0))
                        ramp_floor = float(getattr(self, "ag_ramp_floor", 0.0))
                        ramp_floor = max(0.0, min(1.0, ramp_floor))

                        ramp_factor = _ag_ramp_factor(prog, mode=ramp_mode, power=ramp_power)
                        ratio = max_ratio * (ramp_floor + (1.0 - ramp_floor) * ramp_factor)

                    # If cfg_update_dir is nearly zero, use a fallback norm so AG is not silently disabled.
                    n_cfg_fallback = _norm_per_sample(d_cfg)
                    n_cfg_ref = torch.where(n_cfg > 1e-6, n_cfg, n_cfg_fallback)
                    limit = ratio * n_cfg_ref
                    scale = torch.clamp(limit / (n_delta + 1e-8), max=1.0)
                    ag_delta = ag_delta * _expand_batch_scale(scale.to(ag_delta.dtype), ag_delta)
                    n_applied = _norm_per_sample(ag_delta)

                    debug_metrics = bool(getattr(self, "debug_metrics", False)) or (os.environ.get("AG_DEBUG_METRICS", "0") == "1")
                    if debug_metrics and not hasattr(self, "_ag_dbg_dir_once"):
                        d1 = d_ag_dir.detach().float()
                        d2 = cfg_update_dir.detach().float()
                        n1 = _norm_per_sample(d1).mean()
                        n2 = _norm_per_sample(d2).mean()
                        cos = float((_dot_per_sample(d1, d2).mean().cpu()) / (n1 * n2))
                        sig = float(d1.flatten()[:8192].sum().cpu())
                        print("[AutoGuidance] dir", {"cos_ag_vs_cfg_update": cos, "ag_dir_sig": sig})
                        self._ag_dbg_dir_once = True

                    debug_all = bool(getattr(self, "debug_metrics_all", False)) or (os.environ.get("AG_DEBUG_METRICS_ALL", "0") == "1")
                    if debug_metrics and (debug_all or step == 0 or last):
                        sigma_cur = _to_sigma_scalar(timestep)
                        # Useful sanity checks: if these are ~0, your bad model path isn't actually different.
                        d_cond = (cond_pred_good - bad_cond_pred).float()
                        n_cond = _norm_per_sample(d_cond).mean()
                        print(
                            "[AutoGuidance] metrics_step",
                            step,
                            {
                                "mode": mode,
                                "w_ag": float(self.w_ag),
                                "w": float(w),
                                "cfg": float(self.cfg),
                                "ag_max_ratio": float(max_ratio),
                                "ag_ramp_mode": str(getattr(self, "ag_ramp_mode", AG_RAMP_FLAT)),
                                "ag_ramp_power": float(getattr(self, "ag_ramp_power", 2.0)),
                                "ag_ramp_floor": float(getattr(self, "ag_ramp_floor", 0.0)),
                                "sigma_max": float(sigma_max) if sigma_max is not None else None,
                                "sigma_cur": float(sigma_cur) if sigma_cur is not None else None,
                                "prog": float(prog) if prog is not None else None,
                                "ramp_factor": float(ramp_factor),
                                "ratio_eff": float(ratio),
                                "limit": float(limit.mean().detach().cpu()) if torch.is_tensor(limit) else float(limit),
                                "n_cfg": float(n_cfg.mean().detach().cpu()),
                                "n_delta": float(n_delta.mean().detach().cpu()),
                                "n_delta_applied": float(n_applied.mean().detach().cpu()),
                                "scale": float(scale.mean().detach().cpu()),
                                "n_good_minus_bad_cond": float(n_cond.detach().cpu()),
                            },
                        )

                denoised = cfg_out + ag_delta

            # If requested, postpone applying AG until after sampler_post_cfg_function hooks.
            if ag_delta is not None and post_cfg_mode == AG_POST_CFG_APPLY_AFTER:
                denoised = cfg_out

            denoised_before_post = denoised

            for fn in post_cfg_fns:
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

            denoised_after_post = denoised

            if ag_delta is not None and post_cfg_mode == AG_POST_CFG_APPLY_AFTER:
                denoised = denoised + ag_delta

            # Debug how much post_cfg altered the result (and what we finally return).
            debug_metrics = bool(getattr(self, "debug_metrics", False)) or (os.environ.get("AG_DEBUG_METRICS", "0") == "1")
            debug_all = bool(getattr(self, "debug_metrics_all", False)) or (os.environ.get("AG_DEBUG_METRICS_ALL", "0") == "1")
            if debug_metrics and (debug_all or step == 0 or last):
                try:
                    def _l2(t):
                        return float(t.detach().float().pow(2).sum().sqrt().cpu())

                    n_pre = _l2(denoised_before_post - cfg_out)
                    n_post = _l2(denoised_after_post - cfg_out)
                    n_change = _l2(denoised_after_post - denoised_before_post)
                    n_final = _l2(denoised - cfg_out)
                    print(
                        "[AutoGuidance] post_cfg_effect_step",
                        step,
                        {
                            "n_pre_minus_cfg": n_pre,
                            "n_post_minus_cfg": n_post,
                            "n_post_change": n_change,
                            "n_final_minus_cfg": n_final,
                        },
                    )
                except Exception as e:
                    print("[AutoGuidance] post_cfg_effect failed", repr(e))

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
                "ag_combine_mode": (AG_COMBINE_MODE_CHOICES, {"default": AG_COMBINE_MODE_SEQUENTIAL_DELTA}),
                "ag_delta_mode": (AG_DELTA_MODE_CHOICES, {"default": AG_DELTA_MODE_BAD_CONDITIONAL}),
                "ag_max_ratio": (
                    "FLOAT",
                    {"default": AG_DEFAULT_MAX_RATIO, "min": 0.0, "max": 5.0, "step": 0.05, "round": 0.01},
                ),
                # Only affects project_cfg mode; when False, opposite-direction AG becomes 0.
                "ag_allow_negative": ("BOOLEAN", {"default": True}),
                # Ramp for AG cap over denoise progress:
                # - flat: constant cap (default, avoids zero AG at early steps)
                # - detail_late: stronger late (detail emphasis)
                # - compose_early: stronger early (composition emphasis)
                # - mid_peak: strongest at mid-steps
                "ag_ramp_mode": (AG_RAMP_MODE_CHOICES, {"default": AG_RAMP_FLAT}),
                "ag_ramp_power": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.5, "round": 0.1}),
                # Minimum always-on fraction of ag_max_ratio.
                "ag_ramp_floor": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),

                # Post-CFG hook handling (useful if a hook normalizes/clamps away AG).
                "ag_post_cfg_mode": (AG_POST_CFG_MODE_CHOICES, {"default": AG_POST_CFG_KEEP}),

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
                "debug_metrics_all": ("BOOLEAN", {"default": False}),
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
        ag_combine_mode=AG_COMBINE_MODE_SEQUENTIAL_DELTA,
        ag_delta_mode=AG_DELTA_MODE_BAD_CONDITIONAL,
        ag_max_ratio=AG_DEFAULT_MAX_RATIO,
        ag_allow_negative=True,
        ag_ramp_mode=AG_RAMP_FLAT,
        ag_ramp_power=2.0,
        ag_ramp_floor=0.15,
        ag_post_cfg_mode=AG_POST_CFG_KEEP,
        safe_force_clean_swap=True,
        uuid_only_noop=False,
        debug_swap=True,
        debug_metrics=True,
        debug_metrics_all=False,
    ):
        guider = Guider_AutoGuidanceCFG(
            good_model,
            bad_model,
            swap_mode=swap_mode,
            ag_combine_mode=ag_combine_mode,
            ag_delta_mode=ag_delta_mode,
            ag_max_ratio=ag_max_ratio,
            ag_allow_negative=ag_allow_negative,
            ag_ramp_mode=ag_ramp_mode,
            ag_ramp_power=ag_ramp_power,
            ag_ramp_floor=ag_ramp_floor,
            ag_post_cfg_mode=ag_post_cfg_mode,
            safe_force_clean_swap=safe_force_clean_swap,
            uuid_only_noop=uuid_only_noop,
            debug_swap=debug_swap,
            debug_metrics=debug_metrics,
            debug_metrics_all=debug_metrics_all,
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


# ==================== Impact Pack (FaceDetailer) integration ====================
# Provides a DETAILER_HOOK that injects an AutoGuidance GUIDER into Impact Pack's sampling path,
# without modifying Impact Pack files on disk (runtime monkeypatch).

def _ag_try_patch_comfy_samplers_sample() -> bool:
    """
    Monkeypatch comfy.samplers.sample so that, when a MODEL patcher carries a temporary
    attribute `_ag_detailer_guider`, sampling uses that GUIDER instead of CFGGuider(model).

    This keeps Impact Pack and other extensions on the normal Comfy sampling path.
    """
    try:
        import comfy.samplers as comfy_samplers  # type: ignore
    except Exception:
        return False

    # If some other extension overwrote sample after we patched, re-patch.
    cur = getattr(comfy_samplers, "sample", None)
    if getattr(cur, "_ag_autoguidance_wrapper", False):
        return True

    orig_fn = getattr(comfy_samplers, "sample", None)
    if orig_fn is None or not callable(orig_fn):
        return False

    def _call_guider_sample(guider, noise, latent_samples, sampler, sigmas, denoise_mask, cb, disable_pbar, seed, extra_kwargs=None):
        # Prefer kwargs signature (newer/guider-style), then retry without extra kwargs,
        # then fall back to positional signatures.
        extra_kwargs = extra_kwargs or {}
        try:
            return guider.sample(
                noise,
                latent_samples,
                sampler,
                sigmas,
                denoise_mask=denoise_mask,
                callback=cb,
                disable_pbar=disable_pbar,
                seed=seed,
                **extra_kwargs,
            )
        except TypeError:
            pass
        try:
            return guider.sample(
                noise,
                latent_samples,
                sampler,
                sigmas,
                denoise_mask=denoise_mask,
                callback=cb,
                disable_pbar=disable_pbar,
                seed=seed,
            )
        except TypeError:
            return guider.sample(noise, latent_samples, sampler, sigmas, denoise_mask, cb, disable_pbar, seed)

    def patched_sample(
        model,
        noise,
        positive,
        negative,
        cfg,
        device,
        sampler,
        sigmas,
        model_options=None,
        latent_image=None,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
        **kwargs,
    ):
        guider = getattr(model, "_ag_detailer_guider", None)
        if guider is None or latent_image is None:
            return orig_fn(
                model,
                noise,
                positive,
                negative,
                cfg,
                device,
                sampler,
                sigmas,
                model_options if model_options is not None else {},
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
                **kwargs,
            )

        # One-shot: prevent leakage into later samplers.
        try:
            delattr(model, "_ag_detailer_guider")
        except Exception:
            pass

        # Sync guider with final hook-modified params.
        try:
            if hasattr(guider, "set_conds"):
                guider.set_conds(positive, negative)
        except Exception:
            pass

        try:
            # Our Guider_AutoGuidanceCFG has set_scales(cfg, w_ag)
            if hasattr(guider, "set_scales"):
                w = getattr(guider, "w_ag", None)
                if w is None:
                    w = getattr(guider, "w_autoguide", 2.0)
                guider.set_scales(cfg=float(cfg), w_ag=float(w))
            elif hasattr(guider, "set_cfg"):
                guider.set_cfg(float(cfg))
            else:
                guider.cfg = float(cfg)
        except Exception:
            pass

        # Prefer non-empty per-call model_options, otherwise fall back to ModelPatcher options.
        try:
            if isinstance(model_options, dict) and len(model_options) > 0:
                guider.model_options = model_options
            else:
                guider.model_options = getattr(model, "model_options", guider.model_options)
        except Exception:
            pass

        return _call_guider_sample(
            guider,
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            extra_kwargs=kwargs,
        )

    comfy_samplers._ag_autoguidance_patched = True
    comfy_samplers._ag_autoguidance_original_sample = orig_fn
    comfy_samplers.sample = patched_sample
    try:
        setattr(comfy_samplers.sample, "_ag_autoguidance_wrapper", True)
    except Exception:
        pass
    return True


def _ag_try_import_impact_hook_base():
    try:
        from impact.hooks import DetailerHook  # type: ignore

        return DetailerHook
    except Exception:
        return None


class _AutoGuidanceImpactDetailerHookBase:
    """Fallback base if Impact Pack isn't present at import time."""

    pass


_ImpactDetailerHook = _ag_try_import_impact_hook_base() or _AutoGuidanceImpactDetailerHookBase


class AutoGuidanceImpactDetailerHook(_ImpactDetailerHook):
    """A DETAILER_HOOK for Impact Pack detailers (including FaceDetailer)."""

    def __init__(
        self,
        bad_model,
        *,
        w_autoguide: float = 2.0,
        swap_mode: str = AG_SWAP_MODE_SHARED_SAFE,
        ag_combine_mode: str = AG_COMBINE_MODE_SEQUENTIAL_DELTA,
        ag_delta_mode: str = AG_DELTA_MODE_BAD_CONDITIONAL,
        ag_max_ratio: float = AG_DEFAULT_MAX_RATIO,
        ag_allow_negative: bool = True,
        ag_ramp_mode: str = AG_RAMP_FLAT,
        ag_ramp_power: float = 2.0,
        ag_ramp_floor: float = 0.15,
        ag_post_cfg_mode: str = AG_POST_CFG_KEEP,
        safe_force_clean_swap: bool = True,
        uuid_only_noop: bool = False,
        debug_swap: bool = False,
        debug_metrics: bool = False,
        debug_metrics_all: bool = False,
    ):
        super().__init__()
        self.bad_model = bad_model
        self.w_autoguide = float(w_autoguide)
        self.swap_mode = str(swap_mode)
        self.ag_combine_mode = str(ag_combine_mode)
        self.ag_delta_mode = str(ag_delta_mode)
        self.ag_max_ratio = float(ag_max_ratio)
        self.ag_allow_negative = bool(ag_allow_negative)
        self.ag_ramp_mode = str(ag_ramp_mode)
        self.ag_ramp_power = float(ag_ramp_power)
        self.ag_ramp_floor = float(ag_ramp_floor)
        self.ag_post_cfg_mode = str(ag_post_cfg_mode)
        self.safe_force_clean_swap = bool(safe_force_clean_swap)
        self.uuid_only_noop = bool(uuid_only_noop)
        self.debug_swap = bool(debug_swap)
        self.debug_metrics = bool(debug_metrics)
        self.debug_metrics_all = bool(debug_metrics_all)

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        _ag_try_patch_comfy_samplers_sample()
        try:
            delattr(model, "_ag_detailer_guider")
        except Exception:
            pass

        guider = Guider_AutoGuidanceCFG(
            model,
            self.bad_model,
            swap_mode=self.swap_mode,
            ag_combine_mode=self.ag_combine_mode,
            ag_delta_mode=self.ag_delta_mode,
            ag_max_ratio=self.ag_max_ratio,
            ag_allow_negative=self.ag_allow_negative,
            ag_ramp_mode=self.ag_ramp_mode,
            ag_ramp_power=self.ag_ramp_power,
            ag_ramp_floor=self.ag_ramp_floor,
            ag_post_cfg_mode=self.ag_post_cfg_mode,
            safe_force_clean_swap=self.safe_force_clean_swap,
            uuid_only_noop=self.uuid_only_noop,
            debug_swap=self.debug_swap,
            debug_metrics=self.debug_metrics,
            debug_metrics_all=self.debug_metrics_all,
        )
        try:
            guider.set_conds(positive, negative)
        except Exception:
            pass
        try:
            guider.set_scales(cfg=float(cfg), w_ag=float(self.w_autoguide))
        except Exception:
            try:
                guider.cfg = float(cfg)
                guider.w_ag = float(self.w_autoguide)
            except Exception:
                pass

        try:
            setattr(model, "_ag_detailer_guider", guider)
        except Exception:
            pass

        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


class AutoGuidanceImpactDetailerHookProvider:
    """Node: creates a DETAILER_HOOK for Impact Pack FaceDetailer/Detailer nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bad_model": ("MODEL",),
                "w_autoguide": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.05, "round": 0.01}),
                "swap_mode": (AG_SWAP_MODE_CHOICES, {"default": AG_SWAP_MODE_SHARED_SAFE}),
            },
            "optional": {
                "ag_delta_mode": (AG_DELTA_MODE_CHOICES, {"default": AG_DELTA_MODE_BAD_CONDITIONAL}),
                "ag_combine_mode": (AG_COMBINE_MODE_CHOICES, {"default": AG_COMBINE_MODE_SEQUENTIAL_DELTA}),
                "ag_max_ratio": ("FLOAT", {"default": AG_DEFAULT_MAX_RATIO, "min": 0.0, "max": 5.0, "step": 0.05, "round": 0.01}),
                "ag_allow_negative": ("BOOLEAN", {"default": True}),
                "ag_ramp_mode": (AG_RAMP_MODE_CHOICES, {"default": AG_RAMP_FLAT}),
                "ag_ramp_power": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.5, "round": 0.1}),
                "ag_ramp_floor": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "ag_post_cfg_mode": (AG_POST_CFG_MODE_CHOICES, {"default": AG_POST_CFG_KEEP}),
                "safe_force_clean_swap": ("BOOLEAN", {"default": True}),
                "uuid_only_noop": ("BOOLEAN", {"default": False}),
                "debug_swap": ("BOOLEAN", {"default": False}),
                "debug_metrics": ("BOOLEAN", {"default": False}),
                "debug_metrics_all": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("DETAILER_HOOK",)
    FUNCTION = "get_hook"
    CATEGORY = "sampling/guiders"

    def get_hook(
        self,
        bad_model,
        w_autoguide: float,
        swap_mode=AG_SWAP_MODE_SHARED_SAFE,
        ag_combine_mode=AG_COMBINE_MODE_SEQUENTIAL_DELTA,
        ag_delta_mode=AG_DELTA_MODE_BAD_CONDITIONAL,
        ag_max_ratio=AG_DEFAULT_MAX_RATIO,
        ag_allow_negative=True,
        ag_ramp_mode=AG_RAMP_FLAT,
        ag_ramp_power=2.0,
        ag_ramp_floor=0.15,
        ag_post_cfg_mode=AG_POST_CFG_KEEP,
        safe_force_clean_swap=True,
        uuid_only_noop=False,
        debug_swap=False,
        debug_metrics=False,
        debug_metrics_all=False,
    ):
        hook = AutoGuidanceImpactDetailerHook(
            bad_model,
            w_autoguide=w_autoguide,
            swap_mode=swap_mode,
            ag_combine_mode=ag_combine_mode,
            ag_delta_mode=ag_delta_mode,
            ag_max_ratio=ag_max_ratio,
            ag_allow_negative=ag_allow_negative,
            ag_ramp_mode=ag_ramp_mode,
            ag_ramp_power=ag_ramp_power,
            ag_ramp_floor=ag_ramp_floor,
            ag_post_cfg_mode=ag_post_cfg_mode,
            safe_force_clean_swap=safe_force_clean_swap,
            uuid_only_noop=uuid_only_noop,
            debug_swap=debug_swap,
            debug_metrics=debug_metrics,
            debug_metrics_all=debug_metrics_all,
        )
        return (hook,)


NODE_CLASS_MAPPINGS = {
    "AutoGuidanceCFGGuider": AutoGuidanceCFGGuider,
    "AutoGuidanceImpactDetailerHookProvider": AutoGuidanceImpactDetailerHookProvider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoGuidanceCFGGuider": "AutoGuidance CFG Guider (good+bad)",
    "AutoGuidanceImpactDetailerHookProvider": "AutoGuidance Detailer Hook (Impact Pack)",
}
