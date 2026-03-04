# Design: WanVideoLoRAOptimizer Node

## Problem

The WanVideoWrapper custom node has its own LoRA pipeline using `WANVIDEOMODEL` and `WANVIDLORA` types, separate from ComfyUI's standard `MODEL`/`CLIP` types. Users want to merge multiple WanVideo LoRAs using our optimizer's algorithms (TIES, DARE, auto-strength, SVD compression) but can't connect `WANVIDEOMODEL` to our `MODEL` input.

## Solution

A thin subclass of `LoRAOptimizer` that bridges the type gap. All merging logic is inherited — the subclass only overrides input/output types and wraps the return tuple.

### Why this works

1. **Merging algorithms are architecture-agnostic** — TIES, DARE/DELLA, SVD compression, auto-strength, and conflict detection all operate on generic tensors.
2. **Wan key normalization already exists** — `_detect_architecture` returns `'wan'`, and `_normalize_keys_wan` handles LyCORIS, diffusers, Fun LoRA, finetrainer, and RS-LoRA formats.
3. **Generic key mapping works** — `comfy.lora.model_lora_keys_unet()` creates `key_map[k[:-len(".weight")]] = k` for all model architectures, which matches standardized WanVideo LoRA key prefixes.
4. **CLIP is already optional** — the parent class handles `clip=None` gracefully.
5. **Patch format is compatible** — both `("diff", (tensor,))` tuples and `LoRAAdapter` objects work with WanVideoWrapper's `set_lora_params()` (which checks `hasattr(lora_obj, "weights")` and `tuple[0] == "diff"`).

## Implementation

### New class: `WanVideoLoRAOptimizer` (~30 lines)

Inherits from `LoRAOptimizer`. Overrides:

- **`INPUT_TYPES`**: Copies parent's inputs, replaces `MODEL` → `WANVIDEOMODEL`, removes `clip`, `clip_strength_multiplier`, `free_vram_between_passes`. Changes defaults: `normalize_keys="enabled"`, `cache_patches="disabled"`.
- **`RETURN_TYPES`**: `("WANVIDEOMODEL", "STRING", "LORA_DATA")` — drops CLIP output.
- **`optimize_merge()`**: Calls `super().optimize_merge(model, lora_stack, ..., clip=None, clip_strength_multiplier=0)`, remaps return tuple to skip CLIP.
- **`CATEGORY`**: `"loaders/lora"` (same as parent).
- **`DESCRIPTION`**: Updated for WanVideo context.

### Inputs

| Input | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| model | WANVIDEOMODEL | Yes | — | From WanVideoModelLoader |
| lora_stack | LORA_STACK | Yes | — | Same LoRA Stack nodes |
| output_strength | FLOAT | Yes | 1.0 | Master strength |
| auto_strength | enum | No | disabled | Same |
| optimization_mode | enum | No | per_prefix | Same |
| cache_patches | enum | No | **disabled** | Changed: video models are large |
| compress_patches | enum | No | non_ties | Same |
| svd_device | enum | No | gpu | Same |
| normalize_keys | enum | No | **enabled** | Changed: WanVideo LoRAs vary |
| sparsification | enum | No | disabled | Same |
| sparsification_density | FLOAT | No | 0.7 | Same |
| merge_strategy_override | STRING | No | "" | Same |

### Outputs

| Output | Type |
|--------|------|
| model | WANVIDEOMODEL |
| analysis_report | STRING |
| lora_data | LORA_DATA |

### Registration

```python
NODE_CLASS_MAPPINGS["WanVideoLoRAOptimizer"] = WanVideoLoRAOptimizer
NODE_DISPLAY_NAME_MAPPINGS["WanVideoLoRAOptimizer"] = "WanVideo LoRA Optimizer"
```

### Workflow

```
[WanVideoModelLoader] → WANVIDEOMODEL → [WanVideoLoRAOptimizer] → WANVIDEOMODEL → [WanVideoSampler]
                                                  ↑
                        [LoRA Stack] ─────────────┘
```

Chaining with individual (non-merged) LoRAs:
```
[WanVideoLoraSelect] → WANVIDLORA → [WanVideoModelLoader] → WANVIDEOMODEL → [WanVideoLoRAOptimizer] → WANVIDEOMODEL → [Sampler]
                                                                                      ↑
                                                          [LoRA Stack] ───────────────┘
```

## Files to modify

- `lora_optimizer.py` — add `WanVideoLoRAOptimizer` class (~30 lines), register in mappings

## Verification

1. Load WanVideo model (no LoRA) → optimizer with 2+ WanVideo LoRAs → sampler. Verify LoRA effect appears in generated video.
2. Empty LoRA stack → optimizer returns unmodified WANVIDEOMODEL, no crash.
3. Mixed trainer formats (e.g., one LyCORIS + one diffusers LoRA) with normalize_keys=enabled → both processed correctly.
4. LORA_DATA output → SaveMergedLoRA → loadable via WanVideoLoraSelect.
5. Chain: individual LoRAs via model loader + merged LoRAs via optimizer — both effects visible.
