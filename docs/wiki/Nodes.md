# Nodes

Complete reference for every node in the LoRA Optimizer suite.

---

## LoRA Stack

Builds a list of LoRAs for the optimizer. Chain multiple Stack nodes to add any number of LoRAs.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `lora_name` | COMBO | Yes | LoRA file selector (`.safetensors` from your loras folder) |
| `strength` | FLOAT | Yes | How much this LoRA contributes (default 1.0) |
| `conflict_mode` | COMBO | Yes | Where this LoRA's contributions apply: `all`, `low_conflict`, `high_conflict` |
| `key_filter` | COMBO | Yes | Which prefixes this LoRA contributes to: `all`, `shared_only`, `unique_only` |
| `lora_stack` | LORA_STACK | No | Previous stack to append to (for chaining) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `LORA_STACK` | LORA_STACK | The accumulated LoRA list |

---

## LoRA Stack (Dynamic)

Single node with 1–10 adjustable LoRA slots. Replaces chaining multiple Stack nodes.

### Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `lora_count` | 1–10 | Number of LoRA slots to show |
| `settings_visibility` | `simple`, `advanced` | Simple: one strength slider per LoRA. Advanced: separate model/clip strength, conflict mode, key filter |
| `input_mode` | `dropdown`, `text` | Dropdown: standard file picker. Text: type a name or path, auto-matched against installed LoRAs |
| `base_model_filter` | dynamic | Filters LoRA list by base model (requires [ComfyUI-Lora-Manager](https://github.com/hayden-fr/ComfyUI-Lora-Manager)) |

### Per-LoRA Inputs (Simple Mode)

| Input | Type | Description |
|-------|------|-------------|
| `lora_name_i` | COMBO/STRING | LoRA file |
| `strength_i` | FLOAT | Strength (applies to both model and CLIP) |

### Per-LoRA Inputs (Advanced Mode)

| Input | Type | Description |
|-------|------|-------------|
| `lora_name_i` | COMBO/STRING | LoRA file |
| `model_strength_i` | FLOAT | Model patch strength |
| `clip_strength_i` | FLOAT | CLIP patch strength |
| `conflict_mode_i` | COMBO | `all`, `low_conflict`, `high_conflict` |
| `key_filter_i` | COMBO | `all`, `shared_only`, `unique_only` |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `LORA_STACK` | LORA_STACK | The accumulated LoRA list |

Optional `lora_stack` input accepts a previous stack for chaining with other Stack nodes.

---

## LoRA Optimizer

The simplified auto-optimizer. Sensible defaults, minimal controls. Auto-strength is enabled by default.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | — | Base model from checkpoint loader |
| `lora_stack` | LORA_STACK | Yes | — | Stack of LoRAs to merge |
| `output_strength` | FLOAT | Yes | 1.0 | Master strength multiplier (0–10) |
| `clip` | CLIP | No | — | Text encoder (if using CLIP LoRA keys) |
| `clip_strength_multiplier` | FLOAT | Yes | 1.0 | CLIP strength relative to model strength (0–10) |

Uses fixed defaults internally: `auto_strength=enabled`, `optimization_mode=per_prefix`, `merge_quality=standard`, `compress_patches=non_ties`, `vram_budget=0.0`.

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MODEL` | MODEL | Patched model ready for sampling |
| `CLIP` | CLIP | Patched text encoder |
| `report` | STRING | Analysis report with block strategy map |
| `LORA_DATA` | LORA_DATA | Merged patches for downstream nodes |

---

## LoRA Optimizer (Advanced)

Full-featured optimizer with all parameters exposed.

### Inputs

All inputs from the simple variant, plus:

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_strength` | COMBO | disabled | `enabled` / `disabled` — interference-aware energy normalization |
| `optimization_mode` | COMBO | per_prefix | `per_prefix`, `global`, `weighted_sum_only` |
| `merge_quality` | COMBO | standard | `standard`, `enhanced`, `maximum` |
| `sparsification` | COMBO | disabled | `disabled`, `dare`, `della`, `dare_conflict`, `della_conflict` |
| `sparsification_density` | FLOAT | 0.7 | Fraction of parameters to keep (0.01–1.0) |
| `dare_dampening` | FLOAT | 0.0 | DAREx noise reduction (0–1.0, only affects DARE modes) |
| `compress_patches` | COMBO | non_ties | `non_ties`, `all`, `disabled` |
| `svd_device` | COMBO | gpu | `gpu`, `cpu` — device for SVD compression |
| `cache_patches` | COMBO | enabled | `enabled`, `disabled` — keep merge in RAM for re-execution |
| `free_vram_between_passes` | COMBO | disabled | `enabled`, `disabled` — release GPU cache between passes |
| `normalize_keys` | COMBO | disabled | `enabled`, `disabled` — architecture-aware key normalization |
| `behavior_profile` | COMBO | v1.2 | `v1.2`, `no_slerp`, `classic` — strategy selection logic |
| `architecture_preset` | COMBO | auto | `auto`, `sd_unet`, `dit`, `llm` — numeric threshold tuning |
| `vram_budget` | FLOAT | 0.0 | Fraction of free VRAM for keeping patches on GPU (0.0–1.0) |

### Optional Inputs

| Input | Type | Description |
|-------|------|-------------|
| `merge_strategy_override` | STRING | Force a specific merge strategy (connect from Conflict Editor) |

### Outputs

Same as the simple variant: `MODEL`, `CLIP`, `report` (STRING), `LORA_DATA`.

---

## LoRA AutoTuner

Automated parameter sweep that finds the optimal merge configuration.

### Inputs

All inputs from the Advanced optimizer, plus:

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `top_n` | INT | 3 | Number of candidates to actually merge and score (1–10) |
| `scoring_svd` | COMBO | disabled | `enabled`, `disabled` — SVD-based effective rank scoring |
| `scoring_device` | COMBO | cpu | `cpu`, `gpu` — device for SVD scoring (GPU is 10–50x faster) |
| `record_dataset` | COMBO | disabled | `enabled`, `disabled` — save metrics to JSONL for research |
| `vram_budget` | FLOAT | 0.0 | Fraction of free VRAM for patch placement |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MODEL` | MODEL | Patched model from the best-scoring config |
| `CLIP` | CLIP | Patched text encoder |
| `report` | STRING | Ranked results with heuristic and measured scores |
| `TUNER_DATA` | TUNER_DATA | All ranked configs for Merge Selector |
| `LORA_DATA` | LORA_DATA | Merged patches from the best config |

Marked as `OUTPUT_NODE` for ComfyUI's re-execution optimization.

See [[How It Works#appendix-autotuner]] for details on the two-phase architecture and scoring.

---

## Merge Selector

Applies a specific ranked configuration from AutoTuner results.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | — | Base model |
| `tuner_data` | TUNER_DATA | Yes | — | Output from LoRA AutoTuner |
| `selection` | INT | Yes | 1 | Which ranked config to apply (1 = best, 2 = second best, etc.) |
| `clip` | CLIP | No | — | Text encoder |

Validates that the LoRA stack hasn't changed since the AutoTuner ran (via hash comparison).

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MODEL` | MODEL | Patched model from the selected config |
| `CLIP` | CLIP | Patched text encoder |
| `report` | STRING | Analysis report for the selected config |
| `LORA_DATA` | LORA_DATA | Merged patches from the selected config |

Marked as `OUTPUT_NODE`.

---

## LoRA Conflict Editor

Interactive conflict analysis with per-LoRA override controls.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | MODEL | Yes | Base model |
| `lora_stack` | LORA_STACK | Yes | Stack to analyze |
| `merge_strategy` | COMBO | Yes | `auto`, `ties`, `consensus`, `slerp`, `weighted_average`, `weighted_sum` |

Per-LoRA conflict mode overrides appear dynamically based on the stack.

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `LORA_STACK` | LORA_STACK | Enriched stack with conflict mode settings |
| `analysis_report` | STRING | Pairwise conflict analysis |
| `merge_strategy` | STRING | Strategy override string (connect to optimizer's `merge_strategy_override`) |

### How It Works

1. Loads all LoRAs, computes full-rank diffs per prefix
2. Measures pairwise sign conflict ratios
3. Auto-suggests per-LoRA conflict modes:
   - < 15% average conflict: `all`
   - 15–40%: `low_conflict`
   - \> 40%: `high_conflict`
4. Allows manual override of each LoRA's conflict mode and the merge strategy
5. Outputs feed directly into the optimizer

---

## Save Merged LoRA

Exports merged patches as a standalone `.safetensors` file usable with any standard LoRA loader.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `lora_data` | LORA_DATA | Yes | — | From Optimizer, AutoTuner, or Merge Selector |
| `filename` | STRING | Yes | `merged_lora` | Plain name → saves to ComfyUI loras folder. Absolute path → saves there |
| `save_rank` | INT | Yes | 0 | 0 = use existing layer ranks. Non-zero = force this rank via SVD |
| `bake_strength` | COMBO | Yes | enabled | Bake `output_strength` so the saved LoRA works at strength 1.0 |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `filepath` | STRING | Full path to the saved `.safetensors` file |

---

## Merged LoRA to Hook

Wraps merged patches as conditioning hooks for per-prompt LoRA application.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `lora_data` | LORA_DATA | Yes | From Optimizer, AutoTuner, or Merge Selector |
| `prev_hooks` | HOOKS | No | Chain with existing hooks |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `HOOKS` | HOOKS | Conditioning hooks for use with Cond Set Props |

### Use Cases

- **Per-prompt LoRA** — different merged LoRAs on positive vs negative conditioning
- **Scheduled application** — combine with hook keyframes for step-specific LoRA
- **Regional conditioning** — apply LoRA to specific image regions
- **Preserve base model** — keep MODEL unpatched, apply LoRA only through conditioning

---

## WanVideo LoRA Optimizer (WIP)

Optimizer variant for WanVideo models via [kijai's WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper).

### Differences from Standard Optimizer

| Aspect | Standard | WanVideo |
|--------|----------|----------|
| Model input | `MODEL` | `WANVIDEOMODEL` |
| CLIP | Supported | Not used |
| `normalize_keys` default | disabled | **enabled** |
| `cache_patches` default | enabled | **disabled** |
| `architecture_preset` default | auto | **dit** |

All merge algorithms work identically — TIES, DARE/DELLA, SVD compression, auto-strength, quality enhancements, key normalization.

### Inputs/Outputs

Same as the Advanced optimizer, but with `WANVIDEOMODEL` replacing `MODEL` and no CLIP inputs/outputs. Outputs `LORA_DATA` for Save Merged LoRA.

---

## Merged LoRA → WanVideo (WIP)

Bridges `LORA_DATA` to a WanVideo wrapper model.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `wan_model` | WANVIDEOMODEL | Yes | WanVideo model to patch |
| `lora_data` | LORA_DATA | No | Merged patches to apply |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `WANVIDEOMODEL` | WANVIDEOMODEL | Patched WanVideo model |

Handles the `_orig_mod.` key prefix mismatch from `torch.compile` automatically.
