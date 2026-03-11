# AutoTuner Per Sub-Merge — Design Document

## Problem

When using Merge Formula with AutoTuner (e.g. `(1+2)+3`), sub-merges inherit the parent's fixed settings. This is suboptimal — a character pair might merge best with SLERP while the outer characters-vs-style merge might need TIES. The sub-merge environment is fundamentally different from the outer merge.

## Solution

Each sub-group in a merge formula gets its own full AutoTuner sweep (Phase 1 heuristic + Phase 2 merge & score). Sub-merges are resolved first, then the outer merge runs its own AutoTuner sweep on the resulting virtual LoRAs.

## Flow

```
auto_tune() with formula (1+2)+3:

1. Detect & extract formula from lora_stack
2. Parse tree
3. For each sub-group with 2+ items (e.g. LoRAs 1+2):
   → Run auto_tune(): Phase 1 heuristic + Phase 2 top_n candidates
   → Winner becomes virtual LoRA (via _model_to_virtual_lora)
4. Build final flat stack: [virtual_chars, style_lora]
5. Run outer auto_tune() on that flat stack
6. Return best outer result
```

**Key property:** Sub-merges resolve before the outer sweep. The outer sweep sees virtual LoRAs as fixed inputs — no re-running sub-merges per outer candidate.

## Cost

K sub-groups × top_n sub-merges + top_n outer merges.
Example: `(1+2)+3` with top_n=3 → 3 sub-merges + 3 outer merges = 6 total (vs 3 flat merges without formula).

## Implementation

### New method: `_autotune_resolve_tree()`

Similar to `_resolve_tree_to_stack` but calls `self.auto_tune()` for sub-groups instead of `self.optimize_merge()`.

Input: tree, normalized_stack, model, clip, inherited settings
Output: (resolved_stack, sub_reports)

### Modified: `auto_tune()`

At top, after normalizing stack: detect formula. If present, call `_autotune_resolve_tree()` to get a flat stack of resolved virtual LoRAs + leaves, then continue with normal AutoTuner sweep on that stack.

### Sub-merge auto_tune() settings

Inherited from parent:
- top_n, scoring_speed, scoring_formula, scoring_svd, scoring_device
- architecture_preset (resolved), normalize_keys, auto_strength_floor
- decision_smoothing, smooth_slerp_gate, vram_budget

Overridden for sub-merges:
- diff_cache_mode="disabled" (small stacks, not worth caching)
- cache_patches="disabled" (consumed immediately)
- output_mode="merge" (always need the merged model)
- record_dataset="disabled" (don't pollute parent's dataset)

### Not modified

- `_execute_merge_tree`, `_resolve_tree_to_stack` — untouched, still used by Optimizer node
- `_model_to_virtual_lora` — reused as-is
- All existing AutoTuner scoring/grid logic — untouched

## Edge Cases

| Case | Behavior |
|------|----------|
| No formula | Normal AutoTuner (unchanged) |
| Single LoRA in sub-group | Pass through, no AutoTuner needed |
| Sub-merge auto_tune returns no lora_data | Fall back to passing items through flat |
| Nested groups `((1+2)+3)+4` | Recursive: innermost resolved first |
| Optimizer node (not AutoTuner) + formula | Uses existing `_resolve_tree_to_stack` with parent settings (unchanged) |
