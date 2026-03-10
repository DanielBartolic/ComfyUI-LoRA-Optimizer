# Merge Formula — Design Document

## Problem

When merging 3+ LoRAs, all are merged simultaneously in a flat pass. This means 2 character LoRAs outvote 1 style LoRA in majority-based methods (TIES) and dilute it to 1/3 in averaging methods. Users have no control over merge topology.

## Solution

A **Merge Formula** node that lets users define hierarchical merge order via a text expression. For example, `(1+2) + 3` merges characters first, then blends the result with the style LoRA — giving style 50% influence instead of 33%.

## Node Design

**`LoRAMergeFormula`** — passthrough node:
- Input: `LORA_STACK` + `STRING` (formula)
- Output: `LORA_STACK` (same stack, with formula metadata attached)
- Zero computation — tags the stack with the formula string
- Validates syntax and index range, shows errors in ComfyUI UI

**Metadata transport:** Appends `{"_merge_formula": "(1+2) + 3"}` to the stack list. The optimizer pops this entry before normalizing.

## Formula Syntax

```
(1+2) + 3             — merge LoRAs 1&2 first, then merge result with 3
(1+2):0.6 + 3:0.4     — same, with explicit group blend weights
(1+2) + (3+4) + 5     — two sub-groups then merged with 5
1 + 2 + 3              — flat merge (identical to no formula)
```

- Numbers: 1-indexed position in the LoRA stack
- `+`: combines items at the same tree level
- `()`: defines a sub-merge group
- `:weight`: optional blend weight override for a group or item in the parent merge
- Without `:weight`, items use their original stack weights
- Whitespace ignored

## Execution Model

1. **Parse** formula into a merge tree (recursive descent parser)
2. **Execute leaf-to-root** — each tree node triggers a full `optimize_merge` (two-pass analysis, auto-strength, per-prefix strategy)
3. **Inner results become "virtual LoRAs"** — merged diff patches wrapped as `{"name": "(1+2)", "lora": {key: patch}, "strength": 1.0}`, entering the parent merge at strength 1.0
4. **No formula / flat expression** — identical to current behavior (backward compatible)

### Example: `(1+2):0.6 + 3:0.4` with [char_A, char_B, style]

```
Step 1: optimize_merge([char_A, char_B]) → merged_characters
Step 2: optimize_merge([merged_characters:0.6, style:0.4]) → final
```

### Cost

K groups = K sub-merges + 1 final merge. `(1+2) + 3` is ~2x time vs flat merge. Acceptable since user explicitly opts in.

## Implementation Scope

### New code
- `_parse_merge_formula(formula_str, n_loras)` — recursive descent parser, returns tree, validates indices
- `_execute_merge_tree(tree, lora_stack, ...)` — walks tree leaf-to-root, calls `optimize_merge` per node, wraps results as virtual LoRAs
- `LoRAMergeFormula` node class — INPUT_TYPES, RETURN_TYPES, registered in NODE_CLASS_MAPPINGS

### Modified code
- `optimize_merge` — at top, check for formula metadata; if present, delegate to `_execute_merge_tree`; otherwise unchanged
- `_normalize_stack` — handle virtual LoRA dicts (already-merged patches, no file to load)

### Not modified
- `_merge_diffs`, `_tall_masks`, `_knots_align`, AutoTuner, scoring — untouched. Each sub-merge is a standard `optimize_merge` call.

## Edge Cases

| Case | Behavior |
|------|----------|
| No formula / empty string | Flat merge (current behavior) |
| Out-of-range index | Validation error in node UI |
| Formula doesn't use all LoRAs | Allowed — unused LoRAs excluded |
| Single LoRA in group `(1) + 2` | Inner merge is no-op passthrough |
| Duplicate index `(1+1) + 2` | Allowed |
| Malformed formula | Parse error in node UI |
| Group weight 0 | Group effectively dropped |
| CLIP patches | Inner merges produce both model and CLIP patches; outer merge receives both |
