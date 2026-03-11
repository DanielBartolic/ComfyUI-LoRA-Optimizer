# AutoTuner Per Sub-Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When AutoTuner encounters a merge formula, each sub-group gets its own full AutoTuner sweep before the outer merge runs its sweep.

**Architecture:** Add `_autotune_resolve_tree()` method to `LoRAAutoTuner` that walks the formula tree and calls `self.auto_tune()` recursively for each sub-group. Insert formula detection at the top of `auto_tune()` — if present, resolve sub-merges first, then run the normal AutoTuner sweep on the resulting flat stack of virtual LoRAs.

**Tech Stack:** Python, PyTorch, ComfyUI framework. All changes in `lora_optimizer.py`.

---

### Task 1: Add `_autotune_resolve_tree` method

**Files:**
- Modify: `lora_optimizer.py:7001` (inside `LoRAAutoTuner` class, after `auto_tune` method ~line 7730)
- Test: `tests/test_lora_optimizer.py`

**Step 1: Write the failing test**

Add to `tests/test_lora_optimizer.py` in the formula tests section (after line ~799):

```python
def test_autotune_resolve_tree_calls_auto_tune_for_subgroups(self):
    """_autotune_resolve_tree should call auto_tune for sub-groups with 2+ items."""
    from lora_optimizer import LoRAAutoTuner, _parse_merge_formula

    tuner = LoRAAutoTuner()
    tree = _parse_merge_formula("(1+2)+3", 3)

    # Build a minimal normalized stack with 3 fake LoRAs
    fake_lora_a = {"key_a": torch.randn(4, 4)}
    fake_lora_b = {"key_a": torch.randn(4, 4)}
    fake_lora_c = {"key_c": torch.randn(4, 4)}
    normalized_stack = [
        {"name": "lora_a", "lora": fake_lora_a, "strength": 1.0,
         "clip_strength": None, "metadata": {}},
        {"name": "lora_b", "lora": fake_lora_b, "strength": 1.0,
         "clip_strength": None, "metadata": {}},
        {"name": "lora_c", "lora": fake_lora_c, "strength": 1.0,
         "clip_strength": None, "metadata": {}},
    ]

    # Track auto_tune calls
    calls = []

    def mock_auto_tune(model, lora_stack, output_strength, **kwargs):
        calls.append({"n_loras": len(lora_stack), "names": [l["name"] for l in lora_stack]})
        # Return a minimal 6-tuple with virtual LoRA patches
        virtual_patches = {"key_a": ("diff", (torch.randn(4, 4),))}
        lora_data = {"model_patches": virtual_patches, "clip_patches": {}}
        return (model, None, "sub-report", "", None, lora_data)

    tuner.auto_tune = mock_auto_tune

    at_kwargs = {
        "clip_strength_multiplier": 1.0,
        "top_n": 3,
        "normalize_keys": "disabled",
        "scoring_svd": "disabled",
        "scoring_device": "cpu",
        "architecture_preset": "dit",
        "auto_strength_floor": -1.0,
        "decision_smoothing": 0.25,
        "smooth_slerp_gate": False,
        "vram_budget": 0.0,
        "scoring_speed": "turbo",
        "scoring_formula": "v2",
        "diff_cache_mode": "disabled",
        "diff_cache_ram_pct": 0.5,
    }

    resolved_stack, sub_reports = tuner._autotune_resolve_tree(
        tree, normalized_stack, None, None, **at_kwargs)

    # Should have called auto_tune once for the (1+2) sub-group
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0]["n_loras"], 2)
    self.assertEqual(calls[0]["names"], ["lora_a", "lora_b"])

    # Resolved stack should have 2 items: virtual LoRA + lora_c
    self.assertEqual(len(resolved_stack), 2)
    self.assertTrue(resolved_stack[0].get("_precomputed_diffs"))  # virtual
    self.assertEqual(resolved_stack[1]["name"], "lora_c")
    self.assertEqual(len(sub_reports), 1)
```

**Step 2: Run test to verify it fails**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRAOptimizer::test_autotune_resolve_tree_calls_auto_tune_for_subgroups -v`
Expected: FAIL with `AttributeError: 'LoRAAutoTuner' object has no attribute '_autotune_resolve_tree'`

**Step 3: Write minimal implementation**

Add the following method to `LoRAAutoTuner` class in `lora_optimizer.py`, after the `auto_tune` method (after line ~7730, before `_save_tuner_dataset_entry`):

```python
    def _autotune_resolve_tree(self, tree, normalized_stack, model, clip, **at_kwargs):
        """
        Recursively resolve a merge formula tree, running auto_tune for each
        sub-group with 2+ items.  Returns (resolved_stack, sub_reports).
        Same tree format as _resolve_tree_to_stack but uses AutoTuner for sub-merges.
        """
        sub_reports = []

        if tree["type"] == "leaf":
            item = dict(normalized_stack[tree["index"]])
            if "metadata" in item and isinstance(item["metadata"], dict):
                item["metadata"] = dict(item["metadata"])
            if tree["weight"] is not None:
                item["strength"] = tree["weight"]
            return ([item], sub_reports)

        # Group: resolve each child
        resolved = []
        for child in tree["children"]:
            if child["type"] == "leaf":
                item = dict(normalized_stack[child["index"]])
                if "metadata" in item and isinstance(item["metadata"], dict):
                    item["metadata"] = dict(item["metadata"])
                if child["weight"] is not None:
                    item["strength"] = child["weight"]
                resolved.append(item)
            else:
                # Sub-group: resolve recursively then auto-tune
                sub_stack, child_reports = self._autotune_resolve_tree(
                    child, normalized_stack, model, clip, **at_kwargs)
                sub_reports.extend(child_reports)

                if len(sub_stack) >= 2:
                    try:
                        # Override settings for sub-merge
                        sub_kwargs = dict(at_kwargs)
                        sub_kwargs["cache_patches"] = "disabled"
                        sub_kwargs["record_dataset"] = "disabled"
                        sub_kwargs["output_mode"] = "merge"
                        sub_kwargs["_is_sub_merge"] = True
                        # Evaluator is excluded: it may be prompt-specific and
                        # inappropriate for sub-groups (character-only merge etc.)

                        sub_result = self.auto_tune(
                            model, sub_stack, 1.0, clip=clip, **sub_kwargs)

                        # auto_tune returns 6-tuple
                        sub_model, sub_clip, sub_report, _, _, sub_lora_data = sub_result

                        sub_reports.append(sub_report)

                        if sub_lora_data is None:
                            # Fallback: pass items through
                            for sub_item in sub_stack:
                                item = dict(sub_item)
                                if child.get("weight") is not None:
                                    item["strength"] = child["weight"]
                                resolved.append(item)
                            continue

                        # Build virtual LoRA from sub-merge result
                        sub_model_patches = sub_lora_data.get("model_patches", {})
                        sub_clip_patches = sub_lora_data.get("clip_patches", {})
                        virtual = self._model_to_virtual_lora(
                            sub_model_patches, sub_clip_patches, child)
                        del sub_model, sub_clip, sub_result, sub_lora_data
                        if child["weight"] is not None:
                            virtual["strength"] = child["weight"]
                        resolved.append(virtual)
                    except Exception as e:
                        logging.warning(
                            f"[LoRA AutoTuner] Sub-merge auto_tune failed: {e} — "
                            "falling back to flat merge for this sub-group")
                        for item in sub_stack:
                            resolved.append(item)
                elif len(sub_stack) == 1:
                    item = sub_stack[0]
                    if child["weight"] is not None:
                        item["strength"] = child["weight"]
                    resolved.append(item)

        return (resolved, sub_reports)
```

**Step 4: Run test to verify it passes**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRAOptimizer::test_autotune_resolve_tree_calls_auto_tune_for_subgroups -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add _autotune_resolve_tree for per-sub-merge AutoTuner"
```

---

### Task 2: Integrate formula detection into `auto_tune()`

**Files:**
- Modify: `lora_optimizer.py:7118-7161` (inside `auto_tune`, signature + after stack normalization)
- Test: `tests/test_lora_optimizer.py`

**Review fixes incorporated:**
- Bug 1: Pass `normalize_keys="disabled"` to recursive outer call (stack already normalized)
- Bug 2: Include `clip_strength_multiplier` in `at_kwargs`
- Bug 4: Update `normalized_stack` and `active_loras` when formula resolves to single item
- Bug 6: Add `_suppress_pbar=False` parameter to suppress sub-merge progress bars
- Bug 8: Save/restore `_detected_arch` around recursive outer call
- Bug 9: Pass `_skip_qkv_refusion` to single-LoRA optimize_merge path

**Step 1: Write the failing test**

```python
def test_auto_tune_with_formula_calls_autotune_resolve_tree(self):
    """auto_tune should detect formula and use _autotune_resolve_tree."""
    from lora_optimizer import LoRAAutoTuner

    tuner = LoRAAutoTuner()

    # Build a lora_stack with formula metadata
    fake_lora_a = {"key_a": torch.randn(4, 4)}
    fake_lora_b = {"key_a": torch.randn(4, 4)}
    fake_lora_c = {"key_c": torch.randn(4, 4)}
    lora_stack = [
        {"name": "lora_a", "lora": fake_lora_a, "strength": 1.0},
        {"name": "lora_b", "lora": fake_lora_b, "strength": 1.0},
        {"name": "lora_c", "lora": fake_lora_c, "strength": 1.0},
        {"_merge_formula": "(1+2)+3"},
    ]

    # Track _autotune_resolve_tree calls
    resolve_calls = []

    def mock_resolve(tree, normalized_stack, model, clip, **kwargs):
        resolve_calls.append(tree)
        # Return a flat stack (2 items) so auto_tune continues normally
        return ([normalized_stack[0], normalized_stack[2]], [])

    tuner._autotune_resolve_tree = mock_resolve

    # Mock the rest of auto_tune to avoid needing real models
    # We just need to verify _autotune_resolve_tree was called
    try:
        tuner.auto_tune(None, lora_stack, 1.0)
    except Exception:
        pass  # Will fail later in pipeline — we only check the call happened

    self.assertEqual(len(resolve_calls), 1)
    self.assertEqual(resolve_calls[0]["type"], "group")
```

**Step 2: Run test to verify it fails**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRAOptimizer::test_auto_tune_with_formula_calls_autotune_resolve_tree -v`
Expected: FAIL — `_autotune_resolve_tree` never called (formula not detected in `auto_tune`)

**Step 3: Write minimal implementation**

**3a. Update `auto_tune` method signature** (line ~7118). Add `_is_sub_merge` and `_suppress_pbar`:

Change:
```python
    def auto_tune(self, model, lora_stack, output_strength, clip=None,
                  clip_strength_multiplier=1.0, top_n=3, normalize_keys="disabled",
                  scoring_svd="disabled", scoring_device="gpu",
                  architecture_preset="auto", auto_strength_floor=-1.0, evaluator=None,
                  record_dataset="disabled",
                  cache_patches="enabled",
                  diff_cache_mode="disabled", diff_cache_ram_pct=0.5, vram_budget=0.0,
                  scoring_speed="full", scoring_formula="v2", output_mode="merge",
                  decision_smoothing=0.25, smooth_slerp_gate=False):
```

To:
```python
    def auto_tune(self, model, lora_stack, output_strength, clip=None,
                  clip_strength_multiplier=1.0, top_n=3, normalize_keys="disabled",
                  scoring_svd="disabled", scoring_device="gpu",
                  architecture_preset="auto", auto_strength_floor=-1.0, evaluator=None,
                  record_dataset="disabled",
                  cache_patches="enabled",
                  diff_cache_mode="disabled", diff_cache_ram_pct=0.5, vram_budget=0.0,
                  scoring_speed="full", scoring_formula="v2", output_mode="merge",
                  decision_smoothing=0.25, smooth_slerp_gate=False,
                  _is_sub_merge=False, _suppress_pbar=False):
```

**3b. Insert formula detection** after stack normalization (line ~7140). Replace the block from `# --- Normalize & validate stack ---` through `if len(active_loras) == 1:`:

```python
        # --- Extract merge formula before normalization ---
        merge_formula = None
        clean_stack = []
        for item in lora_stack:
            if isinstance(item, dict) and "_merge_formula" in item:
                merge_formula = item["_merge_formula"]
            else:
                clean_stack.append(item)
        if merge_formula:
            lora_stack = clean_stack

        # --- Normalize & validate stack ---
        normalized_stack = self._normalize_stack(lora_stack, normalize_keys=normalize_keys)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]
        if not active_loras:
            return (model, clip, "No active LoRAs in stack.", "", None, None)

        # --- Formula-based hierarchical auto-tune ---
        if merge_formula and len(active_loras) >= 2:
            try:
                tree = _parse_merge_formula(merge_formula, len(normalized_stack))
            except ValueError as e:
                logging.warning(f"[LoRA AutoTuner] Invalid merge formula: {e} — using flat auto-tune")
                tree = None

            if tree is not None and tree["type"] == "group":
                logging.info(f"[LoRA AutoTuner] Using merge formula: {merge_formula}")

                # Resolve architecture preset before sub-merges
                preset_key, _ = _resolve_arch_preset(
                    architecture_preset, getattr(self, '_detected_arch', None) or 'unknown')

                at_kwargs = {
                    "clip_strength_multiplier": clip_strength_multiplier,
                    "top_n": top_n,
                    "normalize_keys": normalize_keys,
                    "scoring_svd": scoring_svd,
                    "scoring_device": scoring_device,
                    "architecture_preset": preset_key,
                    "auto_strength_floor": auto_strength_floor,
                    "decision_smoothing": decision_smoothing,
                    "smooth_slerp_gate": smooth_slerp_gate,
                    "vram_budget": vram_budget,
                    "scoring_speed": scoring_speed,
                    "scoring_formula": scoring_formula,
                    "diff_cache_mode": diff_cache_mode,
                    "diff_cache_ram_pct": diff_cache_ram_pct,
                }

                resolved_stack, sub_reports = self._autotune_resolve_tree(
                    tree, normalized_stack, model, clip, **at_kwargs)

                if len(resolved_stack) >= 2:
                    # Save _detected_arch — recursive call may overwrite it
                    # when all resolved items are virtual (no arch detection possible)
                    saved_arch = getattr(self, '_detected_arch', None)

                    # Run outer auto_tune on the resolved flat stack (no formula).
                    # normalize_keys="disabled": stack is already normalized.
                    outer_result = self.auto_tune(
                        model, resolved_stack, output_strength,
                        clip=clip,
                        clip_strength_multiplier=clip_strength_multiplier,
                        top_n=top_n,
                        normalize_keys="disabled",
                        scoring_svd=scoring_svd,
                        scoring_device=scoring_device,
                        architecture_preset=preset_key,
                        auto_strength_floor=auto_strength_floor,
                        evaluator=evaluator,
                        record_dataset=record_dataset,
                        cache_patches=cache_patches,
                        diff_cache_mode=diff_cache_mode,
                        diff_cache_ram_pct=diff_cache_ram_pct,
                        vram_budget=vram_budget,
                        scoring_speed=scoring_speed,
                        scoring_formula=scoring_formula,
                        output_mode=output_mode,
                        decision_smoothing=decision_smoothing,
                        smooth_slerp_gate=smooth_slerp_gate,
                    )

                    # Restore _detected_arch
                    self._detected_arch = saved_arch

                    # Prepend sub-reports to the outer report
                    if sub_reports:
                        # outer_result is 6-tuple
                        ret_model, ret_clip, report, analysis_report, tuner_data, lora_data = outer_result
                        separator = "\n" + "=" * 50 + "\n"
                        sub_section = separator.join(sub_reports)
                        report = (
                            "AUTOTUNER FORMULA SUB-MERGE REPORTS\n"
                            + separator + sub_section + separator
                            + "\nFINAL AUTOTUNER REPORT:\n" + report
                        )
                        outer_result = (ret_model, ret_clip, report, analysis_report, tuner_data, lora_data)

                    return outer_result
                elif len(resolved_stack) == 1:
                    # All sub-merges collapsed to one — update state and fall through
                    logging.info("[LoRA AutoTuner] Formula resolved to single LoRA — skipping outer tune")
                    normalized_stack = resolved_stack
                    active_loras = [item for item in normalized_stack if item["strength"] != 0]
                    # Fall through to single-LoRA or normal path below

        if len(active_loras) == 1:
```

**3c. Update progress bar creation** (line ~7204). Replace:

```python
        pbar = comfy.utils.ProgressBar(len(target_groups) + n_pbar_merges)
```

With:

```python
        if _suppress_pbar:
            class _NullPbar:
                def update(self, n): pass
            pbar = _NullPbar()
        else:
            pbar = comfy.utils.ProgressBar(len(target_groups) + n_pbar_merges)
```

**3d. Pass `_skip_qkv_refusion` to Phase 2 candidate merges** (line ~7395). Add to the `super().optimize_merge()` call:

```python
                _skip_qkv_refusion=_is_sub_merge,
```

**3e. Pass `_skip_qkv_refusion` to final full merge** (line ~7602). Add to the `super().optimize_merge()` call:

```python
                _skip_qkv_refusion=_is_sub_merge,
```

**3f. Pass `_skip_qkv_refusion` to single-LoRA path** (line ~7150). Add to the `super().optimize_merge()` call:

```python
                _skip_qkv_refusion=_is_sub_merge,
```

**Step 4: Run test to verify it passes**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRAOptimizer::test_auto_tune_with_formula_calls_autotune_resolve_tree -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v`
Expected: All existing tests pass (1 pre-existing widget order failure)

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: integrate formula detection into AutoTuner with reviewed bug fixes"
```

---

### Task 3: Wire `_suppress_pbar` and `_is_sub_merge` into `_autotune_resolve_tree`

**Files:**
- Modify: `lora_optimizer.py` (inside `_autotune_resolve_tree`)
- Test: `tests/test_lora_optimizer.py`

**Step 1: Write the failing test**

```python
def test_autotune_resolve_tree_passes_sub_merge_flags(self):
    """Sub-merge auto_tune calls should include _is_sub_merge and _suppress_pbar."""
    from lora_optimizer import LoRAAutoTuner, _parse_merge_formula

    tuner = LoRAAutoTuner()
    tree = _parse_merge_formula("(1+2)+3", 3)

    fake_lora_a = {"key_a": torch.randn(4, 4)}
    fake_lora_b = {"key_a": torch.randn(4, 4)}
    fake_lora_c = {"key_c": torch.randn(4, 4)}
    normalized_stack = [
        {"name": "lora_a", "lora": fake_lora_a, "strength": 1.0,
         "clip_strength": None, "metadata": {}},
        {"name": "lora_b", "lora": fake_lora_b, "strength": 1.0,
         "clip_strength": None, "metadata": {}},
        {"name": "lora_c", "lora": fake_lora_c, "strength": 1.0,
         "clip_strength": None, "metadata": {}},
    ]

    calls = []
    def mock_auto_tune(model, lora_stack, output_strength, **kwargs):
        calls.append(kwargs)
        virtual_patches = {"key_a": ("diff", (torch.randn(4, 4),))}
        lora_data = {"model_patches": virtual_patches, "clip_patches": {}}
        return (model, None, "sub-report", "", None, lora_data)

    tuner.auto_tune = mock_auto_tune

    at_kwargs = {
        "clip_strength_multiplier": 1.0,
        "top_n": 3, "normalize_keys": "disabled", "scoring_svd": "disabled",
        "scoring_device": "cpu", "architecture_preset": "dit",
        "auto_strength_floor": -1.0, "decision_smoothing": 0.25,
        "smooth_slerp_gate": False, "vram_budget": 0.0,
        "scoring_speed": "turbo", "scoring_formula": "v2",
        "diff_cache_mode": "disabled", "diff_cache_ram_pct": 0.5,
    }

    tuner._autotune_resolve_tree(tree, normalized_stack, None, None, **at_kwargs)

    self.assertEqual(len(calls), 1)
    self.assertTrue(calls[0].get("_is_sub_merge", False))
    self.assertTrue(calls[0].get("_suppress_pbar", False))
    self.assertEqual(calls[0].get("cache_patches"), "disabled")
    self.assertEqual(calls[0].get("record_dataset"), "disabled")
    self.assertEqual(calls[0].get("output_mode"), "merge")
```

**Step 2: Run test to verify it fails**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRAOptimizer::test_autotune_resolve_tree_passes_sub_merge_flags -v`
Expected: FAIL — `_suppress_pbar` not in kwargs

**Step 3: Implement**

In `_autotune_resolve_tree`, update the sub_kwargs block to also include `_suppress_pbar`:

```python
                        sub_kwargs["_is_sub_merge"] = True
                        sub_kwargs["_suppress_pbar"] = True
```

(The `_is_sub_merge = True` line should already be there from Task 1's implementation.)

**Step 4: Run test to verify it passes**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRAOptimizer::test_autotune_resolve_tree_passes_sub_merge_flags -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "fix: suppress progress bars and pass _is_sub_merge for AutoTuner sub-merges"
```

---

### Task 4: Test single-LoRA sub-group passthrough

**Files:**
- Test: `tests/test_lora_optimizer.py`
- Modify: (none — should already work from Task 1)

**Step 1: Write the test**

```python
def test_autotune_resolve_tree_single_item_passthrough(self):
    """Single-item sub-groups should pass through without calling auto_tune."""
    from lora_optimizer import LoRAAutoTuner, _parse_merge_formula

    tuner = LoRAAutoTuner()
    # (1) + 2 — group with single item should not trigger auto_tune
    tree = _parse_merge_formula("(1)+2", 2)

    fake_lora_a = {"key_a": torch.randn(4, 4)}
    fake_lora_b = {"key_b": torch.randn(4, 4)}
    normalized_stack = [
        {"name": "lora_a", "lora": fake_lora_a, "strength": 1.0,
         "clip_strength": None, "metadata": {}},
        {"name": "lora_b", "lora": fake_lora_b, "strength": 1.0,
         "clip_strength": None, "metadata": {}},
    ]

    calls = []
    def mock_auto_tune(model, lora_stack, output_strength, **kwargs):
        calls.append(True)
        return (model, None, "", "", None, None)

    tuner.auto_tune = mock_auto_tune

    at_kwargs = {
        "clip_strength_multiplier": 1.0,
        "top_n": 3, "normalize_keys": "disabled", "scoring_svd": "disabled",
        "scoring_device": "cpu", "architecture_preset": "dit",
        "auto_strength_floor": -1.0, "decision_smoothing": 0.25,
        "smooth_slerp_gate": False, "vram_budget": 0.0,
        "scoring_speed": "turbo", "scoring_formula": "v2",
        "diff_cache_mode": "disabled", "diff_cache_ram_pct": 0.5,
    }

    resolved_stack, sub_reports = tuner._autotune_resolve_tree(
        tree, normalized_stack, None, None, **at_kwargs)

    # No auto_tune calls — single-item group is passthrough
    self.assertEqual(len(calls), 0)
    # Both items passed through
    self.assertEqual(len(resolved_stack), 2)
    self.assertEqual(resolved_stack[0]["name"], "lora_a")
    self.assertEqual(resolved_stack[1]["name"], "lora_b")
```

**Step 2: Run test**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRAOptimizer::test_autotune_resolve_tree_single_item_passthrough -v`
Expected: PASS (already implemented correctly in Task 1)

**Step 3: Commit**

```bash
git add tests/test_lora_optimizer.py
git commit -m "test: add single-item sub-group passthrough test for AutoTuner formula"
```

---

### Task 5: Test nested formula `((1+2)+3)+4`

**Files:**
- Test: `tests/test_lora_optimizer.py`

**Step 1: Write the test**

```python
def test_autotune_resolve_tree_nested_groups(self):
    """Nested formula ((1+2)+3)+4 should resolve innermost first."""
    from lora_optimizer import LoRAAutoTuner, _parse_merge_formula

    tuner = LoRAAutoTuner()
    tree = _parse_merge_formula("((1+2)+3)+4", 4)

    normalized_stack = []
    for i in range(4):
        normalized_stack.append({
            "name": f"lora_{i}", "lora": {f"key_{i}": torch.randn(4, 4)},
            "strength": 1.0, "clip_strength": None, "metadata": {},
        })

    call_order = []
    def mock_auto_tune(model, lora_stack, output_strength, **kwargs):
        names = [l["name"] for l in lora_stack if not l.get("_precomputed_diffs")]
        virtual_count = sum(1 for l in lora_stack if l.get("_precomputed_diffs"))
        call_order.append({"names": names, "virtual_count": virtual_count,
                           "total": len(lora_stack)})
        virtual_patches = {f"key_v{len(call_order)}": ("diff", (torch.randn(4, 4),))}
        lora_data = {"model_patches": virtual_patches, "clip_patches": {}}
        return (model, None, f"sub-report-{len(call_order)}", "", None, lora_data)

    tuner.auto_tune = mock_auto_tune

    at_kwargs = {
        "clip_strength_multiplier": 1.0,
        "top_n": 3, "normalize_keys": "disabled", "scoring_svd": "disabled",
        "scoring_device": "cpu", "architecture_preset": "dit",
        "auto_strength_floor": -1.0, "decision_smoothing": 0.25,
        "smooth_slerp_gate": False, "vram_budget": 0.0,
        "scoring_speed": "turbo", "scoring_formula": "v2",
        "diff_cache_mode": "disabled", "diff_cache_ram_pct": 0.5,
    }

    resolved_stack, sub_reports = tuner._autotune_resolve_tree(
        tree, normalized_stack, None, None, **at_kwargs)

    # Two auto_tune calls: (1+2) first, then (virtual+3)
    self.assertEqual(len(call_order), 2)
    # First: LoRAs 0 and 1
    self.assertEqual(call_order[0]["names"], ["lora_0", "lora_1"])
    self.assertEqual(call_order[0]["virtual_count"], 0)
    # Second: virtual from (1+2) + lora_2
    self.assertEqual(call_order[1]["total"], 2)
    self.assertEqual(call_order[1]["virtual_count"], 1)

    # Final stack: virtual from ((1+2)+3) + lora_3
    self.assertEqual(len(resolved_stack), 2)
    self.assertTrue(resolved_stack[0].get("_precomputed_diffs"))
    self.assertEqual(resolved_stack[1]["name"], "lora_3")
```

**Step 2: Run test**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRAOptimizer::test_autotune_resolve_tree_nested_groups -v`
Expected: PASS

**Step 3: Run all tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v`
Expected: All pass (1 pre-existing widget order failure)

**Step 4: Commit**

```bash
git add tests/test_lora_optimizer.py
git commit -m "test: add nested formula and full test suite for AutoTuner sub-merge"
```
