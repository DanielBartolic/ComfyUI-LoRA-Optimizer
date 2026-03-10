# Merge Formula Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `LoRAMergeFormula` passthrough node that lets users define hierarchical merge order via a text formula (e.g., `(1+2) + 3`), so sub-groups are merged first using the full pipeline before combining with other LoRAs.

**Architecture:** A formula parser converts the text expression into a merge tree. The optimizer detects the formula metadata in the stack, executes the tree leaf-to-root (each node is a full `optimize_merge` call), and wraps inner results as virtual LoRAs for the outer merge. No changes to `_merge_diffs` or any merge algorithm — this is pure orchestration.

**Tech Stack:** Python 3.10+, PyTorch, ComfyUI node API

---

## Task 1: Formula Parser

**Files:**
- Modify: `lora_optimizer.py` — add `_parse_merge_formula` function (before `LoRAStack` class, around line 160)
- Modify: `tests/test_lora_optimizer.py` — add parser tests

The parser converts `(1+2):0.6 + 3:0.4` into a tree:
```python
# Tree node: either a leaf (LoRA index) or a group (list of children)
# {"type": "leaf", "index": 0, "weight": None}
# {"type": "group", "children": [...], "weight": 0.6}
```

**Step 1: Write failing tests**

Add to `LoRAOptimizerTests` in `tests/test_lora_optimizer.py`:

```python
def test_parse_merge_formula_simple(self):
    """Simple flat formula parses to group of leaves."""
    tree = lora_optimizer._parse_merge_formula("1 + 2 + 3", 3)
    self.assertEqual(tree["type"], "group")
    self.assertEqual(len(tree["children"]), 3)
    for i, child in enumerate(tree["children"]):
        self.assertEqual(child["type"], "leaf")
        self.assertEqual(child["index"], i)

def test_parse_merge_formula_nested(self):
    """Nested formula parses to tree with sub-group."""
    tree = lora_optimizer._parse_merge_formula("(1+2) + 3", 3)
    self.assertEqual(tree["type"], "group")
    self.assertEqual(len(tree["children"]), 2)
    sub = tree["children"][0]
    self.assertEqual(sub["type"], "group")
    self.assertEqual(len(sub["children"]), 2)
    leaf3 = tree["children"][1]
    self.assertEqual(leaf3["type"], "leaf")
    self.assertEqual(leaf3["index"], 2)  # 0-indexed

def test_parse_merge_formula_weights(self):
    """Weights are parsed from :N.N suffix."""
    tree = lora_optimizer._parse_merge_formula("(1+2):0.6 + 3:0.4", 3)
    self.assertAlmostEqual(tree["children"][0]["weight"], 0.6)
    self.assertAlmostEqual(tree["children"][1]["weight"], 0.4)

def test_parse_merge_formula_deep_nesting(self):
    """Deep nesting: ((1+2)+3) + 4."""
    tree = lora_optimizer._parse_merge_formula("((1+2)+3) + 4", 4)
    self.assertEqual(tree["type"], "group")
    self.assertEqual(len(tree["children"]), 2)
    inner = tree["children"][0]
    self.assertEqual(inner["type"], "group")
    self.assertEqual(len(inner["children"]), 2)
    innermost = inner["children"][0]
    self.assertEqual(innermost["type"], "group")
    self.assertEqual(len(innermost["children"]), 2)

def test_parse_merge_formula_single_item(self):
    """Single item is valid."""
    tree = lora_optimizer._parse_merge_formula("1", 1)
    self.assertEqual(tree["type"], "leaf")
    self.assertEqual(tree["index"], 0)

def test_parse_merge_formula_out_of_range(self):
    """Out of range index raises ValueError."""
    with self.assertRaises(ValueError):
        lora_optimizer._parse_merge_formula("1 + 5", 3)

def test_parse_merge_formula_malformed(self):
    """Malformed formula raises ValueError."""
    with self.assertRaises(ValueError):
        lora_optimizer._parse_merge_formula("((1+2", 3)

def test_parse_merge_formula_empty(self):
    """Empty/whitespace formula raises ValueError."""
    with self.assertRaises(ValueError):
        lora_optimizer._parse_merge_formula("", 3)
    with self.assertRaises(ValueError):
        lora_optimizer._parse_merge_formula("   ", 3)
```

**Step 2: Run tests to verify they fail**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m unittest tests.test_lora_optimizer -v -k parse_merge 2>&1 | tail -20`
Expected: FAIL — `_parse_merge_formula` does not exist

**Step 3: Implement the parser**

Add this function before the `LoRAStack` class (around line 160 in `lora_optimizer.py`):

```python
def _parse_merge_formula(formula_str, n_loras):
    """
    Parse a merge formula string into a tree structure.

    Syntax:
        expr   = term (('+') term)*
        term   = atom (':' weight)?
        atom   = NUMBER | '(' expr ')'
        weight = FLOAT

    Numbers are 1-indexed LoRA positions. Returns a tree of:
        {"type": "leaf", "index": int, "weight": float|None}
        {"type": "group", "children": list, "weight": float|None}

    Raises ValueError on malformed input or out-of-range indices.
    """
    formula_str = formula_str.strip()
    if not formula_str:
        raise ValueError("Empty merge formula")

    pos = [0]  # mutable position cursor

    def _skip_ws():
        while pos[0] < len(formula_str) and formula_str[pos[0]] == ' ':
            pos[0] += 1

    def _parse_weight():
        _skip_ws()
        if pos[0] < len(formula_str) and formula_str[pos[0]] == ':':
            pos[0] += 1  # skip ':'
            _skip_ws()
            start = pos[0]
            while pos[0] < len(formula_str) and (formula_str[pos[0]].isdigit() or formula_str[pos[0]] == '.'):
                pos[0] += 1
            if pos[0] == start:
                raise ValueError(f"Expected weight after ':' at position {pos[0]}")
            return float(formula_str[start:pos[0]])
        return None

    def _parse_atom():
        _skip_ws()
        if pos[0] >= len(formula_str):
            raise ValueError("Unexpected end of formula")

        if formula_str[pos[0]] == '(':
            pos[0] += 1  # skip '('
            node = _parse_expr()
            _skip_ws()
            if pos[0] >= len(formula_str) or formula_str[pos[0]] != ')':
                raise ValueError(f"Expected ')' at position {pos[0]}")
            pos[0] += 1  # skip ')'
            weight = _parse_weight()
            node["weight"] = weight
            return node

        # Must be a number
        start = pos[0]
        while pos[0] < len(formula_str) and formula_str[pos[0]].isdigit():
            pos[0] += 1
        if pos[0] == start:
            raise ValueError(f"Unexpected character '{formula_str[pos[0]]}' at position {pos[0]}")
        index_1based = int(formula_str[start:pos[0]])
        if index_1based < 1 or index_1based > n_loras:
            raise ValueError(f"LoRA index {index_1based} out of range (have {n_loras} LoRAs)")
        weight = _parse_weight()
        return {"type": "leaf", "index": index_1based - 1, "weight": weight}

    def _parse_expr():
        children = [_parse_atom()]
        while True:
            _skip_ws()
            if pos[0] < len(formula_str) and formula_str[pos[0]] == '+':
                pos[0] += 1  # skip '+'
                children.append(_parse_atom())
            else:
                break
        if len(children) == 1:
            return children[0]
        return {"type": "group", "children": children, "weight": None}

    result = _parse_expr()
    _skip_ws()
    if pos[0] != len(formula_str):
        raise ValueError(f"Unexpected content at position {pos[0]}: '{formula_str[pos[0]:]}'")
    return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m unittest tests.test_lora_optimizer -v -k parse_merge 2>&1 | tail -20`
Expected: All 8 parse tests PASS

**Step 5: Syntax check and full test suite**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"` and `python -m unittest tests.test_lora_optimizer -v 2>&1 | tail -10`

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add merge formula parser with recursive descent"
```

---

## Task 2: LoRAMergeFormula Node

**Files:**
- Modify: `lora_optimizer.py` — add `LoRAMergeFormula` class (after `LoRAStackDynamic`, before `LoRAOptimizer`)
- Modify: `lora_optimizer.py:9303-9345` — register in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
- Modify: `tests/test_lora_optimizer.py` — add node registration test

**Step 1: Write failing test**

```python
def test_merge_formula_node_registered(self):
    """LoRAMergeFormula is registered in NODE_CLASS_MAPPINGS."""
    self.assertIn("LoRAMergeFormula", lora_optimizer.NODE_CLASS_MAPPINGS)
    self.assertIn("LoRAMergeFormula", lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS)

def test_merge_formula_node_passthrough(self):
    """LoRAMergeFormula passes stack through with formula metadata."""
    node = lora_optimizer.LoRAMergeFormula()
    stack = [{"name": "a", "lora": {}, "strength": 1.0}]
    result = node.apply_formula(stack, "(1)")
    self.assertIsInstance(result, tuple)
    output_stack = result[0]
    # Should contain original items plus metadata entry
    has_formula = any(isinstance(item, dict) and "_merge_formula" in item for item in output_stack)
    self.assertTrue(has_formula)

def test_merge_formula_node_validates(self):
    """LoRAMergeFormula validates formula syntax."""
    node = lora_optimizer.LoRAMergeFormula()
    stack = [{"name": "a", "lora": {}, "strength": 1.0}]
    # Out of range should not crash — node should handle gracefully
    # (ComfyUI nodes should not raise; they should return error in report or log)
    result = node.apply_formula(stack, "(1+2)")  # only 1 LoRA
    output_stack = result[0]
    # Should still return something usable (original stack without formula)
    self.assertIsInstance(output_stack, list)
```

**Step 2: Run tests to verify they fail**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m unittest tests.test_lora_optimizer -v -k merge_formula_node 2>&1 | tail -15`

**Step 3: Implement the node**

Add the class after `LoRAStackDynamic` (find the right insertion point — after the `LoRAStackDynamic` class ends, before `LoRAOptimizer`). You'll need to find where `LoRAStackDynamic` ends. Look for the next class definition after it.

```python
class LoRAMergeFormula:
    """
    Passthrough node that attaches a merge formula to the LoRA stack.
    The formula defines hierarchical merge order, e.g., "(1+2) + 3"
    merges LoRAs 1 & 2 first, then merges the result with LoRA 3.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "The LoRA stack to apply the merge formula to."
                }),
                "formula": ("STRING", {
                    "default": "",
                    "tooltip": "Merge formula defining hierarchical merge order. "
                               "Numbers reference 1-indexed LoRA positions in the stack. "
                               "Use + to combine and () to group sub-merges. "
                               "Example: '(1+2) + 3' merges LoRAs 1 & 2 first, then blends with 3. "
                               "Optional weights: '(1+2):0.6 + 3:0.4'. "
                               "Leave empty for default flat merge."
                }),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "apply_formula"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Attaches a merge formula to the LoRA stack to control hierarchical merge order"

    def apply_formula(self, lora_stack, formula):
        output = list(lora_stack) if lora_stack else []
        formula = formula.strip()
        if not formula:
            return (output,)

        # Count actual LoRAs (exclude any existing formula metadata)
        n_loras = sum(1 for item in output
                      if isinstance(item, dict) and "_merge_formula" not in item)

        # Validate formula
        try:
            _parse_merge_formula(formula, n_loras)
        except ValueError as e:
            logging.warning(f"[LoRA Optimizer] Invalid merge formula: {e} — using flat merge")
            return (output,)

        # Remove any existing formula metadata (in case of chaining)
        output = [item for item in output
                  if not (isinstance(item, dict) and "_merge_formula" in item)]
        output.append({"_merge_formula": formula})
        return (output,)
```

Then register in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` (around line 9303):

Add `"LoRAMergeFormula": LoRAMergeFormula,` to NODE_CLASS_MAPPINGS.
Add `"LoRAMergeFormula": "LoRA Merge Formula",` to NODE_DISPLAY_NAME_MAPPINGS.

**Step 4: Run tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m unittest tests.test_lora_optimizer -v -k merge_formula_node 2>&1 | tail -15`

**Step 5: Syntax check and full suite**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"` and `python -m unittest tests.test_lora_optimizer -v 2>&1 | tail -10`

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add LoRAMergeFormula passthrough node"
```

---

## Task 3: Tree Executor + Virtual LoRA Wrapping

**Files:**
- Modify: `lora_optimizer.py` — add `_execute_merge_tree` method on `LoRAOptimizer` class
- Modify: `tests/test_lora_optimizer.py` — add executor tests

This is the core orchestration: walk the tree, call `optimize_merge` for each sub-group, wrap results as virtual LoRAs.

**Step 1: Write failing tests**

```python
def test_execute_merge_tree_flat(self):
    """Flat tree returns all LoRAs unchanged."""
    tree = {"type": "group", "children": [
        {"type": "leaf", "index": 0, "weight": None},
        {"type": "leaf", "index": 1, "weight": None},
    ], "weight": None}
    stack = [
        {"name": "A", "lora": {"key1": ("diff", (torch.randn(4, 4),))}, "strength": 1.0},
        {"name": "B", "lora": {"key1": ("diff", (torch.randn(4, 4),))}, "strength": 1.0},
    ]
    opt = lora_optimizer.LoRAOptimizer()
    result_stack = opt._collect_tree_leaves(tree, stack)
    # Flat tree with all leaves — should return all items
    self.assertEqual(len(result_stack), 2)

def test_execute_merge_tree_detects_subgroups(self):
    """Nested tree correctly identifies sub-groups that need merging."""
    tree = lora_optimizer._parse_merge_formula("(1+2) + 3", 3)
    # The tree root has 2 children: a group and a leaf
    self.assertEqual(tree["children"][0]["type"], "group")
    self.assertEqual(tree["children"][1]["type"], "leaf")
```

**Step 2: Implement `_execute_merge_tree` and `_collect_tree_leaves`**

Add to the `LoRAOptimizer` class (near `optimize_merge`, around line 5220):

```python
    def _execute_merge_tree(self, tree, lora_stack, model, clip, **merge_kwargs):
        """
        Execute a merge formula tree. Each sub-group triggers a full
        optimize_merge call. Results are wrapped as virtual LoRAs for
        the parent merge.

        Returns: (model_out, clip_out, report, tuner_data, lora_data)
        """
        if tree["type"] == "leaf":
            # Single LoRA — just run normal optimize_merge with a 1-item stack
            idx = tree["index"]
            item = lora_stack[idx]
            if tree["weight"] is not None:
                item = dict(item)
                item["strength"] = tree["weight"]
            return self.optimize_merge(model, [item], merge_kwargs.pop("output_strength", 1.0),
                                       clip=clip, **merge_kwargs)

        # Group node: process each child
        children = tree["children"]
        sub_results = []
        sub_reports = []

        for child in children:
            if child["type"] == "leaf":
                # Leaf: just collect the LoRA dict
                idx = child["index"]
                item = dict(lora_stack[idx])
                if child["weight"] is not None:
                    item["strength"] = child["weight"]
                sub_results.append(item)
            elif child["type"] == "group":
                if len(child["children"]) == 1 and child["children"][0]["type"] == "leaf":
                    # Single-leaf group: no merge needed, just apply weight
                    leaf = child["children"][0]
                    item = dict(lora_stack[leaf["index"]])
                    if child["weight"] is not None:
                        item["strength"] = child["weight"]
                    elif leaf["weight"] is not None:
                        item["strength"] = leaf["weight"]
                    sub_results.append(item)
                else:
                    # Multi-item sub-group: recurse — full optimize_merge
                    sub_stack = []
                    for sub_child in child["children"]:
                        if sub_child["type"] == "leaf":
                            item = dict(lora_stack[sub_child["index"]])
                            if sub_child["weight"] is not None:
                                item["strength"] = sub_child["weight"]
                            sub_stack.append(item)
                        else:
                            # Deeper nesting: recursive call
                            sub_model, sub_clip, sub_report, _, sub_lora_data = \
                                self._execute_merge_tree(sub_child, lora_stack, model, clip, **merge_kwargs)
                            sub_reports.append(sub_report)
                            # Extract patches from sub-merged model as virtual LoRA
                            virtual = self._extract_virtual_lora(model, sub_model, clip, sub_clip, sub_child)
                            if sub_child["weight"] is not None:
                                virtual["strength"] = sub_child["weight"]
                            sub_results.append(virtual)

                    if len(sub_stack) >= 2:
                        # Merge this sub-group
                        sub_model, sub_clip, sub_report, _, sub_lora_data = \
                            self.optimize_merge(model, sub_stack,
                                                merge_kwargs.get("output_strength", 1.0),
                                                clip=clip, **{k: v for k, v in merge_kwargs.items()
                                                              if k != "output_strength"})
                        sub_reports.append(sub_report)
                        virtual = self._extract_virtual_lora(model, sub_model, clip, sub_clip, child)
                        if child["weight"] is not None:
                            virtual["strength"] = child["weight"]
                        sub_results.append(virtual)
                    elif len(sub_stack) == 1:
                        item = sub_stack[0]
                        if child["weight"] is not None:
                            item["strength"] = child["weight"]
                        sub_results.append(item)

        # Final merge of all sub_results
        combined_report_parts = sub_reports
        output_strength = merge_kwargs.pop("output_strength", 1.0)
        final_model, final_clip, final_report, tuner_data, lora_data = \
            self.optimize_merge(model, sub_results, output_strength,
                                clip=clip, **merge_kwargs)
        # Combine reports
        separator = "\n" + "=" * 50 + "\n"
        if combined_report_parts:
            sub_section = separator.join(combined_report_parts)
            final_report = (
                "MERGE FORMULA: Sub-merge reports\n"
                + separator + sub_section + separator
                + "\nFinal merge report:\n" + final_report
            )

        return (final_model, final_clip, final_report, tuner_data, lora_data)

    @staticmethod
    def _extract_virtual_lora(base_model, merged_model, base_clip, merged_clip, tree_node):
        """
        Extract the difference between base and merged models as a virtual LoRA dict.
        The virtual LoRA can be fed into a subsequent optimize_merge call.
        """
        virtual_lora = {}

        # Extract model patches
        if merged_model is not None and hasattr(merged_model, 'model'):
            patches = getattr(merged_model.model, 'patches', {})
            for key, patch_list in patches.items():
                for patch in patch_list:
                    virtual_lora[key] = patch

        # Extract CLIP patches
        if merged_clip is not None:
            clip_patches = {}
            clip_model = getattr(merged_clip, 'patcher', merged_clip)
            if hasattr(clip_model, 'model'):
                for key, patch_list in getattr(clip_model.model, 'patches', {}).items():
                    for patch in patch_list:
                        virtual_lora[key] = patch

        # Build label from tree
        def _tree_label(node):
            if node["type"] == "leaf":
                return str(node["index"] + 1)
            return "(" + "+".join(_tree_label(c) for c in node["children"]) + ")"

        return {
            "name": _tree_label(tree_node),
            "lora": virtual_lora,
            "strength": 1.0,
            "clip_strength": None,
            "conflict_mode": "all",
            "key_filter": "all",
            "metadata": {},
        }
```

**Step 3: Run tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m unittest tests.test_lora_optimizer -v -k execute_merge_tree 2>&1 | tail -15`

**Step 4: Syntax check and full suite**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"` and `python -m unittest tests.test_lora_optimizer -v 2>&1 | tail -10`

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add merge tree executor with virtual LoRA wrapping"
```

---

## Task 4: Integrate into `optimize_merge` + Handle virtual LoRAs in `_normalize_stack`

**Files:**
- Modify: `lora_optimizer.py:5251-5256` — detect formula in stack, delegate to tree executor
- Modify: `lora_optimizer.py:3009-3023` — handle virtual LoRA dicts in `_normalize_stack`
- Modify: `tests/test_lora_optimizer.py` — add integration test

**Step 1: Write failing test**

```python
def test_merge_formula_end_to_end(self):
    """Full pipeline: formula node → optimizer produces valid output."""
    torch.manual_seed(42)
    # Create a 3-LoRA stack with formula metadata
    lora_a = {"key1.weight": ("diff", (torch.randn(8, 8),))}
    lora_b = {"key1.weight": ("diff", (torch.randn(8, 8),))}
    lora_c = {"key1.weight": ("diff", (torch.randn(8, 8),))}
    stack = [
        {"name": "char_a", "lora": lora_a, "strength": 1.0},
        {"name": "char_b", "lora": lora_b, "strength": 1.0},
        {"name": "style", "lora": lora_c, "strength": 1.0},
        {"_merge_formula": "(1+2) + 3"},
    ]
    opt = lora_optimizer.LoRAOptimizer()
    # This should not crash — model=None is fine for diff-based merge
    # (the optimizer handles None model gracefully for the merge pipeline)
    # We mainly verify the formula path is taken and doesn't error
    result = opt.optimize_merge(None, stack, 1.0)
    self.assertIsNotNone(result)
```

**Step 2: Modify `optimize_merge` to detect and handle formula**

At the top of `optimize_merge`, after the `if not lora_stack` guard (around line 5252), add formula detection:

```python
        # Check for merge formula metadata
        merge_formula = None
        clean_stack = []
        if lora_stack:
            for item in lora_stack:
                if isinstance(item, dict) and "_merge_formula" in item:
                    merge_formula = item["_merge_formula"]
                else:
                    clean_stack.append(item)
            if merge_formula:
                lora_stack = clean_stack
```

Then after `normalize_stack` and the `active_loras` filter, before the single-LoRA fast path (around line 5271), add the formula delegation:

```python
        # Formula-based hierarchical merge
        if merge_formula and len(active_loras) >= 2:
            try:
                tree = _parse_merge_formula(merge_formula, len(active_loras))
            except ValueError as e:
                logging.warning(f"[LoRA Optimizer] Invalid merge formula: {e} — using flat merge")
                tree = None

            if tree is not None and tree["type"] == "group":
                logging.info(f"[LoRA Optimizer] Using merge formula: {merge_formula}")
                return self._execute_merge_tree(
                    tree, active_loras, model, clip,
                    output_strength=output_strength,
                    clip_strength_multiplier=clip_strength_multiplier,
                    auto_strength=auto_strength,
                    optimization_mode=optimization_mode,
                    patch_compression=patch_compression,
                    svd_device=svd_device,
                    normalize_keys=normalize_keys,
                    sparsification=sparsification,
                    sparsification_density=sparsification_density,
                    dare_dampening=dare_dampening,
                    merge_refinement=merge_refinement,
                    strategy_set=strategy_set,
                    architecture_preset=architecture_preset,
                    decision_smoothing=decision_smoothing,
                )
```

**Step 3: Handle virtual LoRAs in `_normalize_stack`**

In `_normalize_stack` (around line 3009), the `isinstance(first, dict)` branch already handles dicts. Virtual LoRAs have the same format as LoRAStack dicts (`name`, `lora`, `strength`), so they should work as-is. However, we need to make sure the metadata entry `{"_merge_formula": ...}` is filtered out before normalization.

Add at the very top of `_normalize_stack`, before the `if not lora_stack` guard:

```python
        # Filter out formula metadata entries
        if lora_stack:
            lora_stack = [item for item in lora_stack
                          if not (isinstance(item, dict) and "_merge_formula" in item)]
```

**Step 4: Run tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m unittest tests.test_lora_optimizer -v -k merge_formula_end 2>&1 | tail -15`

**Step 5: Full test suite**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m unittest tests.test_lora_optimizer -v 2>&1 | tail -10`

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: integrate merge formula into optimize_merge pipeline"
```

---

## Task 5: Final Verification

**Files:**
- Modify: `tests/test_lora_optimizer.py` — optional: add more edge case tests

**Step 1: Run full test suite and syntax check**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"` and `python -m unittest tests.test_lora_optimizer -v 2>&1 | tail -20`

Expected: All new tests pass, only pre-existing widget order test fails.

**Step 2: Verify node appears correctly**

Run: `python -c "from lora_optimizer import NODE_CLASS_MAPPINGS; print('LoRAMergeFormula' in NODE_CLASS_MAPPINGS)"`
Expected: `True`

**Step 3: Commit any final fixes**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "test: add merge formula edge case tests"
```

---

## Summary

| Task | What |
|------|------|
| 1 | Formula parser (recursive descent, tree output) |
| 2 | LoRAMergeFormula passthrough node |
| 3 | Tree executor + virtual LoRA wrapping |
| 4 | Integration into optimize_merge + _normalize_stack |
| 5 | Final verification |

## Key Design Decisions

- **Formula metadata travels in the stack** as `{"_merge_formula": "(1+2)+3"}` — filtered out before normalization
- **Each sub-merge is a full `optimize_merge`** call — no shortcuts
- **Virtual LoRAs** enter outer merge at strength 1.0 — auto-strength handles scaling
- **`_extract_virtual_lora`** pulls patches from the ComfyUI model patcher's internal state
- **Backward compatible** — no formula = identical to current behavior
