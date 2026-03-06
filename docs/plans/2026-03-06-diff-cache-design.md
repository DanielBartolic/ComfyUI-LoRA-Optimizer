# Design: AutoTuner Diff Cache

## Problem

AutoTuner Phase 2 evaluates top-N candidate configs by running a full `optimize_merge()` for each. Every candidate recomputes raw LoRA diffs (A@B matmul) from scratch in `_merge_one_prefix()` (lines 3822-3832), even though diffs depend only on LoRA content — not merge config.

For 3 candidates × 500 prefixes × 3 LoRAs = **4500 redundant matmuls**.

## Solution

Cache raw diffs after the first candidate's merge, reuse for subsequent candidates. The cache stores `diff[prefix][lora_index] → tensor` on CPU (or disk via mmap).

## What's Cacheable

| Component | Cacheable | Reason |
|---|---|---|
| Raw diffs (mat_up @ mat_down × alpha/rank) | Yes | Pure function of LoRA matrices |
| Target shapes per prefix | Yes | Constant |
| LoRA effective strengths | Yes | Same stack across candidates |
| TIES trim masks | No | Depends on density param |
| DARE/DELLA masks | No | Depends on density + dampening |
| Merge algorithm output | No | Different mode per candidate |
| Quality enhancements (KnOTS SVD) | No | Depends on quality level |

## Architecture

### New Parameter on AutoTuner Node

`diff_cache_mode`: combo with options:
- `"ram"` (default) — cache diffs in a dict on CPU
- `"disk"` — cache diffs as memory-mapped temp files
- `"disabled"` — no caching (current behavior)

### Cache Lifecycle

1. **First candidate merge** → compute diffs normally, store in cache
2. **Subsequent candidates** → retrieve diffs from cache, skip A@B
3. **After Phase 2 completes** → clear cache (RAM) or delete temp dir (disk)

### Implementation

#### RAM Mode

```python
# In AutoTuner, before Phase 2 loop:
self._diff_cache = {}  # key: (lora_prefix, lora_index) → CPU tensor

# In _merge_one_prefix, at diff computation (line 3822):
cache_key = (lora_prefix, i)
if self._diff_cache is not None and cache_key in self._diff_cache:
    diff = self._diff_cache[cache_key].to(device)
else:
    diff = torch.mm(mat_up..., mat_down...) * (alpha / rank)
    if self._diff_cache is not None:
        self._diff_cache[cache_key] = diff.cpu()
```

#### Disk Mode

```python
import tempfile

# Before Phase 2:
self._diff_cache_dir = tempfile.mkdtemp(prefix="lora_diff_cache_")

# Store: torch.save(diff.cpu(), os.path.join(dir, f"{prefix}_{i}.pt"))
# Load:  diff = torch.load(path, map_location=device, mmap=True)
```

`torch.load(..., mmap=True)` (PyTorch 2.1+) memory-maps the file, so tensors are paged in on demand without consuming RSS.

### Memory Estimates

| Model | Prefixes | Avg Diff Size | 3 LoRAs | RAM |
|---|---|---|---|---|
| SD 1.5 | ~300 | ~0.5MB | 450MB | Trivial |
| SDXL | ~500 | ~2MB | 3GB | Fine |
| Flux | ~1000 | ~4MB | 12GB | Needs 32GB+ system |

### Where to Modify

1. **AutoTuner node INPUT_TYPES** — add `diff_cache_mode` combo
2. **AutoTuner Phase 2 loop** (~line 4492) — init cache before loop, clear after
3. **`_merge_one_prefix`** (~line 3822) — check cache before computing diff
4. **`optimize_merge`** — thread the cache object through (add `_diff_cache` param alongside existing `_analysis_cache`)

### Cleanup

```python
# After Phase 2 (line ~4570):
if self._diff_cache is not None:
    self._diff_cache.clear()
    self._diff_cache = None
if self._diff_cache_dir:
    shutil.rmtree(self._diff_cache_dir, ignore_errors=True)
```

## Expected Speedup

For top_n=3: eliminates 2/3 of A@B matmuls in Pass 2. Since diff computation is a significant fraction of merge time (especially on CPU), this should reduce Phase 2 wall time by **30-50%**.

For top_n=5+: eliminates 4/5 of matmuls → **50-70%** reduction.

## Risks

- **RAM pressure**: 12GB for Flux. Mitigated by disk mode option.
- **Cache invalidation**: Not needed — cache lives only for one AutoTuner run, cleared after Phase 2.
- **Thread safety**: `_merge_one_prefix` is called from ThreadPoolExecutor on CPU mode. Dict reads are thread-safe in CPython (GIL). Writes need care — first candidate populates, subsequent candidates read. Since candidates run sequentially (line 4492), this is safe.
