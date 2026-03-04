import { app } from "/scripts/app.js";

const HIDDEN_TAG = "loraopt_hidden";
const origProps = {};

function toggleWidget(node, widget, show, suffix = "") {
    if (!widget) return;

    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
        };
    }

    widget.hidden = !show;
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show
        ? origProps[widget.name].origComputeSize
        : () => [0, -4];

    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            toggleWidget(node, w, show, ":" + widget.name);
        }
    }
}

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function interceptWidgetValue(widget, onChange) {
    let widgetValue = widget.value;
    const desc =
        Object.getOwnPropertyDescriptor(widget, "value") ||
        Object.getOwnPropertyDescriptor(
            Object.getPrototypeOf(widget),
            "value"
        );

    Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get() {
            return desc?.get ? desc.get.call(widget) : widgetValue;
        },
        set(newVal) {
            if (desc?.set) {
                desc.set.call(widget, newVal);
            } else {
                widgetValue = newVal;
            }
            onChange(newVal);
        },
    });
}

function updateVisibility(node) {
    const modeWidget = findWidget(node, "mode");
    const countWidget = findWidget(node, "lora_count");
    if (!modeWidget || !countWidget) return;

    const isSimple = modeWidget.value === "simple";
    const count = countWidget.value;
    const MAX = 10;

    for (let i = 1; i <= MAX; i++) {
        const visible = i <= count;

        toggleWidget(node, findWidget(node, `lora_name_${i}`), visible);
        toggleWidget(node, findWidget(node, `strength_${i}`), visible && isSimple);
        toggleWidget(node, findWidget(node, `model_strength_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `clip_strength_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `conflict_mode_${i}`), visible);
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
    app.canvas?.setDirty?.(true, true);
}

// --- Base Model Filter (requires ComfyUI-Lora-Manager) ---

const LM_LORAS_LIST_URL = "/api/lm/loras/list";
const LM_BASE_MODELS_URL = "/api/lm/loras/base-models?limit=100";
const LM_AUTOCOMPLETE_URL = "/api/lm/loras/relative-paths";
const PAGE_SIZE = 100;

async function fetchJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}

/**
 * Fetch ALL LoRAs from the Lora Manager (paginating through the 100-per-page cap)
 * and build a Map of relative_path -> base_model for local filtering.
 */
async function buildLoraBaseModelMap(fullLoraList) {
    const map = new Map();
    let page = 1;
    let totalPages = 1;

    do {
        const params = new URLSearchParams({
            page: String(page),
            page_size: String(PAGE_SIZE),
            sort_by: "name",
        });
        const data = await fetchJson(`${LM_LORAS_LIST_URL}?${params}`);
        totalPages = data.total_pages || 1;

        for (const item of data.items || []) {
            if (!item.file_path || !item.base_model) continue;
            const apiPath = item.file_path;
            // Match API absolute path against ComfyUI's relative paths
            for (const relPath of fullLoraList) {
                if (relPath === "None") continue;
                if (
                    apiPath === relPath ||
                    apiPath.endsWith("/" + relPath) ||
                    apiPath.endsWith("\\" + relPath)
                ) {
                    map.set(relPath, item.base_model);
                    break;
                }
            }
        }
        page++;
    } while (page <= totalPages);

    return map;
}

function setComboOptions(widget, options) {
    widget.options.values = options;
    // Do NOT reset the selected value — preserve it even if not in the filtered list.
    // ComfyUI COMBO widgets allow values not in the options list; they just can't be
    // re-selected from the dropdown. This prevents silent data loss when switching filters.
}

async function initBaseModelFilter(node, retries = 0) {
    const filterWidget = findWidget(node, "base_model_filter");
    if (!filterWidget) return;

    // Cache the full LoRA list from the first lora_name widget
    const firstLoraWidget = findWidget(node, "lora_name_1");
    if (!firstLoraWidget) return;
    const loraValues = firstLoraWidget.options?.values;
    if (!loraValues || loraValues.length <= 1) {
        // Widget not yet populated, retry (max 20 attempts = 10 seconds)
        if (retries < 20) {
            setTimeout(() => initBaseModelFilter(node, retries + 1), 500);
        }
        return;
    }
    const fullLoraList = [...loraValues];

    // Try to detect Lora Manager and fetch base models
    let baseModels;
    try {
        const data = await fetchJson(LM_BASE_MODELS_URL);
        baseModels = (data.base_models || [])
            .map((m) => m.name)
            .filter(Boolean);
    } catch {
        // Lora Manager not installed — hide filter widget
        toggleWidget(node, filterWidget, false);
        updateVisibility(node);
        return;
    }

    if (baseModels.length === 0) {
        toggleWidget(node, filterWidget, false);
        updateVisibility(node);
        return;
    }

    // Build the path → base_model map once (handles pagination)
    let loraBaseModelMap;
    try {
        loraBaseModelMap = await buildLoraBaseModelMap(fullLoraList);
    } catch {
        toggleWidget(node, filterWidget, false);
        updateVisibility(node);
        return;
    }

    // Populate filter dropdown
    const filterOptions = ["All", ...baseModels];
    setComboOptions(filterWidget, filterOptions);

    // Apply current filter value (handles workflow restore)
    if (filterWidget.value && filterWidget.value !== "All") {
        applyLoraFilter(node, filterWidget.value, fullLoraList, loraBaseModelMap);
    }

    // Intercept future filter changes
    interceptWidgetValue(filterWidget, (newVal) => {
        applyLoraFilter(node, newVal, fullLoraList, loraBaseModelMap);
    });
}

function applyLoraFilter(node, baseModel, fullLoraList, loraBaseModelMap) {
    const MAX = 10;
    let filteredList;

    if (baseModel === "All") {
        filteredList = fullLoraList;
    } else {
        filteredList = fullLoraList.filter(
            (name) =>
                name === "None" || loraBaseModelMap.get(name) === baseModel
        );
        // If nothing matched, fall back to full list
        if (filteredList.length <= 1) {
            filteredList = fullLoraList;
        }
    }

    for (let i = 1; i <= MAX; i++) {
        const w = findWidget(node, `lora_name_${i}`);
        if (w) setComboOptions(w, filteredList);
    }

    app.canvas?.setDirty?.(true, true);
}

// --- LoRA Text Autocomplete (requires ComfyUI-Lora-Manager) ---

class LoraTextAutocomplete {
    constructor(textareaEl) {
        this.textarea = textareaEl;
        this.items = [];
        this.selectedIdx = -1;
        this._debounceTimer = null;

        this.dropdown = document.createElement("div");
        this.dropdown.style.cssText =
            "position:absolute;z-index:9999;background:#1a1a2e;border:1px solid #555;" +
            "max-height:200px;overflow-y:auto;display:none;font-size:13px;" +
            "border-radius:4px;box-shadow:0 4px 12px rgba(0,0,0,0.4);";
        document.body.appendChild(this.dropdown);

        this._onInput = this._handleInput.bind(this);
        this._onKeyDown = this._handleKeyDown.bind(this);
        this._onBlur = () => setTimeout(() => this._hide(), 150);
        this.textarea.addEventListener("input", this._onInput);
        this.textarea.addEventListener("keydown", this._onKeyDown);
        this.textarea.addEventListener("blur", this._onBlur);
    }

    _getCurrentLine() {
        const val = this.textarea.value;
        const pos = this.textarea.selectionStart;
        const start = val.lastIndexOf("\n", pos - 1) + 1;
        let end = val.indexOf("\n", pos);
        if (end === -1) end = val.length;
        return { text: val.slice(start, end), start, end };
    }

    _handleInput() {
        clearTimeout(this._debounceTimer);
        this._debounceTimer = setTimeout(async () => {
            const { text } = this._getCurrentLine();
            // Strip trailing :N or :N:N for search term
            const term = text.replace(/:[^:]*$/, "").replace(/:[^:]*$/, "").trim();
            if (term.length < 2) {
                this._hide();
                return;
            }
            await this._search(term);
        }, 200);
    }

    async _search(term) {
        try {
            const params = new URLSearchParams({ search: term, limit: "20" });
            const data = await fetchJson(`${LM_AUTOCOMPLETE_URL}?${params}`);
            this.items = data.relative_paths || data || [];
            if (!Array.isArray(this.items)) this.items = [];
        } catch {
            this.items = [];
        }
        if (this.items.length === 0) {
            this._hide();
            return;
        }
        this.selectedIdx = 0;
        this._render(term);
        this._show();
    }

    _render(searchTerm) {
        this.dropdown.innerHTML = "";
        const lowerTerm = searchTerm.toLowerCase();
        this.items.forEach((path, idx) => {
            const row = document.createElement("div");
            row.style.cssText =
                "padding:4px 8px;cursor:pointer;white-space:nowrap;color:#e0e0e0;";
            if (idx === this.selectedIdx) {
                row.style.background = "#3a3a5e";
            }
            // Highlight matching portion
            const lowerPath = path.toLowerCase();
            const matchIdx = lowerPath.indexOf(lowerTerm);
            if (matchIdx >= 0) {
                const before = path.slice(0, matchIdx);
                const match = path.slice(matchIdx, matchIdx + searchTerm.length);
                const after = path.slice(matchIdx + searchTerm.length);
                row.innerHTML =
                    `${this._esc(before)}<b style="color:#7ca8ff">${this._esc(match)}</b>${this._esc(after)}`;
            } else {
                row.textContent = path;
            }
            row.addEventListener("mousedown", (e) => {
                e.preventDefault();
                this._insert(path);
            });
            row.addEventListener("mouseenter", () => {
                this.selectedIdx = idx;
                this._render(searchTerm);
            });
            this.dropdown.appendChild(row);
        });
    }

    _esc(str) {
        const d = document.createElement("span");
        d.textContent = str;
        return d.innerHTML;
    }

    _show() {
        const rect = this.textarea.getBoundingClientRect();
        this.dropdown.style.left = `${rect.left + window.scrollX}px`;
        this.dropdown.style.top = `${rect.bottom + window.scrollY + 2}px`;
        this.dropdown.style.width = `${rect.width}px`;
        this.dropdown.style.display = "block";
    }

    _hide() {
        this.dropdown.style.display = "none";
        this.items = [];
        this.selectedIdx = -1;
    }

    _insert(path) {
        const { start, end } = this._getCurrentLine();
        const val = this.textarea.value;
        this.textarea.value = val.slice(0, start) + path + val.slice(end);
        const newPos = start + path.length;
        this.textarea.setSelectionRange(newPos, newPos);
        this.textarea.dispatchEvent(new Event("input", { bubbles: true }));
        this._hide();
    }

    _handleKeyDown(e) {
        if (this.dropdown.style.display === "none") return;

        if (e.key === "ArrowDown") {
            e.preventDefault();
            this.selectedIdx = (this.selectedIdx + 1) % this.items.length;
            this._render(this._getCurrentLine().text.replace(/:[^:]*$/, "").replace(/:[^:]*$/, "").trim());
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            this.selectedIdx = (this.selectedIdx - 1 + this.items.length) % this.items.length;
            this._render(this._getCurrentLine().text.replace(/:[^:]*$/, "").replace(/:[^:]*$/, "").trim());
        } else if (e.key === "Enter" && this.selectedIdx >= 0) {
            e.preventDefault();
            this._insert(this.items[this.selectedIdx]);
        } else if (e.key === "Escape") {
            e.preventDefault();
            this._hide();
        }
    }

    destroy() {
        clearTimeout(this._debounceTimer);
        this.textarea.removeEventListener("input", this._onInput);
        this.textarea.removeEventListener("keydown", this._onKeyDown);
        this.textarea.removeEventListener("blur", this._onBlur);
        this.dropdown.remove();
    }
}

async function initLoraTextAutocomplete(node, retries = 0) {
    const textWidget = findWidget(node, "lora_text");
    if (!textWidget) return;

    // widget.inputEl may not exist yet (lazy DOM creation)
    if (!textWidget.inputEl) {
        if (retries < 20) {
            setTimeout(() => initLoraTextAutocomplete(node, retries + 1), 500);
        }
        return;
    }

    // Probe Lora Manager availability
    try {
        await fetchJson(LM_BASE_MODELS_URL);
    } catch {
        return; // No Lora Manager — text input works, just no autocomplete
    }

    const ac = new LoraTextAutocomplete(textWidget.inputEl);
    node._loraTextAutocomplete = ac;
}

// --- Node Registration ---

app.registerExtension({
    name: "LoRAOptimizer.LoRAStackDynamic",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAStackDynamic") return;

        // Intercept mode and lora_count changes to update visibility
        for (const w of node.widgets || []) {
            if (w.name !== "mode" && w.name !== "lora_count") continue;
            interceptWidgetValue(w, () => updateVisibility(node));
        }

        // Initial visibility update — delay to ensure widgets are fully initialized
        setTimeout(() => {
            updateVisibility(node);
            // Initialize base model filter after visibility is set
            initBaseModelFilter(node);
            // Initialize autocomplete for lora_text widget
            initLoraTextAutocomplete(node);
        }, 100);

        // Cleanup on node removal
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            if (this._loraTextAutocomplete) {
                this._loraTextAutocomplete.destroy();
                this._loraTextAutocomplete = null;
            }
            if (origOnRemoved) origOnRemoved.call(this);
        };
    },
});

app.registerExtension({
    name: "LoRAOptimizer.LoRAConflictEditor",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAConflictEditor") return;
        // All 10 conflict_mode slots are always visible.
        // Unused slots (beyond the stack size) default to "auto" and are ignored.
    },
});
