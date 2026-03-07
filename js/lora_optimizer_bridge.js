import { app } from "/scripts/app.js";

/**
 * AutoTuner ↔ Optimizer Bridge
 *
 * 1. Widget sync: when the Optimizer runs with settings_source="from_autotuner",
 *    applied_settings in the UI message updates the Optimizer's knobs.
 *
 * 2. Opposing switches: when tuner_data is connected between an AutoTuner and
 *    an Optimizer, their mode switches stay in sync as opposites:
 *      Optimizer "from_autotuner" ↔ AutoTuner "merge"
 *      Optimizer "manual"         ↔ AutoTuner "tuning_only"
 */

const BRIDGE_WIDGETS = [
    "optimization_mode",
    "merge_quality",
    "sparsification",
    "sparsification_density",
    "dare_dampening",
    "auto_strength",
];

// Mapping between the two switches
const OPTIMIZER_TO_AUTOTUNER = {
    "from_autotuner": "merge",
    "manual": "tuning_only",
};
const AUTOTUNER_TO_OPTIMIZER = {
    "merge": "from_autotuner",
    "tuning_only": "manual",
};

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

/**
 * Find the AutoTuner node connected to this Optimizer's tuner_data input,
 * or the Optimizer node connected to this AutoTuner's tuner_data output.
 */
function findConnectedAutoTuner(optimizerNode) {
    // Find the tuner_data input slot index
    if (!optimizerNode.inputs) return null;
    for (let i = 0; i < optimizerNode.inputs.length; i++) {
        const input = optimizerNode.inputs[i];
        if (input.name === "tuner_data" && input.link != null) {
            const linkInfo = app.graph.links[input.link];
            if (linkInfo) {
                const sourceNode = app.graph.getNodeById(linkInfo.origin_id);
                if (sourceNode && sourceNode.comfyClass === "LoRAAutoTuner") {
                    return sourceNode;
                }
            }
        }
    }
    return null;
}

function findConnectedOptimizer(autotunerNode) {
    // Find tuner_data output slot
    if (!autotunerNode.outputs) return null;
    for (let i = 0; i < autotunerNode.outputs.length; i++) {
        const output = autotunerNode.outputs[i];
        if (output.name === "tuner_data" && output.links) {
            for (const linkId of output.links) {
                const linkInfo = app.graph.links[linkId];
                if (linkInfo) {
                    const targetNode = app.graph.getNodeById(linkInfo.target_id);
                    if (targetNode && targetNode.comfyClass === "LoRAOptimizer") {
                        return targetNode;
                    }
                }
            }
        }
    }
    return null;
}

// Guard against infinite recursion during sync
let _syncing = false;

function syncOppositeSwitch(sourceWidget, sourceNode, targetFinder, mapping) {
    if (_syncing) return;
    const targetNode = targetFinder(sourceNode);
    if (!targetNode) return;

    const targetWidgetName = sourceWidget.name === "settings_source" ? "output_mode" : "settings_source";
    const targetWidget = findWidget(targetNode, targetWidgetName);
    if (!targetWidget) return;

    const expectedValue = mapping[sourceWidget.value];
    if (expectedValue && targetWidget.value !== expectedValue) {
        _syncing = true;
        try {
            targetWidget.value = expectedValue;
        } finally {
            _syncing = false;
        }
        app.canvas?.setDirty?.(true, true);
    }
}

// --- LoRAOptimizer extension ---
app.registerExtension({
    name: "LoRAOptimizer.AutoTunerBridge",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAOptimizer") return;

        // Widget sync on execution (applied_settings from Python)
        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            if (origOnExecuted) {
                origOnExecuted.call(this, message);
            }

            if (!message || !message.applied_settings || !message.applied_settings[0]) {
                return;
            }

            let config;
            try {
                config = JSON.parse(message.applied_settings[0]);
            } catch {
                return;
            }

            for (const name of BRIDGE_WIDGETS) {
                const widget = findWidget(this, name);
                if (widget && config[name] !== undefined) {
                    widget.value = config[name];
                }
            }

            app.canvas?.setDirty?.(true, true);
        };

        // Opposing switch sync: settings_source → output_mode
        const settingsWidget = findWidget(node, "settings_source");
        if (settingsWidget) {
            const origCallback = settingsWidget.callback;
            settingsWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                syncOppositeSwitch(settingsWidget, node, findConnectedAutoTuner, OPTIMIZER_TO_AUTOTUNER);
            };
        }
    },
});

// --- LoRAAutoTuner extension ---
app.registerExtension({
    name: "LoRAOptimizer.AutoTunerOutputMode",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAAutoTuner") return;

        // Opposing switch sync: output_mode → settings_source
        const outputModeWidget = findWidget(node, "output_mode");
        if (outputModeWidget) {
            const origCallback = outputModeWidget.callback;
            outputModeWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                syncOppositeSwitch(outputModeWidget, node, findConnectedOptimizer, AUTOTUNER_TO_OPTIMIZER);
            };
        }
    },
});
