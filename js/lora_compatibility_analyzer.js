import { app } from "/scripts/app.js";

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

app.registerExtension({
    name: "LoRAOptimizer.CompatibilityAnalyzer",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRACompatibilityAnalyzer") return;

        if (!node.properties) node.properties = {};
        if (!node.properties.createdGroupIds) node.properties.createdGroupIds = [];

        // Clean up child stack nodes when analyzer is deleted
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            const tracked = this.properties?.createdGroupIds || [];
            for (const id of tracked) {
                if (id != null) {
                    const child = app.graph.getNodeById(id);
                    if (child && child.comfyClass === "LoRAStackDynamic") {
                        app.graph.remove(child);
                    }
                }
            }
            if (origOnRemoved) origOnRemoved.call(this);
        };

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.call(this, message);

            const groups = message?.groups;
            if (!groups) return;

            const tracked = node.properties.createdGroupIds || [];
            const newTracked = [];

            // Update or create nodes for each group
            for (let i = 0; i < groups.length; i++) {
                const group = groups[i];
                let stackNode = null;

                // Try to reuse existing tracked node
                if (i < tracked.length && tracked[i] != null) {
                    stackNode = app.graph.getNodeById(tracked[i]);
                }

                if (stackNode && stackNode.comfyClass === "LoRAStackDynamic") {
                    // Update existing node
                    populateStackNode(stackNode, group, i);
                    newTracked.push(stackNode.id);
                } else {
                    // Create new node
                    const created = LiteGraph.createNode("LoRAStackDynamic");
                    if (!created) {
                        console.warn("[CompatibilityAnalyzer] LoRAStackDynamic node type not found");
                        continue;
                    }
                    app.graph.add(created);
                    created.pos = [
                        node.pos[0] + node.size[0] + 50,
                        node.pos[1] + i * 250,
                    ];
                    created.title = `Group ${i + 1} (Analyzer)`;

                    // Delay population to let widgets initialize (must exceed
                    // the 100ms initial-visibility timeout in lora_stack_dynamic.js)
                    setTimeout(() => populateStackNode(created, group, i), 200);
                    newTracked.push(created.id);
                }
            }

            // Remove orphaned nodes (groups shrank)
            for (let i = groups.length; i < tracked.length; i++) {
                if (tracked[i] != null) {
                    const orphan = app.graph.getNodeById(tracked[i]);
                    if (orphan && orphan.comfyClass === "LoRAStackDynamic") {
                        app.graph.remove(orphan);
                    }
                }
            }

            node.properties.createdGroupIds = newTracked;
            app.canvas?.setDirty?.(true, true);
        };
    },
});

function populateStackNode(stackNode, group, groupIndex) {
    const modeWidget = findWidget(stackNode, "input_mode");
    if (modeWidget) modeWidget.value = "text";

    const countWidget = findWidget(stackNode, "lora_count");
    if (countWidget) countWidget.value = group.loras.length;

    for (let i = 0; i < group.loras.length; i++) {
        const textWidget = findWidget(stackNode, `lora_name_text_${i + 1}`);
        if (textWidget) textWidget.value = group.loras[i].name;

        const strengthWidget = findWidget(stackNode, `strength_${i + 1}`);
        if (strengthWidget) strengthWidget.value = group.loras[i].strength;
    }
}
