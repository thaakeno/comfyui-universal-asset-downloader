import { app } from "/scripts/app.js";

app.registerExtension({
	name: "Comfy.UniversalAssetDownloader",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "UniversalAssetDownloader") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (message.text) {
					const downloadMessage = this.widgets.find(w => w.name === "download_message");
					downloadMessage.value = message.text[0];
				}
			};

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				const downloadWidget = this.addWidget("button", "Download Now", "download", () => {
                    const node = this;
					app.queuePrompt(0, 1, { [node.id]: node });
				});

				this.addWidget("text", "Status:", "", "download_message", { multiline: true });
			};
		}
	},
});