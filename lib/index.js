import { Service } from "@deepseek-ai/cordis";
import z from "@deepseek-ai/schemastery";
import { BlockAssembler, contentHasImage, createUserMessage, deepFreeze, freezeMessage } from "@deepseek-ai/dsh-llm";
//#region lib/types/prompt.js
/**
* Model-facing recognition prompt and the stable logged recognition block.
* @module @deepseek-ai/dsh-vision-delegate/prompt
*/
/**
* The system prompt one recognition call sends to the vision model. Pinned
* verbatim: snapshots and end-to-end coverage depend on this exact text.
*/
const DEFAULT_RECOGNITION_PROMPT = "You are an image-recognition assistant for a text-only model that cannot see images.\nDescribe each attached image precisely and completely so the other model can work from your text alone:\n- Transcribe all visible text verbatim, including UI labels, code, logs, and numbers.\n- Describe layout, colors, and positions of key elements.\n- Note anything task-relevant, such as errors, diffs, charts, or selections.\nReply with the description only.";
/**
* Render the recognition block appended to a user message before it is logged.
* @param model - the vision model id that produced the description.
* @param description - the model's reply, already stripped to text.
* @returns the stable text block the main model reads and the transcript shows.
*/
function renderRecognitionText(model, description) {
	return `[attached image recognized by vision model "${model}"]\n${description}`;
}
//#endregion
//#region lib/types/index.js
/**
* Vision delegation: when the routed model cannot see images, an image-capable
* model describes user-attached images before they enter the step. The
* recognition text rides a separate plugin-sourced context message next to the
* image-carrying user message, so the text-only route works from text while
* the user's own message keeps only the image and the original text (the chat
* bubble stays clean). The image blocks stay in the log so the transcript
* keeps rendering them. A request-level `llm/stream` filter strips image
* blocks from any request whose route does not declare image input, so
* text-only continuation of an image-carrying session never bricks on the
* adapter's content guard.
*
* The recognition model is configured explicitly (`provider`/`model`) or
* auto-detected as the first registered model declaring `image` input. The
* plugin is inert until a usable route exists: admission gates fall back to
* their previous refusal, and no request is rewritten.
*
* @module @deepseek-ai/dsh-vision-delegate
*/
/** Default maximum output tokens for one recognition call. */
const DEFAULT_MAX_OUTPUT_TOKENS = 1024;
/** Producer id stamped on the recognition context message. */
const RECOGNITION_PRODUCER = "@deepseek-ai/dsh-vision-delegate";
/** Replace image blocks with attachment placeholders, walking tool-result nesting. */
function stripImageBlocks(blocks) {
	return blocks.map((block) => {
		if (block.type === "image") return {
			type: "text",
			text: `[image attachment ${String(block.attachment.attachmentId)}]`
		};
		if (block.type === "tool-result") return {
			...block,
			content: stripImageBlocks(block.content)
		};
		return block;
	});
}
/**
* Vision delegation service. Mounted on the root context, it registers the
* `agent/pre-step` rewrite (recognition) and the `llm/stream` filter (image
* strip for routes without image input).
*/
var VisionDelegate = class extends Service {
	static inject = ["llm"];
	static Config = z.object({
		enabled: z.boolean().default(true),
		provider: z.string(),
		model: z.string(),
		prompt: z.string().default(DEFAULT_RECOGNITION_PROMPT),
		maxOutputTokens: z.number().step(1).min(1).default(DEFAULT_MAX_OUTPUT_TOKENS)
	});
	enabled;
	explicitProvider;
	explicitModel;
	prompt;
	maxOutputTokens;
	visionRoute;
	visionRouteResolved = false;
	modelInfoCache = /* @__PURE__ */ new Map();
	constructor(ctx, config) {
		super(ctx, "visionDelegate");
		this.enabled = config.enabled ?? true;
		this.explicitProvider = config.provider;
		this.explicitModel = config.model;
		this.prompt = config.prompt ?? "You are an image-recognition assistant for a text-only model that cannot see images.\nDescribe each attached image precisely and completely so the other model can work from your text alone:\n- Transcribe all visible text verbatim, including UI labels, code, logs, and numbers.\n- Describe layout, colors, and positions of key elements.\n- Note anything task-relevant, such as errors, diffs, charts, or selections.\nReply with the description only.";
		this.maxOutputTokens = config.maxOutputTokens ?? 1024;
		if (this.explicitProvider === void 0 !== (this.explicitModel === void 0)) throw new Error("vision-delegate: provider and model must be configured together");
		if (!this.enabled) return;
		ctx.on("agent/pre-step", (payload, next) => this.delegatePreStep(payload, next));
		ctx.on("llm/stream", (options, next) => this.filterStream(options, next));
		ctx.on("llm/adapters-updated", () => {
			this.visionRouteResolved = false;
			this.modelInfoCache.clear();
		});
	}
	/**
	* Whether a usable vision route exists right now. An explicit route that
	* resolves without image input rejects: misconfiguration fails at the
	* earliest point that can observe it rather than at the first image.
	* @returns true when delegation can serve image admission.
	*/
	async isActive() {
		if (!this.enabled) return false;
		if (this.explicitProvider !== void 0) {
			await this.resolveVisionRoute();
			return true;
		}
		return await this.resolveVisionRoute() !== void 0;
	}
	/** Resolve and cache the vision route: explicit config, else first image-capable model. */
	async resolveVisionRoute() {
		if (this.visionRouteResolved) return this.visionRoute;
		const llm = this.ctx.llm;
		let route;
		if (this.explicitProvider !== void 0 && this.explicitModel !== void 0) {
			const info = await llm.resolveModelInfo(this.explicitProvider, this.explicitModel);
			if (info.inputModalities === void 0 || !info.inputModalities.includes("image")) throw new Error(`vision-delegate: configured model "${this.explicitModel}" on provider "${this.explicitProvider}" does not declare image input`);
			route = {
				provider: this.explicitProvider,
				model: this.explicitModel
			};
		} else for (const provider of llm.listProviders()) {
			let models;
			try {
				models = await llm.listModels(provider.id);
			} catch {
				continue;
			}
			for (const model of models) {
				let info;
				try {
					info = await llm.resolveModelInfo(provider.id, model.id);
				} catch {
					continue;
				}
				if (info.inputModalities !== void 0 && info.inputModalities.includes("image")) {
					route = {
						provider: provider.id,
						model: model.id
					};
					break;
				}
			}
			if (route !== void 0) break;
		}
		this.visionRoute = route;
		this.visionRouteResolved = true;
		return route;
	}
	/** Cached, failure-tolerant modality lookup for one routed request. */
	async modelInfo(provider, model) {
		const key = `${provider}\u0000${model}`;
		if (this.modelInfoCache.has(key)) return this.modelInfoCache.get(key);
		let info;
		try {
			info = await this.ctx.llm.resolveModelInfo(provider, model);
		} catch {
			info = void 0;
		}
		this.modelInfoCache.set(key, info);
		return info;
	}
	/** The route that serves the upcoming request: logged header config, else agent options. */
	routedModel(agent) {
		const header = agent.session.requestHeader()?.config;
		const provider = header?.provider ?? agent.options.provider;
		const model = header?.model ?? agent.options.model;
		if (provider === void 0 || model === void 0) return void 0;
		return {
			provider,
			model
		};
	}
	/**
	* Pre-step rewrite: for a route that cannot see images, every image-carrying
	* message is followed by a separate plugin-sourced context message carrying
	* the recognition text. The user message keeps its image and original text,
	* the model reads the description from the injected message, and the chat
	* surface renders it as a collapsed context row instead of inside the user
	* bubble. Recognition failure throws, failing the turn loudly instead of
	* silently continuing without the image.
	*/
	async delegatePreStep(payload, next) {
		if (!payload.messages.some((message) => contentHasImage(message.content))) return next();
		const routed = this.routedModel(payload.agent);
		const routedInfo = routed === void 0 ? void 0 : await this.modelInfo(routed.provider, routed.model);
		if (routedInfo !== void 0 && routedInfo.inputModalities?.includes("image")) return next();
		const decision = await next();
		if (decision.kind === "reject") return decision;
		const rewritten = [];
		for (const message of decision.messages) {
			const images = [];
			for (const block of message.content) if (block.type === "image") images.push(block);
			if (images.length === 0) {
				rewritten.push(message);
				continue;
			}
			payload.signal.throwIfAborted();
			const route = await this.resolveVisionRoute();
			if (route === void 0) throw new Error("vision-delegate: this message contains images but no image-capable model is available; add an image-capable model (or configure vision-delegate provider/model) to recognize images");
			const description = await this.describe(route, images, payload.signal);
			rewritten.push(message);
			rewritten.push(freezeMessage(createUserMessage({
				content: [{
					type: "text",
					text: description
				}],
				source: {
					kind: "plugin",
					plugin: RECOGNITION_PRODUCER,
					form: "notice",
					summary: `Image recognized by vision model "${route.model}"`
				}
			})));
		}
		return {
			...decision,
			messages: rewritten
		};
	}
	/** Ask the vision route to describe one message's images; throws on any failed finish. */
	async describe(route, images, signal) {
		const options = {
			provider: route.provider,
			model: route.model,
			system: this.prompt,
			messages: [createUserMessage({
				content: [...images],
				source: {
					kind: "plugin",
					plugin: RECOGNITION_PRODUCER
				}
			})],
			maxTokens: this.maxOutputTokens,
			signal
		};
		const assembler = new BlockAssembler();
		for await (const chunk of this.ctx.llm.stream(options)) assembler.push(chunk);
		signal.throwIfAborted();
		const finish = assembler.finish;
		if (finish.kind === "error" || finish.kind === "aborted") throw new Error(`vision-delegate: image recognition by model "${route.model}" failed: ${finish.failure.message}`);
		const text = assembler.blocks().filter((block) => block.type === "text").map((block) => block.text).join("\n");
		if (text.trim().length === 0) throw new Error(`vision-delegate: image recognition by model "${route.model}" returned no text`);
		return renderRecognitionText(route.model, text);
	}
	/**
	* Request-level image filter: a request routed to a model without image
	* input gets its image blocks replaced by attachment placeholders, keeping
	* the adjacent logged recognition text. Routes that declare image input,
	* and requests without images, pass through untouched.
	*/
	filterStream(options, next) {
		if (!options.messages.some((message) => contentHasImage(message.content))) return next();
		return (async function* () {
			const info = await this.modelInfo(options.provider, options.model);
			if (info !== void 0 && info.inputModalities?.includes("image")) {
				yield* next();
				return;
			}
			const messages = options.messages.map((message) => freezeMessage({
				...message,
				content: stripImageBlocks(message.content)
			}));
			const filtered = {
				...options,
				messages
			};
			yield* next(Object.isFrozen(options) ? deepFreeze(filtered) : filtered);
		}).call(this);
	}
};
//#endregion
export { DEFAULT_MAX_OUTPUT_TOKENS, VisionDelegate, VisionDelegate as default };
