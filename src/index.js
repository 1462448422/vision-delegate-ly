/**
 * Vision delegation: when the routed model cannot see images, an image-capable
 * model describes user-attached images before they enter the step. The
 * recognition text is appended to the durable user message (so the main model
 * works from text alone), while the image blocks stay in the log so the
 * transcript keeps rendering them. A request-level `llm/stream` filter strips
 * image blocks from any request whose route does not declare image input, so
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
import { Service } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
import { BlockAssembler, contentHasImage, createUserMessage, deepFreeze, freezeMessage, } from '@deepseek-ai/dsh-llm';
import { DEFAULT_RECOGNITION_PROMPT, renderRecognitionText } from "./prompt.js";
/** Default maximum output tokens for one recognition call. */
export const DEFAULT_MAX_OUTPUT_TOKENS = 1024;
/** Replace image blocks with attachment placeholders, walking tool-result nesting. */
function stripImageBlocks(blocks) {
    return blocks.map((block) => {
        if (block.type === 'image') {
            return { type: 'text', text: `[image attachment ${String(block.attachment.attachmentId)}]` };
        }
        if (block.type === 'tool-result') {
            return { ...block, content: stripImageBlocks(block.content) };
        }
        return block;
    });
}
/**
 * Vision delegation service. Mounted on the root context, it registers the
 * `agent/pre-step` rewrite (recognition) and the `llm/stream` filter (image
 * strip for routes without image input).
 */
export class VisionDelegate extends Service {
    static inject = ['llm'];
    static Config = z.object({
        enabled: z.boolean().default(true),
        provider: z.string(),
        model: z.string(),
        prompt: z.string().default(DEFAULT_RECOGNITION_PROMPT),
        maxOutputTokens: z.number().step(1).min(1).default(DEFAULT_MAX_OUTPUT_TOKENS),
    });
    enabled;
    explicitProvider;
    explicitModel;
    prompt;
    maxOutputTokens;
    visionRoute;
    visionRouteResolved = false;
    modelInfoCache = new Map();
    constructor(ctx, config) {
        super(ctx, 'visionDelegate');
        this.enabled = config.enabled ?? true;
        this.explicitProvider = config.provider;
        this.explicitModel = config.model;
        this.prompt = config.prompt ?? DEFAULT_RECOGNITION_PROMPT;
        this.maxOutputTokens = config.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS;
        if ((this.explicitProvider === undefined) !== (this.explicitModel === undefined)) {
            throw new Error('vision-delegate: provider and model must be configured together');
        }
        if (!this.enabled)
            return;
        ctx.on('agent/pre-step', (payload, next) => this.delegatePreStep(payload, next));
        ctx.on('llm/stream', (options, next) => this.filterStream(options, next));
        ctx.on('llm/adapters-updated', () => {
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
        if (!this.enabled)
            return false;
        if (this.explicitProvider !== undefined) {
            await this.resolveVisionRoute();
            return true;
        }
        return await this.resolveVisionRoute() !== undefined;
    }
    /** Resolve and cache the vision route: explicit config, else first image-capable model. */
    async resolveVisionRoute() {
        if (this.visionRouteResolved)
            return this.visionRoute;
        const llm = this.ctx.llm;
        let route;
        if (this.explicitProvider !== undefined && this.explicitModel !== undefined) {
            const info = await llm.resolveModelInfo(this.explicitProvider, this.explicitModel);
            if (info.inputModalities === undefined || !info.inputModalities.includes('image')) {
                throw new Error(`vision-delegate: configured model "${this.explicitModel}" on provider `
                    + `"${this.explicitProvider}" does not declare image input`);
            }
            route = { provider: this.explicitProvider, model: this.explicitModel };
        }
        else {
            for (const provider of llm.listProviders()) {
                let models;
                try {
                    models = await llm.listModels(provider.id);
                }
                catch {
                    // One route's catalog failure must not veto the others.
                    continue;
                }
                for (const model of models) {
                    let info;
                    try {
                        info = await llm.resolveModelInfo(provider.id, model.id);
                    }
                    catch {
                        // One model's broken metadata must not veto the route's others.
                        continue;
                    }
                    if (info.inputModalities !== undefined && info.inputModalities.includes('image')) {
                        route = { provider: provider.id, model: model.id };
                        break;
                    }
                }
                if (route !== undefined)
                    break;
            }
        }
        this.visionRoute = route;
        this.visionRouteResolved = true;
        return route;
    }
    /** Cached, failure-tolerant modality lookup for one routed request. */
    async modelInfo(provider, model) {
        const key = `${provider}\u0000${model}`;
        if (this.modelInfoCache.has(key))
            return this.modelInfoCache.get(key);
        let info;
        try {
            info = await this.ctx.llm.resolveModelInfo(provider, model);
        }
        catch {
            // An unresolvable route cannot declare image input; treat it as text-only
            // so image blocks never reach an adapter that would reject them.
            info = undefined;
        }
        this.modelInfoCache.set(key, info);
        return info;
    }
    /** The route that serves the upcoming request: logged header config, else agent options. */
    routedModel(agent) {
        const header = agent.session.requestHeader()?.config;
        const provider = header?.provider ?? agent.options.provider;
        const model = header?.model ?? agent.options.model;
        if (provider === undefined || model === undefined)
            return undefined;
        return { provider, model };
    }
    /**
     * Pre-step rewrite: routes that cannot see images get a recognition text
     * block appended to every image-carrying message before it is logged and
     * sent. Recognition failure throws, failing the turn loudly instead of
     * silently continuing without the image.
     */
    async delegatePreStep(payload, next) {
        if (!payload.messages.some(message => contentHasImage(message.content)))
            return next();
        const routed = this.routedModel(payload.agent);
        const routedInfo = routed === undefined ? undefined : await this.modelInfo(routed.provider, routed.model);
        // A native vision route sees the image itself; delegation is for the rest.
        if (routedInfo !== undefined && routedInfo.inputModalities?.includes('image'))
            return next();
        const decision = await next();
        if (decision.kind === 'reject')
            return decision;
        const rewritten = [];
        for (const message of decision.messages) {
            const images = [];
            for (const block of message.content) {
                if (block.type === 'image')
                    images.push(block);
            }
            if (images.length === 0) {
                rewritten.push(message);
                continue;
            }
            payload.signal.throwIfAborted();
            const route = await this.resolveVisionRoute();
            if (route === undefined) {
                throw new Error('vision-delegate: this message contains images but no image-capable model is available; '
                    + 'add an image-capable model (or configure vision-delegate provider/model) to recognize images');
            }
            const description = await this.describe(route, images, payload.signal);
            rewritten.push(freezeMessage({
                ...message,
                content: [...message.content, { type: 'text', text: description }],
            }));
        }
        return { ...decision, messages: rewritten };
    }
    /** Ask the vision route to describe one message's images; throws on any failed finish. */
    async describe(route, images, signal) {
        const options = {
            provider: route.provider,
            model: route.model,
            system: this.prompt,
            messages: [createUserMessage({
                    content: [...images],
                    source: { kind: 'plugin', plugin: 'vision-delegate' },
                })],
            maxTokens: this.maxOutputTokens,
            signal,
        };
        const assembler = new BlockAssembler();
        for await (const chunk of this.ctx.llm.stream(options)) {
            assembler.push(chunk);
        }
        signal.throwIfAborted();
        const finish = assembler.finish;
        if (finish.kind === 'error' || finish.kind === 'aborted') {
            throw new Error(`vision-delegate: image recognition by model "${route.model}" failed: ${finish.failure.message}`);
        }
        const text = assembler.blocks()
            .filter((block) => block.type === 'text')
            .map(block => block.text)
            .join('\n');
        if (text.trim().length === 0) {
            throw new Error(`vision-delegate: image recognition by model "${route.model}" returned no text`);
        }
        return renderRecognitionText(route.model, text);
    }
    /**
     * Request-level image filter: a request routed to a model without image
     * input gets its image blocks replaced by attachment placeholders, keeping
     * the adjacent logged recognition text. Routes that declare image input,
     * and requests without images, pass through untouched.
     */
    filterStream(options, next) {
        if (!options.messages.some(message => contentHasImage(message.content)))
            return next();
        return (async function* () {
            const info = await this.modelInfo(options.provider, options.model);
            if (info !== undefined && info.inputModalities?.includes('image')) {
                yield* next();
                return;
            }
            const messages = options.messages.map(message => freezeMessage({
                ...message,
                content: stripImageBlocks(message.content),
            }));
            const filtered = { ...options, messages };
            yield* next(Object.isFrozen(options) ? deepFreeze(filtered) : filtered);
        }).call(this);
    }
}
export default VisionDelegate;
//# sourceMappingURL=index.js.map