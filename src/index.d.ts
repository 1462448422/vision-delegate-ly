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
import { Context, Service } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
declare module '@deepseek-ai/cordis' {
    interface Context {
        /** Optional vision-delegation service; absent when the plugin is not mounted. */
        visionDelegate: VisionDelegate;
    }
}
/** Default maximum output tokens for one recognition call. */
export declare const DEFAULT_MAX_OUTPUT_TOKENS = 1024;
/** Composition configuration. Every field is optional: the default is on with auto-detection. */
export interface Config {
    /** Mount the plugin but skip all delegation and filtering. */
    enabled?: boolean;
    /** Explicit vision provider route; auto-detected when omitted. */
    provider?: string;
    /** Explicit vision model id; auto-detected when omitted. */
    model?: string;
    /** System prompt for the recognition call. */
    prompt?: string;
    /** Maximum output tokens for one recognition call. */
    maxOutputTokens?: number;
}
/**
 * Vision delegation service. Mounted on the root context, it registers the
 * `agent/pre-step` rewrite (recognition) and the `llm/stream` filter (image
 * strip for routes without image input).
 */
export declare class VisionDelegate extends Service {
    static inject: string[];
    static Config: z<Config>;
    private readonly enabled;
    private readonly explicitProvider?;
    private readonly explicitModel?;
    private readonly prompt;
    private readonly maxOutputTokens;
    private visionRoute;
    private visionRouteResolved;
    private readonly modelInfoCache;
    constructor(ctx: Context, config: Config);
    /**
     * Whether a usable vision route exists right now. An explicit route that
     * resolves without image input rejects: misconfiguration fails at the
     * earliest point that can observe it rather than at the first image.
     * @returns true when delegation can serve image admission.
     */
    isActive(): Promise<boolean>;
    /** Resolve and cache the vision route: explicit config, else first image-capable model. */
    private resolveVisionRoute;
    /** Cached, failure-tolerant modality lookup for one routed request. */
    private modelInfo;
    /** The route that serves the upcoming request: logged header config, else agent options. */
    private routedModel;
    /**
     * Pre-step rewrite: routes that cannot see images get a recognition text
     * block appended to every image-carrying message before it is logged and
     * sent. Recognition failure throws, failing the turn loudly instead of
     * silently continuing without the image.
     */
    private delegatePreStep;
    /** Ask the vision route to describe one message's images; throws on any failed finish. */
    private describe;
    /**
     * Request-level image filter: a request routed to a model without image
     * input gets its image blocks replaced by attachment placeholders, keeping
     * the adjacent logged recognition text. Routes that declare image input,
     * and requests without images, pass through untouched.
     */
    private filterStream;
}
export default VisionDelegate;
//# sourceMappingURL=index.d.ts.map