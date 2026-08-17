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

import { Context, Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent, PreStepDecision } from '@deepseek-ai/dsh-agent'
import {
  BlockAssembler,
  contentHasImage,
  createUserMessage,
  deepFreeze,
  freezeMessage,
} from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock,
  GenerateOptions,
  ImageBlock,
  LlmModelInfo,
  LlmResolvedModelInfo,
  StreamChunk,
  UserMessage,
} from '@deepseek-ai/dsh-llm'
import { DEFAULT_RECOGNITION_PROMPT, renderRecognitionText } from './prompt.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    /** Optional vision-delegation service; absent when the plugin is not mounted. */
    visionDelegate: VisionDelegate
  }
}

/** Default maximum output tokens for one recognition call. */
export const DEFAULT_MAX_OUTPUT_TOKENS = 1024

/** Producer id stamped on the recognition context message. */
const RECOGNITION_PRODUCER = '@deepseek-ai/dsh-vision-delegate'

/** Composition configuration. Every field is optional: the default is on with auto-detection. */
export interface Config {
  /** Mount the plugin but skip all delegation and filtering. */
  enabled?: boolean
  /** Explicit vision provider route; auto-detected when omitted. */
  provider?: string
  /** Explicit vision model id; auto-detected when omitted. */
  model?: string
  /** System prompt for the recognition call. */
  prompt?: string
  /** Maximum output tokens for one recognition call. */
  maxOutputTokens?: number
}

/** One resolved image-capable route. */
interface VisionRoute {
  provider: string
  model: string
}

/** Replace image blocks with attachment placeholders, walking tool-result nesting. */
function stripImageBlocks(blocks: readonly ContentBlock[]): ContentBlock[] {
  return blocks.map((block): ContentBlock => {
    if (block.type === 'image') {
      return { type: 'text', text: `[image attachment ${String(block.attachment.attachmentId)}]` }
    }
    if (block.type === 'tool-result') {
      return { ...block, content: stripImageBlocks(block.content) }
    }
    return block
  })
}

/**
 * Vision delegation service. Mounted on the root context, it registers the
 * `agent/pre-step` rewrite (recognition) and the `llm/stream` filter (image
 * strip for routes without image input).
 */
export class VisionDelegate extends Service {
  static inject = ['llm']

  static Config: z<Config> = z.object({
    enabled: z.boolean().default(true),
    provider: z.string(),
    model: z.string(),
    prompt: z.string().default(DEFAULT_RECOGNITION_PROMPT),
    maxOutputTokens: z.number().step(1).min(1).default(DEFAULT_MAX_OUTPUT_TOKENS),
  })

  private readonly enabled: boolean
  private readonly explicitProvider: string | undefined
  private readonly explicitModel: string | undefined
  private readonly prompt: string
  private readonly maxOutputTokens: number
  private visionRoute: VisionRoute | undefined
  private visionRouteResolved = false
  private readonly modelInfoCache = new Map<string, LlmResolvedModelInfo | undefined>()

  constructor(ctx: Context, config: Config) {
    super(ctx, 'visionDelegate')
    this.enabled = config.enabled ?? true
    this.explicitProvider = config.provider
    this.explicitModel = config.model
    this.prompt = config.prompt ?? DEFAULT_RECOGNITION_PROMPT
    this.maxOutputTokens = config.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS
    if ((this.explicitProvider === undefined) !== (this.explicitModel === undefined)) {
      throw new Error('vision-delegate: provider and model must be configured together')
    }
    if (!this.enabled) return
    ctx.on('agent/pre-step', (payload, next) => this.delegatePreStep(payload, next))
    ctx.on('llm/stream', (options, next) => this.filterStream(options, next))
    ctx.on('llm/adapters-updated', () => {
      this.visionRouteResolved = false
      this.modelInfoCache.clear()
    })
  }

  /**
   * Whether a usable vision route exists right now. An explicit route that
   * resolves without image input rejects: misconfiguration fails at the
   * earliest point that can observe it rather than at the first image.
   * @returns true when delegation can serve image admission.
   */
  async isActive(): Promise<boolean> {
    if (!this.enabled) return false
    if (this.explicitProvider !== undefined) {
      await this.resolveVisionRoute()
      return true
    }
    return await this.resolveVisionRoute() !== undefined
  }

  /** Resolve and cache the vision route: explicit config, else first image-capable model. */
  private async resolveVisionRoute(): Promise<VisionRoute | undefined> {
    if (this.visionRouteResolved) return this.visionRoute
    const llm = this.ctx.llm
    let route: VisionRoute | undefined
    if (this.explicitProvider !== undefined && this.explicitModel !== undefined) {
      const info = await llm.resolveModelInfo(this.explicitProvider, this.explicitModel)
      if (info.inputModalities === undefined || !info.inputModalities.includes('image')) {
        throw new Error(
          `vision-delegate: configured model "${this.explicitModel}" on provider `
          + `"${this.explicitProvider}" does not declare image input`,
        )
      }
      route = { provider: this.explicitProvider, model: this.explicitModel }
    } else {
      for (const provider of llm.listProviders()) {
        let models: readonly LlmModelInfo[]
        try {
          models = await llm.listModels(provider.id)
        } catch {
          // One route's catalog failure must not veto the others.
          continue
        }
        for (const model of models) {
          let info: LlmResolvedModelInfo
          try {
            info = await llm.resolveModelInfo(provider.id, model.id)
          } catch {
            // One model's broken metadata must not veto the route's others.
            continue
          }
          if (info.inputModalities !== undefined && info.inputModalities.includes('image')) {
            route = { provider: provider.id, model: model.id }
            break
          }
        }
        if (route !== undefined) break
      }
    }
    this.visionRoute = route
    this.visionRouteResolved = true
    return route
  }

  /** Cached, failure-tolerant modality lookup for one routed request. */
  private async modelInfo(provider: string, model: string): Promise<LlmResolvedModelInfo | undefined> {
    const key = `${provider}\u0000${model}`
    if (this.modelInfoCache.has(key)) return this.modelInfoCache.get(key)
    let info: LlmResolvedModelInfo | undefined
    try {
      info = await this.ctx.llm.resolveModelInfo(provider, model)
    } catch {
      // An unresolvable route cannot declare image input; treat it as text-only
      // so image blocks never reach an adapter that would reject them.
      info = undefined
    }
    this.modelInfoCache.set(key, info)
    return info
  }

  /** The route that serves the upcoming request: logged header config, else agent options. */
  private routedModel(agent: Agent): VisionRoute | undefined {
    const header = agent.session.requestHeader()?.config
    const provider = header?.provider ?? agent.options.provider
    const model = header?.model ?? agent.options.model
    if (provider === undefined || model === undefined) return undefined
    return { provider, model }
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
  private async delegatePreStep(
    payload: { agent: Agent; messages: UserMessage[]; signal: AbortSignal },
    next: () => Promise<PreStepDecision>,
  ): Promise<PreStepDecision> {
    if (!payload.messages.some(message => contentHasImage(message.content))) return next()
    const routed = this.routedModel(payload.agent)
    const routedInfo = routed === undefined ? undefined : await this.modelInfo(routed.provider, routed.model)
    // A native vision route sees the image itself; delegation is for the rest.
    if (routedInfo !== undefined && routedInfo.inputModalities?.includes('image')) return next()
    const decision = await next()
    if (decision.kind === 'reject') return decision
    const rewritten: UserMessage[] = []
    for (const message of decision.messages) {
      const images: ImageBlock[] = []
      for (const block of message.content) {
        if (block.type === 'image') images.push(block)
      }
      if (images.length === 0) {
        rewritten.push(message)
        continue
      }
      payload.signal.throwIfAborted()
      const route = await this.resolveVisionRoute()
      if (route === undefined) {
        throw new Error(
          'vision-delegate: this message contains images but no image-capable model is available; '
          + 'add an image-capable model (or configure vision-delegate provider/model) to recognize images',
        )
      }
      const description = await this.describe(route, images, payload.signal)
      // The user message stays verbatim (the image keeps rendering in the
      // bubble); the recognition text rides a separate context message so the
      // model reads it without the chat surface showing it inside the message.
      rewritten.push(message)
      rewritten.push(freezeMessage(createUserMessage({
        content: [{ type: 'text', text: description }],
        source: {
          kind: 'plugin',
          plugin: RECOGNITION_PRODUCER,
          form: 'notice',
          summary: `Image recognized by vision model "${route.model}"`,
        },
      })))
    }
    return { ...decision, messages: rewritten }
  }

  /** Ask the vision route to describe one message's images; throws on any failed finish. */
  private async describe(route: VisionRoute, images: readonly ImageBlock[], signal: AbortSignal): Promise<string> {
    const options: GenerateOptions = {
      provider: route.provider,
      model: route.model,
      system: this.prompt,
      messages: [createUserMessage({
        content: [...images],
        source: { kind: 'plugin', plugin: RECOGNITION_PRODUCER },
      })],
      maxTokens: this.maxOutputTokens,
      signal,
    }
    const assembler = new BlockAssembler()
    for await (const chunk of this.ctx.llm.stream(options)) {
      assembler.push(chunk)
    }
    signal.throwIfAborted()
    const finish = assembler.finish
    if (finish.kind === 'error' || finish.kind === 'aborted') {
      throw new Error(
        `vision-delegate: image recognition by model "${route.model}" failed: ${finish.failure.message}`,
      )
    }
    const text = assembler.blocks()
      .filter((block): block is Extract<ContentBlock, { type: 'text' }> => block.type === 'text')
      .map(block => block.text)
      .join('\n')
    if (text.trim().length === 0) {
      throw new Error(`vision-delegate: image recognition by model "${route.model}" returned no text`)
    }
    return renderRecognitionText(route.model, text)
  }

  /**
   * Request-level image filter: a request routed to a model without image
   * input gets its image blocks replaced by attachment placeholders, keeping
   * the adjacent logged recognition text. Routes that declare image input,
   * and requests without images, pass through untouched.
   */
  private filterStream(
    options: GenerateOptions,
    next: (finalOptions?: GenerateOptions) => AsyncIterable<StreamChunk>,
  ): AsyncIterable<StreamChunk> {
    if (!options.messages.some(message => contentHasImage(message.content))) return next()
    return (async function* (this: VisionDelegate) {
      const info = await this.modelInfo(options.provider, options.model)
      if (info !== undefined && info.inputModalities?.includes('image')) {
        yield* next()
        return
      }
      const messages = options.messages.map(message => freezeMessage({
        ...message,
        content: stripImageBlocks(message.content),
      }))
      const filtered: GenerateOptions = { ...options, messages }
      yield* next(Object.isFrozen(options) ? deepFreeze(filtered) : filtered)
    }).call(this)
  }
}

export default VisionDelegate
