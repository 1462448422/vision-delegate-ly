/**
 * Vision delegation over the REAL agent loop with scripted adapters:
 * recognition context injection for text-only routes, pass-through for native
 * vision routes, loud failure, auto-detection, and the request-level image
 * filter.
 */

import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { AttachmentId } from '@deepseek-ai/dsh-attachment'
import {
  createUserMessage,
  deepFreeze,
  LlmAdapter,
  LlmRuntime,
  markAgentLoopRequest,
} from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock,
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ModelModality,
  StreamChunk,
  UserMessage,
} from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import VisionDelegate from '../src/index.ts'
import { DEFAULT_RECOGNITION_PROMPT } from '../src/prompt.ts'

type ScriptEntry = StreamChunk[] | Error

/** Small request-recording adapter with a declared input modality set. */
class ScriptedAdapter extends LlmAdapter {
  readonly requests: GenerateOptions[] = []

  constructor(
    private readonly script: ScriptEntry[],
    private readonly modalities?: readonly ModelModality[],
  ) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: provider }
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve([{
      provider,
      id: provider,
      name: `${provider} model`,
      ...this.modalities === undefined ? {} : { inputModalities: this.modalities },
    }])
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      ...this.modalities === undefined ? {} : { inputModalities: this.modalities },
    })
  }

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    const entry = this.script.shift()
    if (entry === undefined) throw new Error('ScriptedAdapter: script exhausted')
    if (entry instanceof Error) throw entry
    for (const chunk of entry) yield chunk
  }
}

/** One successful text response. */
function textResponse(text: string): StreamChunk[] {
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'block-end', index: 0, block: { type: 'text', text } },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
}

/** One failed response, as the adapter contract reports it. */
function errorResponse(message: string): StreamChunk[] {
  return [{
    type: 'finish',
    reason: { kind: 'error', failure: { message, code: 'UPSTREAM' } },
  }]
}

/** A minimal durable image block. */
function imageBlock(): ContentBlock {
  return {
    type: 'image',
    attachment: {
      attachmentId: AttachmentId(`sha256:${'a'.repeat(64)}`),
      mediaType: 'image/png',
      bytes: 4,
      width: 2,
      height: 2,
    },
  }
}

/** Complete request text as one string for ordering assertions. */
function requestText(request: GenerateOptions): string {
  return request.messages
    .flatMap(message => message.content)
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('\n')
}

/** All image blocks in a request, walking tool-result nesting. */
function requestImages(request: GenerateOptions): ContentBlock[] {
  return request.messages
    .flatMap(message => message.content)
    .filter(block => block.type === 'image')
}

interface LoopHarness {
  readonly ctx: Context
  readonly agent: Agent
  readonly textAdapter: ScriptedAdapter
  readonly errors: string[]
}

const contexts: Context[] = []

afterEach(async () => {
  await Promise.allSettled(contexts.splice(0).map(context => context.fiber.dispose()))
})

/** A real loop with a text-only route and error observation; vision adapters join per test. */
async function loopHarness(textScript: ScriptEntry[] = [], agentRoute?: { provider: string; model: string }): Promise<LoopHarness> {
  const ctx = new Context()
  contexts.push(ctx)
  await mountAgentLoopTestDependencies(ctx)
  await ctx.plugin(AgentLoop, { agents: [] })
  const textAdapter = new ScriptedAdapter(textScript, ['text'])
  ctx.llm.registerAdapter(['text'], textAdapter)
  const errors: string[] = []
  ctx.on('agent/error', ({ error }) => {
    errors.push(error instanceof Error ? error.message : String(error))
  })
  const agent = ctx.agentLoop.create(SessionId(`vision-session-${Math.random()}`), agentRoute ?? {
    provider: 'text',
    model: 'text',
  })
  return { ctx, agent, textAdapter, errors }
}

function userMessageWithImage(): UserMessage {
  return createUserMessage({
    content: [{ type: 'text', text: 'what is this?' }, imageBlock()],
    source: { kind: 'user' },
  })
}

describe('vision delegation over the real loop', () => {
  it('injects recognition as a context message, keeps the image logged, and strips it from the text-only request', async () => {
    const { ctx, agent, textAdapter } = await loopHarness([textResponse('ok')])
    const visionAdapter = new ScriptedAdapter([textResponse('a red circle')], ['text', 'image'])
    ctx.llm.registerAdapter(['vision'], visionAdapter)
    await ctx.plugin(VisionDelegate)

    agent.followup(userMessageWithImage())
    await agent.whenIdle()

    // One recognition call on the vision route, carrying the image blocks and the pinned prompt.
    expect(visionAdapter.requests).toHaveLength(1)
    const recognition = visionAdapter.requests[0]!
    expect(recognition.system).toBe(DEFAULT_RECOGNITION_PROMPT)
    expect(requestImages(recognition)).toHaveLength(1)

    // The conversation request on the text-only route never sees an image block,
    // and reads the placeholder plus the recognition text instead.
    const conversation = textAdapter.requests[0]!
    expect(requestImages(conversation)).toHaveLength(0)
    expect(requestText(conversation)).toContain('[image attachment sha256:')
    expect(requestText(conversation)).toContain('a red circle')

    // The durable user message keeps the image (transcript display); the
    // recognition text rides a separate plugin-sourced context message so the
    // user bubble stays clean.
    const logged = agent.session.events.filter(event => event.type === 'user/message')
    expect(logged).toHaveLength(2)
    const userContent = (logged[0]!.data as { content: ContentBlock[] }).content
    expect(userContent.some(block => block.type === 'image')).toBe(true)
    expect(userContent.filter(block => block.type === 'text').map(block => block.text).join('\n'))
      .not.toContain('recognized by vision model')
    const context = logged[1]!.data as { content: ContentBlock[]; source: Record<string, unknown> }
    expect(context.content.filter(block => block.type === 'text').map(block => block.text).join('\n'))
      .toContain('recognized by vision model "vision"')
    expect(context.source).toMatchObject({
      kind: 'plugin',
      plugin: '@deepseek-ai/dsh-vision-delegate',
      form: 'notice',
    })
  })

  it('leaves native vision routes untouched', async () => {
    const { ctx, agent, textAdapter } = await loopHarness([], { provider: 'vision', model: 'vision' })
    const visionAdapter = new ScriptedAdapter([textResponse('saw it myself')], ['text', 'image'])
    ctx.llm.registerAdapter(['vision'], visionAdapter)
    await ctx.plugin(VisionDelegate)

    agent.followup(userMessageWithImage())
    await agent.whenIdle()

    // The single vision call is the conversation itself, with the image intact.
    expect(visionAdapter.requests).toHaveLength(1)
    expect(requestImages(visionAdapter.requests[0]!)).toHaveLength(1)
    expect(requestText(visionAdapter.requests[0]!)).not.toContain('recognized by vision model')
    expect(textAdapter.requests).toHaveLength(0)
  })

  it('fails the turn loudly when recognition fails', async () => {
    const { ctx, agent, textAdapter, errors } = await loopHarness([textResponse('never reached')])
    const visionAdapter = new ScriptedAdapter([errorResponse('vision backend down')], ['text', 'image'])
    ctx.llm.registerAdapter(['vision'], visionAdapter)
    await ctx.plugin(VisionDelegate)

    agent.followup(userMessageWithImage())
    await agent.whenIdle()

    expect(errors.join('\n')).toContain('image recognition by model "vision" failed: vision backend down')
    expect(agent.session.events.some(event => event.type === 'user/message')).toBe(false)
    expect(textAdapter.requests).toHaveLength(0)
  })

  it('fails the turn loudly when no image-capable model exists', async () => {
    const { ctx, agent, errors } = await loopHarness([textResponse('never reached')])
    await ctx.plugin(VisionDelegate)

    agent.followup(userMessageWithImage())
    await agent.whenIdle()

    expect(errors.join('\n')).toContain('no image-capable model is available')
  })

  it('rejects an explicitly configured route without image input', async () => {
    const { ctx, agent, errors } = await loopHarness([textResponse('never reached')])
    await ctx.plugin(VisionDelegate, { provider: 'text', model: 'text' })
    expect(ctx.get('visionDelegate')).toBeDefined()
    await expect(ctx.get('visionDelegate')!.isActive()).rejects.toThrow(/does not declare image input/)

    agent.followup(userMessageWithImage())
    await agent.whenIdle()
    expect(errors.join('\n')).toContain('does not declare image input')
  })

  it('does nothing while disabled', async () => {
    const { ctx, agent, textAdapter } = await loopHarness([textResponse('ok')])
    const visionAdapter = new ScriptedAdapter([], ['text', 'image'])
    ctx.llm.registerAdapter(['vision'], visionAdapter)
    await ctx.plugin(VisionDelegate, { enabled: false })

    agent.followup(userMessageWithImage())
    await agent.whenIdle()

    expect(visionAdapter.requests).toHaveLength(0)
    expect(requestText(textAdapter.requests[0]!)).not.toContain('recognized by vision model')
  })

  it('unregisters with its fiber', async () => {
    const { ctx } = await loopHarness()
    const fiber = await ctx.plugin(VisionDelegate)
    expect(ctx.get('visionDelegate')).toBeDefined()
    await fiber.dispose()
    expect(ctx.get('visionDelegate')).toBeUndefined()
  })
})

describe('vision route resolution', () => {
  it('auto-detects the first image-capable route and re-resolves on registry changes', async () => {
    const { ctx } = await loopHarness()
    await ctx.plugin(VisionDelegate)
    const service = ctx.get('visionDelegate')!
    expect(await service.isActive()).toBe(false)

    ctx.llm.registerAdapter(['vision'], new ScriptedAdapter([], ['text', 'image']))
    expect(await service.isActive()).toBe(true)
  })
})

describe('request-level image filter', () => {
  it('strips image blocks for text-only routes, keeping placeholders and text', async () => {
    const ctx = new Context()
    contexts.push(ctx)
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(VisionDelegate)
    const adapter = new ScriptedAdapter([textResponse('done')], ['text'])
    ctx.llm.registerAdapter(['text'], adapter)

    const message = createUserMessage({
      content: [{ type: 'text', text: 'before' }, imageBlock(), { type: 'text', text: 'after' }],
      source: { kind: 'user' },
    })
    for await (const _chunk of ctx.llm.stream({ provider: 'text', model: 'text', messages: [message] })) { /* drain */ }

    const sent = adapter.requests[0]!
    expect(requestImages(sent)).toHaveLength(0)
    const text = requestText(sent)
    expect(text).toContain('[image attachment sha256:')
    expect(text).toContain('before')
    expect(text).toContain('after')
  })

  it('passes images through to vision routes', async () => {
    const ctx = new Context()
    contexts.push(ctx)
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(VisionDelegate)
    const adapter = new ScriptedAdapter([textResponse('done')], ['text', 'image'])
    ctx.llm.registerAdapter(['vision'], adapter)

    const message = createUserMessage({
      content: [imageBlock()],
      source: { kind: 'user' },
    })
    for await (const _chunk of ctx.llm.stream({ provider: 'vision', model: 'vision', messages: [message] })) { /* drain */ }

    expect(requestImages(adapter.requests[0]!)).toHaveLength(1)
  })

  it('rewrites frozen loop-built requests without mutation', async () => {
    const ctx = new Context()
    contexts.push(ctx)
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(VisionDelegate)
    const adapter = new ScriptedAdapter([textResponse('done')], ['text'])
    ctx.llm.registerAdapter(['text'], adapter)

    const message = createUserMessage({
      content: [imageBlock()],
      source: { kind: 'user' },
    })
    const request = markAgentLoopRequest(deepFreeze({
      provider: 'text',
      model: 'text',
      messages: [message],
    }))
    for await (const _chunk of ctx.llm.stream(request)) { /* drain */ }

    expect(requestImages(request)).toHaveLength(1)
    expect(requestImages(adapter.requests[0]!)).toHaveLength(0)
  })
})
