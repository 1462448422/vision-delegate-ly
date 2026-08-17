# @deepseek-ai/dsh-vision-delegate

English | [中文](README.zh.md)

Vision delegation for text-only model routes. When the routed model does not declare image input, an image-capable model (explicitly configured, or auto-detected as the first registered model declaring `image` input) describes user-attached images before the message enters its step; the recognition text rides a separate plugin-sourced context message next to the image-carrying user message, while the image blocks stay in the log so the transcript keeps rendering them.

## Composition

```yaml
- id: vision-delegate
  name: '@deepseek-ai/dsh-vision-delegate'
```

Every config key is optional. The zero-config default is on, with the vision route auto-detected.

```yaml
- id: vision-delegate
  name: '@deepseek-ai/dsh-vision-delegate'
  config:
    enabled: true
    provider: acme-gateway        # explicit vision route (with model); both or neither
    model: qwen3.7-plus
    prompt: '...'                 # recognition system prompt
    maxOutputTokens: 1024
```

## Behavior

- **Recognition (`agent/pre-step`)** — an image-carrying message whose route cannot see images gets one recognition call per message before the step is logged. The call sends all of that message's image blocks to the vision route with the configured system prompt; the reply becomes a separate `[attached image recognized by vision model "..."]` context message (plugin-sourced, `notice` form), so the model reads the description while the user bubble keeps only the image and the original text. Recognition failure throws, so the turn fails loudly instead of silently continuing without the image.
- **Request filter (`llm/stream`)** — any request routed to a model without image input has its image blocks replaced by `[image attachment <id>]` placeholders, keeping the adjacent logged recognition text. Requests without images, and routes that declare image input, pass through untouched.
- **Inert when unusable** — with auto-detection and no image-capable model registered, admission gates keep their previous refusals and no request is rewritten. An explicit route that does not declare image input rejects at the first admission check rather than at the first image.
- **Native vision untouched** — a session whose routed model already sees images is never delegated: the model reads the image directly.

## Model Experience

### Recognition call

#### What the model sees

The vision model receives only the attached image blocks and the recognition system prompt; it has no session history, tools, or workspace context.

#### Token effect

One auxiliary call per image-carrying message, bounded by `maxOutputTokens` (default 1024), on the vision route. The recognition text rides a separate context message and is resent with the history on every later request until compaction shadows it.

#### KV Cache effect

The main route's requests are unchanged in shape; the context message follows the user message in place. The auxiliary call reuses no conversation cache.

## Known Limitations and Deferred Work

- **Stale-header window after a model switch** — delegation consults the last logged request header, falling back to agent options. In the first request right after switching away from a vision model, an image may bypass recognition; the request filter still strips it, and resending the image re-delegates. A first request right after switching *to* a vision model may delegate redundantly.
- **Per-message granularity** — all images in one message are described by one recognition call; the description does not attribute text to individual images.
- **No GUI configuration** — the plugin composes from cordis.yml only; the settings surface has no card for it yet. `read_image` keeps its own strict route gate.
- **No independent quality check** — the recognition reply is logged verbatim; nothing validates that it actually describes the images.
