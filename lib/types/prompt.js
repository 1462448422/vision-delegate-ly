/**
 * Model-facing recognition prompt and the stable logged recognition block.
 * @module @deepseek-ai/dsh-vision-delegate/prompt
 */
/**
 * The system prompt one recognition call sends to the vision model. Pinned
 * verbatim: snapshots and end-to-end coverage depend on this exact text.
 */
export const DEFAULT_RECOGNITION_PROMPT = 'You are an image-recognition assistant for a text-only model that cannot see images.\n'
    + 'Describe each attached image precisely and completely so the other model can work from your text alone:\n'
    + '- Transcribe all visible text verbatim, including UI labels, code, logs, and numbers.\n'
    + '- Describe layout, colors, and positions of key elements.\n'
    + '- Note anything task-relevant, such as errors, diffs, charts, or selections.\n'
    + 'Reply with the description only.';
/**
 * Render the recognition block appended to a user message before it is logged.
 * @param model - the vision model id that produced the description.
 * @param description - the model's reply, already stripped to text.
 * @returns the stable text block the main model reads and the transcript shows.
 */
export function renderRecognitionText(model, description) {
    return `[attached image recognized by vision model "${model}"]\n${description}`;
}
//# sourceMappingURL=prompt.js.map