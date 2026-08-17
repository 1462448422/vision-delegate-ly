/**
 * Model-facing recognition prompt and the stable logged recognition block.
 * @module @deepseek-ai/dsh-vision-delegate/prompt
 */
/**
 * The system prompt one recognition call sends to the vision model. Pinned
 * verbatim: snapshots and end-to-end coverage depend on this exact text.
 */
export declare const DEFAULT_RECOGNITION_PROMPT: string;
/**
 * Render the recognition block appended to a user message before it is logged.
 * @param model - the vision model id that produced the description.
 * @param description - the model's reply, already stripped to text.
 * @returns the stable text block the main model reads and the transcript shows.
 */
export declare function renderRecognitionText(model: string, description: string): string;
//# sourceMappingURL=prompt.d.ts.map