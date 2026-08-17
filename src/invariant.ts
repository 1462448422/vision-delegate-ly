/** Package-owned invariant companion. @module @deepseek-ai/dsh-vision-delegate/invariant */

/* jscpd:ignore-start */
import type { Context } from '@deepseek-ai/cordis'
import type { InvariantInstaller } from '@deepseek-ai/dsh-invariants'

const PACKAGE_NAME = '@deepseek-ai/dsh-vision-delegate'

/** Cordis companion plugin name. */
export const name = 'vision-delegate-invariant'
/** Service required before the companion can reserve package ownership. */
export const inject = ['invariants']

/**
 * No runtime invariant: the recognition rewrite replaces the exact
 * `user/message` event the model request derives from, so the model-visible
 * recognition text already IS the durable record. The `llm/stream` image
 * strip is a live-request filter that only removes blocks the logged message
 * still carries; replay cannot reconstruct which request-time route served a
 * given historical step, so no offline relation can check it.
 */
const install: InvariantInstaller = () => {}

/** Register this package's invariant companion. */
export const apply = (ctx: Context): Promise<() => void> =>
  Promise.resolve(ctx.invariants.register(PACKAGE_NAME, install))
/* jscpd:ignore-end */
