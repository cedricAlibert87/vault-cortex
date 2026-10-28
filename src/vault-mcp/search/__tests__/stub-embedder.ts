import type { Embedder } from "../embedder.js"
import { EMBEDDING_DIMENSIONS } from "../embedder.js"

type EmbedderStub = Pick<Embedder, "embedText" | "embedBatch">

/** Completes a two-method test stub into the full Embedder shape: queries embed
 *  like passages (no prefixes), under a fixed model signature. The stub's own
 *  members keep their mock types so tests can still drive them. */
export const stubEmbedder = <Stub extends EmbedderStub>(
  stub: Stub,
): Stub & Omit<Embedder, keyof EmbedderStub> => ({
  ...stub,
  embedQuery: stub.embedText,
  model: "stub-model",
  dimensions: EMBEDDING_DIMENSIONS,
  signature: `stub-model|${String(EMBEDDING_DIMENSIONS)}|`,
})
