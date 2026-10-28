/** Lazy-loading embedding pipeline for local ONNX models.
 *
 * Factory-closure pattern: `createEmbedder(logger, options)` returns the embed API.
 * The model is downloaded on first use and cached by the transformers library.
 * Bi-encoder for vector embeddings — see `reranker.ts` for the cross-encoder. */

import { createHash } from "node:crypto"
import type { Logger } from "../../logger.js"
import { describeError } from "../../utils/describe-error.js"

export const DEFAULT_EMBEDDING_MODEL = "Xenova/bge-small-en-v1.5"

export const EMBEDDING_DIMENSIONS = 384

/** SHA-256 content hash — used to skip re-embedding unchanged chunks. */
export const contentHash = (text: string): string => createHash("sha256").update(text).digest("hex")

export type EmbedderOptions = {
  /** Hugging Face id of a feature-extraction model that transformers.js can
   *  run (ONNX weights, q8 variant available). */
  model?: string
  /** Width of the model's output vector. The vec0 tables are declared with it. */
  dimensions?: number
  /** Text prepended to a search query before embedding — asymmetric models
   *  such as the E5 family expect "query: ". Empty for bge-small-en. */
  queryPrefix?: string
  /** Text prepended to an indexed chunk before embedding — "passage: " for E5. */
  passagePrefix?: string
}

export type Embedder = ReturnType<typeof createEmbedder>

export const createEmbedder = (logger: Logger, options?: EmbedderOptions) => {
  const model = options?.model ?? DEFAULT_EMBEDDING_MODEL
  const dimensions = options?.dimensions ?? EMBEDDING_DIMENSIONS
  const queryPrefix = options?.queryPrefix ?? ""
  const passagePrefix = options?.passagePrefix ?? ""
  /** Identifies the vectors this embedder produces. Stored vectors made under
   *  another signature are unusable with this one and must be rebuilt. */
  const signature = `${model}|${String(dimensions)}|${passagePrefix}`

  type TransformersPipeline = Awaited<
    ReturnType<typeof import("@huggingface/transformers").pipeline<"feature-extraction">>
  >

  let pipelineInstance: TransformersPipeline | null = null
  // Guards against concurrent callers both triggering a model download —
  // the second caller awaits the first's promise instead of loading again.
  let pipelineLoading: Promise<TransformersPipeline> | null = null

  const getPipeline = async (): Promise<TransformersPipeline> => {
    if (pipelineInstance) return pipelineInstance
    if (pipelineLoading) return pipelineLoading

    pipelineLoading = (async () => {
      try {
        const startMs = performance.now()
        const { pipeline } = await import("@huggingface/transformers")
        const instance = await pipeline("feature-extraction", model, {
          dtype: "q8",
          // Pin ONNX to 1 thread — a small (~33M) model is too small for
          // intra-op parallelism to offset thread dispatch overhead, and the
          // default (0 = all cores) saturates the CPU on low-vCPU hosts.
          session_options: { intraOpNumThreads: 1, interOpNumThreads: 1 },
        })
        const elapsedMs = Math.round(performance.now() - startMs)
        logger.info("embedding model loaded", { model, dimensions, elapsedMs })
        pipelineInstance = instance
        return instance
      } catch (error) {
        logger.warn("embedding model failed to load", {
          model,
          error: describeError(error),
        })
        // Allow retry on next call (e.g. transient network failure during download)
        pipelineLoading = null
        throw error
      }
    })()

    return pipelineLoading
  }

  /** The transformers pipeline returns a Tensor whose `.data` is typed as
   *  `DataArray` (a union of typed arrays). Feature-extraction with mean
   *  pooling always produces Float32Array — guard at runtime rather than
   *  using an `as` cast. */
  const toFloat32Array = (data: unknown): Float32Array => {
    if (data instanceof Float32Array) return data
    throw new Error(`expected Float32Array from embedding pipeline, got ${typeof data}`)
  }

  const assertDimensions = (vector: Float32Array, inputCount: number): void => {
    if (vector.length === inputCount * dimensions) return
    throw new Error(
      `embedding model ${model} returned ${String(vector.length / inputCount)} dimensions, expected ${String(dimensions)} (EMBEDDING_DIMENSIONS)`,
    )
  }

  const embedWithPrefix = async (text: string, prefix: string): Promise<Float32Array> => {
    const pipe = await getPipeline()
    const output = await pipe(prefix + text, { pooling: "mean", normalize: true })
    const vector = new Float32Array(toFloat32Array(output.data))

    assertDimensions(vector, 1)
    return vector
  }

  /** Embed one indexed chunk (passage side). Mean-pooled, L2-normalized. */
  const embedText = (text: string): Promise<Float32Array> => embedWithPrefix(text, passagePrefix)

  /** Embed one search query (query side). Same vector space as embedText;
   *  only the prefix differs, for models that distinguish the two sides. */
  const embedQuery = (text: string): Promise<Float32Array> => embedWithPrefix(text, queryPrefix)

  /** Embed multiple chunks in a single pipeline call. Returns one Float32Array per input. */
  const embedBatch = async (texts: readonly string[]): Promise<Float32Array[]> => {
    if (texts.length === 0) return []
    const pipe = await getPipeline()
    const output = await pipe(
      texts.map((text) => passagePrefix + text),
      { pooling: "mean", normalize: true },
    )
    const data = toFloat32Array(output.data)

    assertDimensions(data, texts.length)
    // The pipeline returns all embeddings concatenated in a single flat array
    // (e.g. 3 inputs × 384 dims = 1152 floats). Slice out each input's embedding.
    return texts.map(
      (_text, index) => new Float32Array(data.slice(index * dimensions, (index + 1) * dimensions)),
    )
  }

  return { embedText, embedQuery, embedBatch, model, dimensions, signature }
}
