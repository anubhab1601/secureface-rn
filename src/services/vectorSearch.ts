/**
 * SecureFace AI (React Native) - Vector Search Engine
 * 1:N Cosine Similarity Search — pure TypeScript math, no platform deps.
 */

import type { FaceEmbedding, Identity, MatchResult } from '../types/face';

export const DEFAULT_MATCH_THRESHOLD = 0.985; // Strict threshold for Geometric Euclidean distance
export const EMBEDDING_DIM = 11; // 11-dimensional geometric footprint

export function vectorNorm(v: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  return Math.sqrt(sum);
}

export function normalizeVector(v: Float32Array): Float32Array {
  const norm = vectorNorm(v);
  if (norm === 0) return new Float32Array(v.length);
  const result = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) {
    result[i] = v[i] / norm;
  }
  return result;
}

export function dotProduct(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Dimension mismatch: ${a.length} vs ${b.length}`);
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function euclideanDistance(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Dimension mismatch: ${a.length} vs ${b.length}`);
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  // For geometric footprints, Euclidean distance is much more discriminative than pure Cosine angle.
  // We convert it to a 0-1 similarity score to maintain compatibility with the rest of the app.
  const dist = euclideanDistance(a, b);
  return 1 / (1 + dist);
}

export function searchIdentity(
  queryEmbedding: FaceEmbedding,
  identities: Identity[],
  threshold: number = DEFAULT_MATCH_THRESHOLD,
): MatchResult | null {
  if (identities.length === 0) return null;

  let bestMatch: Identity | null = null;
  let bestSimilarity = -1;

  for (const identity of identities) {
    const similarity = cosineSimilarity(queryEmbedding, identity.embedding);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestMatch = identity;
    }
  }

  if (!bestMatch) return null;

  return {
    identity: bestMatch,
    similarity: bestSimilarity,
    isMatch: bestSimilarity >= threshold,
  };
}

export function serializeEmbedding(embedding: Float32Array): number[] {
  return Array.from(embedding);
}

export function deserializeEmbedding(data: number[]): Float32Array {
  return new Float32Array(data);
}
