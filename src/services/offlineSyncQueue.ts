/**
 * SecureFace AI (React Native) - Offline Sync Queue
 * Uses @react-native-community/netinfo for connectivity detection
 * instead of window.addEventListener('online'/'offline').
 */

import NetInfo from '@react-native-community/netinfo';
import {
  getPendingSyncPayloads,
  removeSyncPayload,
  updateSyncPayloadRetry,
  enqueueOfflinePayload,
  getSyncQueueCount,
} from './databaseManager';
import type {SyncPayload} from './databaseManager';
import {serializeEmbedding} from './vectorSearch';
import type {Identity, MatchResult} from '../types/face';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface SyncConfig {
  apiBaseUrl: string;
  authHeader: string | null;
  customHeaders: Record<string, string>;
  autoFlushIntervalMs: number;
  maxRetries: number;
  baseRetryDelayMs: number;
  verbose: boolean;
}

const DEFAULT_CONFIG: SyncConfig = {
  apiBaseUrl: '/api/v1/secureface',
  authHeader: null,
  customHeaders: {},
  autoFlushIntervalMs: 60_000,
  maxRetries: 5,
  baseRetryDelayMs: 1_000,
  verbose: true,
};

// ============================================================================
// STATE
// ============================================================================

let config: SyncConfig = {...DEFAULT_CONFIG};
let autoFlushTimer: ReturnType<typeof setInterval> | null = null;
let isFlushing = false;
let isOnline = true;
let netInfoUnsubscribe: (() => void) | null = null;

// ============================================================================
// NETWORK LISTENERS (NetInfo)
// ============================================================================

function initNetworkListeners(): void {
  netInfoUnsubscribe = NetInfo.addEventListener(state => {
    const wasOnline = isOnline;
    isOnline = state.isConnected === true;

    if (!wasOnline && isOnline && config.verbose) {
      console.log('[OfflineSync] Network online — triggering flush');
      flush().catch(console.error);
    }
    if (wasOnline && !isOnline && config.verbose) {
      console.log('[OfflineSync] Network offline — queuing payloads');
    }
  });
}

function removeNetworkListeners(): void {
  if (netInfoUnsubscribe) {
    netInfoUnsubscribe();
    netInfoUnsubscribe = null;
  }
}

// ============================================================================
// PAYLOAD CREATORS
// ============================================================================

function hashEmbedding(embedding: Float32Array): string {
  const serialized = serializeEmbedding(embedding);
  let hash = 0;
  const str = serialized.map(v => v.toFixed(4)).join(',');
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0;
  }
  return `emb_${Math.abs(hash).toString(16).padStart(8, '0')}`;
}

export async function enqueueVerificationEvent(
  matchResult: MatchResult | null,
  livenessScore: number,
  livenessPass: boolean,
  challengeType: string,
): Promise<string> {
  const payload: Record<string, unknown> = {
    event: 'face_verification',
    timestamp: new Date().toISOString(),
    result: {
      matched: matchResult?.isMatch ?? false,
      identityId: matchResult?.identity.id ?? null,
      similarity: matchResult?.similarity ?? 0,
    },
    liveness: {score: livenessScore, pass: livenessPass, challengeType},
    device: {platform: 'android', runtime: 'react-native'},
  };
  const id = enqueueOfflinePayload('verification', payload);
  if (isOnline && !isFlushing) flush().catch(console.error);
  return id;
}

export async function enqueueRegistrationEvent(
  identity: Identity,
  employeeId: string,
): Promise<string> {
  const payload: Record<string, unknown> = {
    event: 'face_registration',
    timestamp: new Date().toISOString(),
    identity: {
      id: identity.id,
      name: identity.name,
      employeeId,
      embeddingHash: hashEmbedding(identity.embedding),
      registeredAt: new Date(identity.registeredAt).toISOString(),
    },
    device: {platform: 'android', runtime: 'react-native'},
  };
  const id = enqueueOfflinePayload('registration', payload);
  if (isOnline && !isFlushing) flush().catch(console.error);
  return id;
}

// ============================================================================
// FLUSH ENGINE
// ============================================================================

function getEndpointUrl(type: SyncPayload['type']): string {
  const base = config.apiBaseUrl.replace(/\/+$/, '');
  return `${base}/events/${type}`;
}

async function sendPayload(payload: SyncPayload): Promise<boolean> {
  const url = getEndpointUrl(payload.type);
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...config.customHeaders,
  };
  if (config.authHeader) headers.Authorization = config.authHeader;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({syncId: payload.id, ...payload.payload}),
    });
    if (response.ok) return true;
    if (response.status >= 400 && response.status < 500 && response.status !== 429) return true;
    return false;
  } catch {
    return false;
  }
}

export async function flush(): Promise<{
  sent: number;
  failed: number;
  remaining: number;
}> {
  if (isFlushing || !isOnline) {
    return {sent: 0, failed: 0, remaining: getSyncQueueCount()};
  }

  isFlushing = true;
  let sent = 0;
  let failed = 0;

  try {
    const payloads = getPendingSyncPayloads().sort(
      (a, b) => a.createdAt - b.createdAt,
    );

    for (const payload of payloads) {
      if (payload.retryCount >= config.maxRetries) {
        removeSyncPayload(payload.id);
        failed++;
        continue;
      }

      if (payload.retryCount > 0) {
        const delay = config.baseRetryDelayMs * Math.pow(2, payload.retryCount);
        await new Promise<void>(r => setTimeout(() => r(), delay));
      }

      const success = await sendPayload(payload);
      if (success) {
        removeSyncPayload(payload.id);
        sent++;
      } else {
        updateSyncPayloadRetry(payload.id);
        failed++;
        if (!isOnline) break;
      }
    }

    return {sent, failed, remaining: getSyncQueueCount()};
  } finally {
    isFlushing = false;
  }
}

// ============================================================================
// AUTO-SYNC LIFECYCLE
// ============================================================================

export function startAutoSync(intervalMs?: number): void {
  stopAutoSync();
  initNetworkListeners();
  const interval = intervalMs ?? config.autoFlushIntervalMs;
  if (interval <= 0) return;
  autoFlushTimer = setInterval(() => {
    if (isOnline && !isFlushing) flush().catch(console.error);
  }, interval);
}

export function stopAutoSync(): void {
  if (autoFlushTimer) {
    clearInterval(autoFlushTimer);
    autoFlushTimer = null;
  }
  removeNetworkListeners();
}

export function configureSyncQueue(updates: Partial<SyncConfig>): void {
  config = {...config, ...updates};
}

export const OfflineSyncQueue = {
  enqueueVerificationEvent,
  enqueueRegistrationEvent,
  flush,
  startAutoSync,
  stopAutoSync,
  configureSyncQueue,
};

export default OfflineSyncQueue;
