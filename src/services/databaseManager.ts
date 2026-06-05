/**
 * SecureFace AI (React Native) - DatabaseManager
 * Persistence via react-native-mmkv v4 (Nitro Modules / C++ JSI bridge).
 * Uses `createMMKV()` factory — v4 API.
 */

import {createMMKV} from 'react-native-mmkv';
import type {MMKV} from 'react-native-mmkv';
import {serializeEmbedding, deserializeEmbedding} from './vectorSearch';
import type {Identity, FaceEmbedding, MatchResult} from '../types/face';

// ============================================================================
// MMKV INSTANCES (lazy-initialized)
// ============================================================================

let _storage: MMKV | null = null;
let _logStorage: MMKV | null = null;
let _syncStorage: MMKV | null = null;

function getStorage(): MMKV {
  if (!_storage) _storage = createMMKV({id: 'secureface-main'});
  return _storage;
}

function getLogStorage(): MMKV {
  if (!_logStorage) _logStorage = createMMKV({id: 'secureface-logs'});
  return _logStorage;
}

function getSyncStorage(): MMKV {
  if (!_syncStorage) _syncStorage = createMMKV({id: 'secureface-sync'});
  return _syncStorage;
}

const KEYS = {
  IDENTITIES: 'identities',
  VERIFICATION_LOGS: 'verification_logs',
  SYNC_QUEUE: 'sync_queue',
} as const;

// ============================================================================
// TYPES
// ============================================================================

interface SerializedIdentity {
  id: string;
  name: string;
  employeeId: string;
  embedding: number[];
  registeredAt: number;
  imageData?: string;
  metadata?: Record<string, unknown>;
}

export interface VerificationLogEntry {
  id: string;
  timestamp: number;
  identityId: string | null;
  identityName: string | null;
  similarity: number;
  isMatch: boolean;
  livenessScore: number;
  livenessPass: boolean;
  challengeType: string;
  processingTimeMs: number;
  deviceInfo: string;
}

export interface SyncPayload {
  id: string;
  type: 'verification' | 'registration' | 'deletion';
  payload: Record<string, unknown>;
  createdAt: number;
  retryCount: number;
  lastAttempt: number | null;
}

// ============================================================================
// SERIALIZATION HELPERS
// ============================================================================

function serializeIdentity(
  identity: Identity,
  employeeId?: string,
): SerializedIdentity {
  return {
    id: identity.id,
    name: identity.name,
    employeeId:
      employeeId ??
      (identity.metadata?.employeeId as string | undefined) ??
      identity.id,
    embedding: serializeEmbedding(identity.embedding),
    registeredAt: identity.registeredAt,
    imageData: identity.imageData,
    metadata: identity.metadata,
  };
}

function deserializeIdentity(s: SerializedIdentity): Identity {
  return {
    id: s.id,
    name: s.name,
    embedding: deserializeEmbedding(s.embedding) as FaceEmbedding,
    registeredAt: s.registeredAt,
    imageData: s.imageData,
    metadata: {...s.metadata, employeeId: s.employeeId},
  };
}

function readJSON<T>(mmkv: MMKV, key: string, fallback: T): T {
  const raw = mmkv.getString(key);
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeJSON(mmkv: MMKV, key: string, value: unknown): void {
  mmkv.set(key, JSON.stringify(value));
}

// ============================================================================
// IDENTITY PERSISTENCE
// ============================================================================

export function savePerson(identity: Identity, employeeId?: string): void {
  const storage = getStorage();
  const all = readJSON<SerializedIdentity[]>(storage, KEYS.IDENTITIES, []);
  const serialized = serializeIdentity(identity, employeeId);
  const filtered = all.filter(i => i.id !== identity.id);
  filtered.push(serialized);
  writeJSON(storage, KEYS.IDENTITIES, filtered);
  console.log(
    `[DatabaseManager] Saved identity: ${identity.name} (${identity.id})`,
  );
}

export function getAllPersons(): Identity[] {
  const all = readJSON<SerializedIdentity[]>(
    getStorage(),
    KEYS.IDENTITIES,
    [],
  );
  return all.map(deserializeIdentity);
}

export function getPerson(id: string): Identity | null {
  const all = readJSON<SerializedIdentity[]>(
    getStorage(),
    KEYS.IDENTITIES,
    [],
  );
  const found = all.find(i => i.id === id);
  return found ? deserializeIdentity(found) : null;
}

export function deletePerson(id: string): void {
  const storage = getStorage();
  const all = readJSON<SerializedIdentity[]>(storage, KEYS.IDENTITIES, []);
  writeJSON(
    storage,
    KEYS.IDENTITIES,
    all.filter(i => i.id !== id),
  );
  console.log(`[DatabaseManager] Deleted identity: ${id}`);
}

export function getPersonCount(): number {
  return readJSON<SerializedIdentity[]>(getStorage(), KEYS.IDENTITIES, [])
    .length;
}

// ============================================================================
// VERIFICATION LOGS
// ============================================================================

export function saveVerificationLog(
  entry: Omit<VerificationLogEntry, 'id'>,
): string {
  const id = `vlog_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  const logs = getLogStorage();
  const all = readJSON<VerificationLogEntry[]>(
    logs,
    KEYS.VERIFICATION_LOGS,
    [],
  );
  all.push({...entry, id});
  if (all.length > 500) all.splice(0, all.length - 500);
  writeJSON(logs, KEYS.VERIFICATION_LOGS, all);
  return id;
}

export function createVerificationLogFromMatch(
  matchResult: MatchResult | null,
  livenessScore: number,
  livenessPass: boolean,
  challengeType: string,
  processingTimeMs: number,
): Omit<VerificationLogEntry, 'id'> {
  return {
    timestamp: Date.now(),
    identityId: matchResult?.identity.id ?? null,
    identityName: matchResult?.identity.name ?? null,
    similarity: matchResult?.similarity ?? 0,
    isMatch: matchResult?.isMatch ?? false,
    livenessScore,
    livenessPass,
    challengeType,
    processingTimeMs,
    deviceInfo: 'ReactNative/Android',
  };
}

export function getVerificationLogs(limit = 100): VerificationLogEntry[] {
  return readJSON<VerificationLogEntry[]>(
    getLogStorage(),
    KEYS.VERIFICATION_LOGS,
    [],
  )
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, limit);
}

// ============================================================================
// OFFLINE SYNC QUEUE
// ============================================================================

export function enqueueOfflinePayload(
  type: SyncPayload['type'],
  payload: Record<string, unknown>,
): string {
  const id = `sync_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  const sync = getSyncStorage();
  const all = readJSON<SyncPayload[]>(sync, KEYS.SYNC_QUEUE, []);
  all.push({
    id,
    type,
    payload,
    createdAt: Date.now(),
    retryCount: 0,
    lastAttempt: null,
  });
  writeJSON(sync, KEYS.SYNC_QUEUE, all);
  return id;
}

export function getPendingSyncPayloads(): SyncPayload[] {
  return readJSON<SyncPayload[]>(getSyncStorage(), KEYS.SYNC_QUEUE, []);
}

export function removeSyncPayload(id: string): void {
  const sync = getSyncStorage();
  const all = readJSON<SyncPayload[]>(sync, KEYS.SYNC_QUEUE, []);
  writeJSON(
    sync,
    KEYS.SYNC_QUEUE,
    all.filter(s => s.id !== id),
  );
}

export function updateSyncPayloadRetry(id: string): void {
  const sync = getSyncStorage();
  const all = readJSON<SyncPayload[]>(sync, KEYS.SYNC_QUEUE, []);
  const item = all.find(s => s.id === id);
  if (item) {
    item.retryCount++;
    item.lastAttempt = Date.now();
    writeJSON(sync, KEYS.SYNC_QUEUE, all);
  }
}

export function getSyncQueueCount(): number {
  return readJSON<SyncPayload[]>(getSyncStorage(), KEYS.SYNC_QUEUE, []).length;
}

// ============================================================================
// STORE HYDRATION
// ============================================================================

export function hydrateStore(
  registerIdentity: (identity: Identity) => void,
): number {
  try {
    const identities = getAllPersons();
    for (const identity of identities) {
      registerIdentity(identity);
    }
    console.log(
      `[DatabaseManager] Hydrated store with ${identities.length} identities`,
    );
    return identities.length;
  } catch (error) {
    console.error('[DatabaseManager] Failed to hydrate store:', error);
    return 0;
  }
}

export function clearAllData(): void {
  getStorage().remove(KEYS.IDENTITIES);
  getLogStorage().remove(KEYS.VERIFICATION_LOGS);
  getSyncStorage().remove(KEYS.SYNC_QUEUE);
  console.log('[DatabaseManager] All data cleared');
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

export const DatabaseManager = {
  savePerson,
  getAllPersons,
  getPerson,
  deletePerson,
  getPersonCount,
  saveVerificationLog,
  createVerificationLogFromMatch,
  getVerificationLogs,
  enqueueOfflinePayload,
  getPendingSyncPayloads,
  removeSyncPayload,
  updateSyncPayloadRetry,
  getSyncQueueCount,
  hydrateStore,
  clearAllData,
};

export default DatabaseManager;
