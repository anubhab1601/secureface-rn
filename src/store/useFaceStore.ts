/**
 * SecureFace AI (React Native) - Zustand State Store
 * Central state management — identical API to web version.
 */

import { create } from 'zustand';
import {
  LivenessChallenge,
  ChallengeState,
  ProcessingStatus,
} from '../types/face';
import type {
  FaceStoreState,
  FaceDetection,
  AntiSpoofResult,
  EARHistory,
  FaceEmbedding,
  Identity,
  MatchResult,
  FrameProcessorConfig,
  CameraConfig,
} from '../types/face';

const DEFAULT_PROCESSOR_CONFIG: FrameProcessorConfig = {
  enableAntiSpoofing: true,
  enableEmbeddingExtraction: true,
  matchThreshold: 0.992,
  antiSpoofThreshold: 0.85,
  maxChallengeAttempts: 3,
  challengeTimeoutMs: 4000,
  frameSkip: 1,
};

const DEFAULT_CAMERA_CONFIG: CameraConfig = {
  width: 640,
  height: 480,
  facing: 'front',
  frameRate: 30,
};

export const useFaceStore = create<FaceStoreState>((set) => ({
  status: ProcessingStatus.IDLE,
  error: null,
  currentDetection: null,
  detectionHistory: [],
  antiSpoofResult: null,
  currentChallenge: LivenessChallenge.NONE,
  challengeState: ChallengeState.IDLE,
  earHistory: [],
  currentEmbedding: null,
  identities: [],
  lastMatch: null,
  processorConfig: DEFAULT_PROCESSOR_CONFIG,
  cameraConfig: DEFAULT_CAMERA_CONFIG,

  setStatus: (status) => set({ status }),
  setError: (error) => set({ error }),

  setFaceDetection: (detection) =>
    set((state) => ({
      currentDetection: detection,
      detectionHistory: detection
        ? [...state.detectionHistory.slice(-29), detection]
        : state.detectionHistory,
    })),

  setAntiSpoofResult: (result) => set({ antiSpoofResult: result }),
  setChallenge: (challenge) => set({ currentChallenge: challenge }),
  setChallengeState: (challengeState) => set({ challengeState }),

  addEARHistory: (entry) =>
    set((state) => ({
      earHistory: [...state.earHistory.slice(-99), entry],
    })),
  clearEARHistory: () => set({ earHistory: [] }),

  setCurrentEmbedding: (embedding) => set({ currentEmbedding: embedding }),

  registerIdentity: (identity) =>
    set((state) => ({
      identities: [
        ...state.identities.filter((i) => i.id !== identity.id),
        identity,
      ],
    })),

  removeIdentity: (id) =>
    set((state) => ({
      identities: state.identities.filter((i) => i.id !== id),
    })),

  setLastMatch: (match) => set({ lastMatch: match }),

  updateProcessorConfig: (config) =>
    set((state) => ({
      processorConfig: { ...state.processorConfig, ...config },
    })),

  updateCameraConfig: (config) =>
    set((state) => ({
      cameraConfig: { ...state.cameraConfig, ...config },
    })),

  reset: () =>
    set({
      status: ProcessingStatus.IDLE,
      error: null,
      currentDetection: null,
      detectionHistory: [],
      antiSpoofResult: null,
      currentChallenge: LivenessChallenge.NONE,
      challengeState: ChallengeState.IDLE,
      earHistory: [],
      currentEmbedding: null,
      lastMatch: null,
    }),
}));
