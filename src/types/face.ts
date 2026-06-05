/**
 * SecureFace AI (React Native) - Core Type Definitions
 * Production-grade type system for face detection, anti-spoofing, and identity verification.
 * Ported from web version — all browser-specific types replaced with RN equivalents.
 */

// ============================================================================
// GEOMETRY & LANDMARK TYPES
// ============================================================================

/** 2D Point with sub-pixel precision */
export interface Point2D {
  x: number;
  y: number;
}

/** 3D Point for depth-aware landmarks */
export interface Point3D extends Point2D {
  z: number;
}

/** Axis-aligned bounding box with normalized [0,1] coordinates */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Complete facial landmark set from detection backend */
export interface FaceLandmarks {
  positions: Point2D[];
  leftEye: Point2D[];
  rightEye: Point2D[];
  nose: Point2D[];
  mouth: Point2D[];
  jawOutline: Point2D[];
}

// ============================================================================
// CONST OBJECTS (no enums — compatible with erasableSyntaxOnly)
// ============================================================================

/** Active challenge types for liveness verification */
export const LivenessChallenge = {
  BLINK: 'BLINK',
  TURN_LEFT: 'TURN_LEFT',
  TURN_RIGHT: 'TURN_RIGHT',
  SMILE: 'SMILE',
  NONE: 'NONE',
} as const;

export type LivenessChallenge = (typeof LivenessChallenge)[keyof typeof LivenessChallenge];

/** States of the challenge-response state machine */
export const ChallengeState = {
  IDLE: 'IDLE',
  PROMPTING: 'PROMPTING',
  DETECTING: 'DETECTING',
  SUCCESS: 'SUCCESS',
  FAILED: 'FAILED',
  TIMEOUT: 'TIMEOUT',
} as const;

export type ChallengeState = (typeof ChallengeState)[keyof typeof ChallengeState];

/** Global face processing status */
export const ProcessingStatus = {
  IDLE: 'idle',
  INITIALIZING: 'initializing',
  SCANNING: 'scanning',
  DETECTING: 'detecting',
  EXTRACTING: 'extracting',
  VERIFYING: 'verifying',
  REGISTERING: 'registering',
  SUCCESS: 'success',
  ERROR: 'error',
  UNAUTHORIZED: 'unauthorized',
} as const;

export type ProcessingStatus = (typeof ProcessingStatus)[keyof typeof ProcessingStatus];

// ============================================================================
// FACE DETECTION RESULT
// ============================================================================

/** Face orientation in 3D Euler angles (degrees) */
export interface FaceOrientation {
  eulerX: number;
  eulerY: number;
  eulerZ: number;
}

/** Individual eye state with open probability */
export interface EyeState {
  openProbability: number;
  landmarks: Point2D[];
  aspectRatio: number;
}

/** Complete face detection result per frame */
export interface FaceDetection {
  id: string;
  boundingBox: BoundingBox;
  landmarks: FaceLandmarks;
  orientation: FaceOrientation;
  leftEye: EyeState;
  rightEye: EyeState;
  smilingProbability: number;
  timestamp: number;
}

// ============================================================================
// ANTI-SPOOFING & LIVENESS
// ============================================================================

/** EAR (Eye Aspect Ratio) tracking for blink detection */
export interface EARHistory {
  timestamp: number;
  leftEAR: number;
  rightEAR: number;
  avgEAR: number;
}

/** Passive texture analysis result */
export interface TextureAnalysis {
  varianceScore: number;
  depthClueScore: number;
  isLive: boolean;
  confidence: number;
}

/** Complete anti-spoofing result */
export interface AntiSpoofResult {
  isLive: boolean;
  confidence: number;
  passiveResult: TextureAnalysis;
  activeResult: ActiveChallengeResult | null;
  overallScore: number;
}

/** Active challenge response result */
export interface ActiveChallengeResult {
  challenge: LivenessChallenge;
  state: ChallengeState;
  startTime: number;
  endTime: number;
  success: boolean;
  attempts: number;
}

// ============================================================================
// EMBEDDING & IDENTITY
// ============================================================================

/** 128-dimensional face embedding vector */
export type FaceEmbedding = Float32Array & { length: 128 };

/** Registered identity with embedding and metadata */
export interface Identity {
  id: string;
  name: string;
  embedding: FaceEmbedding;
  registeredAt: number;
  imageData?: string;
  metadata?: Record<string, unknown>;
}

/** 1:N Match result with similarity score */
export interface MatchResult {
  identity: Identity;
  similarity: number;
  isMatch: boolean;
}

/** Embedding extraction status */
export interface EmbeddingExtraction {
  status: 'idle' | 'extracting' | 'success' | 'error';
  embedding: FaceEmbedding | null;
  error?: string;
}

// ============================================================================
// FRAME PROCESSOR (React Native)
// ============================================================================

/** Frame processor configuration */
export interface FrameProcessorConfig {
  enableAntiSpoofing: boolean;
  enableEmbeddingExtraction: boolean;
  matchThreshold: number;
  antiSpoofThreshold: number;
  maxChallengeAttempts: number;
  challengeTimeoutMs: number;
  frameSkip: number;
}

/** Camera configuration */
export interface CameraConfig {
  width: number;
  height: number;
  facing: 'front' | 'back';
  frameRate: number;
}

/** Frame processor output after analysis */
export interface FrameProcessorOutput {
  detection: FaceDetection | null;
  antiSpoofResult: AntiSpoofResult | null;
  embedding: FaceEmbedding | null;
  matchResult: MatchResult | null;
  processingTimeMs: number;
}

// ============================================================================
// APPLICATION STATE
// ============================================================================

/** Complete Zustand store state */
export interface FaceStoreState {
  // Status
  status: ProcessingStatus;
  error: string | null;

  // Detection
  currentDetection: FaceDetection | null;
  detectionHistory: FaceDetection[];

  // Anti-spoofing
  antiSpoofResult: AntiSpoofResult | null;
  currentChallenge: LivenessChallenge;
  challengeState: ChallengeState;
  earHistory: EARHistory[];

  // Embeddings
  currentEmbedding: FaceEmbedding | null;

  // Identity database
  identities: Identity[];
  lastMatch: MatchResult | null;

  // Configuration
  processorConfig: FrameProcessorConfig;
  cameraConfig: CameraConfig;

  // Actions
  setStatus: (status: ProcessingStatus) => void;
  setError: (error: string | null) => void;
  setFaceDetection: (detection: FaceDetection | null) => void;
  setAntiSpoofResult: (result: AntiSpoofResult | null) => void;
  setChallenge: (challenge: LivenessChallenge) => void;
  setChallengeState: (state: ChallengeState) => void;
  addEARHistory: (entry: EARHistory) => void;
  clearEARHistory: () => void;
  setCurrentEmbedding: (embedding: FaceEmbedding | null) => void;
  registerIdentity: (identity: Identity) => void;
  removeIdentity: (id: string) => void;
  setLastMatch: (match: MatchResult | null) => void;
  updateProcessorConfig: (config: Partial<FrameProcessorConfig>) => void;
  updateCameraConfig: (config: Partial<CameraConfig>) => void;
  reset: () => void;
}
