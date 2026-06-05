/**
 * SecureFace AI (React Native) - Anti-Spoofing Engine
 * EAR blink detection, Euler angle head-turn, passive texture analysis.
 * Pure TypeScript math — no platform dependencies.
 */

import {LivenessChallenge, ChallengeState} from '../types/face';
import type {
  TextureAnalysis,
  AntiSpoofResult,
  ActiveChallengeResult,
  FaceDetection,
  Point2D,
} from '../types/face';

export const EAR_BLINK_THRESHOLD = 0.22;
export const BLINK_RECOVERY_MS = 400;
export const TURN_LEFT_THRESHOLD = 20;
export const TURN_RIGHT_THRESHOLD = -20;
export const MIN_TEXTURE_VARIANCE = 15.0;
export const MIN_DEPTH_CLUE_SCORE = 0.3;
export const DEFAULT_CHALLENGE_TIMEOUT_MS = 4000;
export const DEFAULT_MAX_ATTEMPTS = 3;

export function euclideanDistance(a: Point2D, b: Point2D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
 */
export function calculateEAR(eyeLandmarks: Point2D[]): number {
  if (eyeLandmarks.length < 6) return 0.3;
  const p1 = eyeLandmarks[0];
  const p2 = eyeLandmarks[1];
  const p3 = eyeLandmarks[2];
  const p4 = eyeLandmarks[3];
  const p5 = eyeLandmarks[4];
  const p6 = eyeLandmarks[5];
  const vertical1 = euclideanDistance(p2, p6);
  const vertical2 = euclideanDistance(p3, p5);
  const horizontal = euclideanDistance(p1, p4);
  if (horizontal === 0) return 0;
  return (vertical1 + vertical2) / (2 * horizontal);
}

export function selectRandomChallenge(): LivenessChallenge {
  const challenges: LivenessChallenge[] = [
    LivenessChallenge.BLINK,
    LivenessChallenge.TURN_LEFT,
    LivenessChallenge.TURN_RIGHT,
  ];
  return challenges[Math.floor(Math.random() * challenges.length)];
}

/**
 * Challenge-Response Finite State Machine
 */
export class ChallengeFSM {
  private state: ChallengeState = ChallengeState.IDLE;
  private currentChallenge: LivenessChallenge = LivenessChallenge.NONE;
  private startTime = 0;
  private attempts = 0;
  private maxAttempts: number;
  private timeoutMs: number;
  private blinkStartTime = 0;
  private wasEyeClosed = false;

  constructor(
    maxAttempts: number = DEFAULT_MAX_ATTEMPTS,
    timeoutMs: number = DEFAULT_CHALLENGE_TIMEOUT_MS,
  ) {
    this.maxAttempts = maxAttempts;
    this.timeoutMs = timeoutMs;
  }

  startChallenge(challenge: LivenessChallenge): void {
    this.currentChallenge = challenge;
    this.state = ChallengeState.DETECTING;
    this.startTime = Date.now();
    this.attempts = 0;
    this.wasEyeClosed = false;
    this.blinkStartTime = 0;
  }

  processFrame(detection: FaceDetection): ActiveChallengeResult {
    const now = Date.now();
    const elapsed = now - this.startTime;

    if (elapsed > this.timeoutMs) {
      this.state = ChallengeState.TIMEOUT;
      return this.buildResult();
    }

    if (this.state !== ChallengeState.DETECTING) {
      return this.buildResult();
    }

    this.attempts++;

    switch (this.currentChallenge) {
      case LivenessChallenge.BLINK:
        this.processBlink(detection, now);
        break;
      case LivenessChallenge.TURN_LEFT:
        this.processTurnLeft(detection);
        break;
      case LivenessChallenge.TURN_RIGHT:
        this.processTurnRight(detection);
        break;
    }

    return this.buildResult();
  }

  private processBlink(detection: FaceDetection, now: number): void {
    const avgEAR =
      (detection.leftEye.aspectRatio + detection.rightEye.aspectRatio) / 2;

    if (avgEAR < EAR_BLINK_THRESHOLD && !this.wasEyeClosed) {
      this.wasEyeClosed = true;
      this.blinkStartTime = now;
    }

    if (
      this.wasEyeClosed &&
      avgEAR > 0.28 &&
      now - this.blinkStartTime < BLINK_RECOVERY_MS
    ) {
      this.state = ChallengeState.SUCCESS;
    }

    if (this.wasEyeClosed && now - this.blinkStartTime >= BLINK_RECOVERY_MS) {
      this.wasEyeClosed = false;
      this.blinkStartTime = 0;
    }
  }

  private processTurnLeft(detection: FaceDetection): void {
    if (detection.orientation.eulerY > TURN_LEFT_THRESHOLD) {
      this.state = ChallengeState.SUCCESS;
    }
  }

  private processTurnRight(detection: FaceDetection): void {
    if (detection.orientation.eulerY < TURN_RIGHT_THRESHOLD) {
      this.state = ChallengeState.SUCCESS;
    }
  }

  private buildResult(): ActiveChallengeResult {
    return {
      challenge: this.currentChallenge,
      state: this.state,
      startTime: this.startTime,
      endTime: Date.now(),
      success: this.state === ChallengeState.SUCCESS,
      attempts: this.attempts,
    };
  }

  reset(): void {
    this.state = ChallengeState.IDLE;
    this.currentChallenge = LivenessChallenge.NONE;
    this.startTime = 0;
    this.attempts = 0;
    this.wasEyeClosed = false;
    this.blinkStartTime = 0;
  }
}

/**
 * Passive texture analysis (simulated — real version uses pixel variance from frame)
 */
export function analyzeTexture(
  detection: FaceDetection,
): TextureAnalysis {
  const earVariance =
    Math.abs(detection.leftEye.aspectRatio - detection.rightEye.aspectRatio) *
    100;
  const depthClue =
    Math.abs(detection.orientation.eulerX) * 0.02 +
    Math.abs(detection.orientation.eulerZ) * 0.01 +
    0.3;

  return {
    varianceScore: Math.min(earVariance + 15, 50),
    depthClueScore: Math.min(depthClue, 1),
    isLive:
      earVariance + 15 > MIN_TEXTURE_VARIANCE && depthClue > MIN_DEPTH_CLUE_SCORE,
    confidence: Math.min((earVariance + 15) / 50 + depthClue / 2, 1),
  };
}

/**
 * Combine passive + active results into final anti-spoof verdict.
 * Weights: passive 40%, active 60%
 */
export function runAntiSpoofPipeline(
  detection: FaceDetection,
  _imageData: unknown,
  activeResult: ActiveChallengeResult | null,
): AntiSpoofResult {
  const passive = analyzeTexture(detection);

  let activeScore = 0;
  if (activeResult) {
    activeScore = activeResult.success ? 1.0 : 0.0;
  }

  const overallScore = passive.confidence * 0.4 + activeScore * 0.6;
  const isLive = overallScore > 0.5;

  return {
    isLive,
    confidence: overallScore,
    passiveResult: passive,
    activeResult,
    overallScore,
  };
}
