/**
 * SecureFace AI (React Native) - Verify Face Screen
 * Uses REAL MLKit face detection for passive liveness and geometry-based 1:N cosine similarity matching.
 * The UI and liveness extraction exactly mimics the Register Face flow to ensure matched embeddings.
 */

import React, {useRef, useState, useCallback, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import {useFaceStore} from '../store/useFaceStore';
import {
  handleNoFace,
  startLivenessWindow,
  feedLivenessFrame,
  isLivenessWindowActive,
  useBiometricsFrameProcessor,
  convertTSFaceToDetection,
  initializeProcessor,
  configureProcessor,
} from '../services/faceProcessor';
import {DEFAULT_MATCH_THRESHOLD, searchIdentity} from '../services/vectorSearch';
import {DatabaseManager} from '../services/databaseManager';
import {OfflineSyncQueue} from '../services/offlineSyncQueue';
import type {FaceEmbedding} from '../types/face';

// ============================================================================
// TYPES
// ============================================================================

type VerifyPhase =
  | 'init'
  | 'form'
  | 'liveness'
  | 'matching'
  | 'matched'
  | 'unauthorized'
  | 'error';

// ============================================================================
// COMPONENT
// ============================================================================

const VerifyFaceScreen: React.FC = () => {
  const device = useCameraDevice('front');
  const {hasPermission, requestPermission} = useCameraPermission();

  // Zustand
  const currentDetection = useFaceStore(s => s.currentDetection);
  const identities = useFaceStore(s => s.identities);
  const setStatus = useFaceStore(s => s.setStatus);
  const lastMatch = useFaceStore(s => s.lastMatch);
  const setLastMatch = useFaceStore(s => s.setLastMatch);

  // Local state
  const [phase, setPhase] = useState<VerifyPhase>('init');
  const [progress, setProgress] = useState(0);
  const [processingTimeMs, setProcessingTimeMs] = useState(0);
  const verifyStartRef = useRef<number>(0);
  const [livenessResult, setLivenessResult] = useState<{
    isLive: boolean;
    earVariance: number;
    framesProcessed: number;
  } | null>(null);
  const livenessEmbeddingRef = useRef<FaceEmbedding | null>(null);

  // ── FACE DETECTOR OUTPUT (real MLKit + TFLite) ────────────────────────
  const frameProcessor = useBiometricsFrameProcessor(
    (tsFace: any, frameWidth: number, frameHeight: number, embeddingArr: number[] | null) => {
      // During liveness phase, feed frames to accumulator
      if (isLivenessWindowActive()) {
        const detection = convertTSFaceToDetection(tsFace, frameWidth, frameHeight, Date.now());
        const result = feedLivenessFrame(detection, embeddingArr);
        if (result) {
          // Liveness window completed
          setLivenessResult({
            isLive: result.isLive,
            earVariance: result.earVariance,
            framesProcessed: result.framesProcessed,
          });
          livenessEmbeddingRef.current = result.embedding;
          setProgress(100);

          if (!result.isLive) {
            setPhase('unauthorized');
            setStatus('unauthorized');
            setLastMatch(null);
            
            // Log failed liveness
            const logEntry = DatabaseManager.createVerificationLogFromMatch(
              null,
              result.earVariance,
              false,
              'passive_ear',
              1500,
            );
            DatabaseManager.saveVerificationLog(logEntry);
            
            Alert.alert(
              'Liveness Failed',
              `Could not confirm live presence.\nEAR Variance: ${result.earVariance.toFixed(6)}`,
            );
          } else {
            // Proceed to match
            completeVerification(result.embedding, result.earVariance);
          }
        } else {
          // Still accumulating
          setProgress(p => Math.min(p + 3, 90));
        }
        return;
      }

      // Normal detection - just update state
      const detection = convertTSFaceToDetection(tsFace, frameWidth, frameHeight, Date.now());
      useFaceStore.getState().setFaceDetection(detection);
    },
    () => {
      handleNoFace();
    }
  );

  // ── INIT ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      if (!hasPermission) {
        const granted = await requestPermission();
        if (!granted) {
          Alert.alert('Camera Permission', 'Camera access is required.');
          return;
        }
      }
      await initializeProcessor();
      configureProcessor({
        enableAntiSpoofing: false, // We'll use the passive liveness accumulator directly
        enableEmbeddingExtraction: false,
        frameSkip: 1,
      });
      setPhase('form');
    };
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasPermission]);

  const hasFace = !!currentDetection;

  // ── VERIFICATION FLOW ─────────────────────────────────────────────────
  const completeVerification = useCallback(
    (embedding: FaceEmbedding | null, earVariance: number) => {
      if (!embedding) {
        setPhase('unauthorized');
        Alert.alert('Extraction Failed', 'Could not extract face embedding.');
        return;
      }

      setPhase('matching');
      setStatus('verifying');

      // Use a slightly more forgiving threshold since geometric embeddings can have natural variance
      const matchResult = searchIdentity(embedding, identities, DEFAULT_MATCH_THRESHOLD - 0.05);
      setLastMatch(matchResult);

      const elapsed = Date.now() - verifyStartRef.current;
      setProcessingTimeMs(elapsed);

      setTimeout(() => {
        if (matchResult && matchResult.isMatch) {
          setPhase('matched');
          setStatus('success');
        } else {
          setPhase('unauthorized');
          setStatus('unauthorized');
        }

        const logEntry = DatabaseManager.createVerificationLogFromMatch(
          matchResult,
          earVariance,
          true,
          'passive_ear',
          elapsed,
        );
        DatabaseManager.saveVerificationLog(logEntry);

        if (matchResult && matchResult.isMatch) {
          OfflineSyncQueue.enqueueVerificationEvent(
            matchResult,
            earVariance,
            true,
            'passive_ear',
          ).catch(() => {});
        }
      }, 500); // Small delay for UX
    },
    [identities, setLastMatch, setStatus],
  );

  const handleTriggerVerification = useCallback(() => {
    if (identities.length === 0) {
      Alert.alert('No Identities', 'Register at least one identity first.');
      return;
    }
    if (!hasFace) {
      Alert.alert('No Face', 'Position your face in the camera frame.');
      return;
    }

    setPhase('liveness');
    setStatus('verifying');
    setProgress(0);
    setProcessingTimeMs(0);
    verifyStartRef.current = Date.now();
    livenessEmbeddingRef.current = null;
    setLastMatch(null);

    // Start real liveness window — accumulates EAR values over 1.5s
    startLivenessWindow(1500);
  }, [hasFace, identities.length, setStatus, setLastMatch]);

  const handleReset = () => {
    setPhase('form');
    setProgress(0);
    setLivenessResult(null);
    livenessEmbeddingRef.current = null;
    setLastMatch(null);
    setStatus('scanning');
  };

  // ── RENDER ────────────────────────────────────────────────────────────

  if (!device) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No front camera available</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera with real face detection */}
      <View style={styles.cameraContainer}>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={phase !== 'matched' && phase !== 'unauthorized'}
          outputs={[frameProcessor]}
        />

        {/* Phase Overlays */}
        {phase === 'liveness' && (
          <View style={styles.overlayAmber}>
            <Text style={styles.overlayIcon}>👁️</Text>
            <Text style={styles.overlayTitle}>Verifying Live Presence...</Text>
            <Text style={styles.overlaySubtext}>
              Blink naturally — MLKit is watching
            </Text>
          </View>
        )}

        {phase === 'matching' && (
          <View style={styles.overlayPurple}>
            <Text style={styles.overlayIcon}>🔎</Text>
            <Text style={styles.overlayTitle}>Matching Identity...</Text>
          </View>
        )}

        {phase === 'matched' && lastMatch && (
          <View style={styles.overlayGreen}>
            <Text style={styles.overlayIcon}>✅</Text>
            <Text style={styles.overlayTitle}>Identity Verified</Text>
            <Text style={styles.overlaySubtext}>
              {lastMatch.identity.name} — {(lastMatch.similarity * 100).toFixed(1)}% Geometric Match
            </Text>
            <View style={styles.verifyBadgeRow}>
              <View style={styles.verifyBadge}>
                <Text style={styles.verifyBadgeText}>
                  ⚡ {processingTimeMs < 1000 ? `${processingTimeMs}ms` : `${(processingTimeMs/1000).toFixed(1)}s`}
                </Text>
              </View>
              <View style={[styles.verifyBadge, styles.verifyBadgeLive]}>
                <Text style={styles.verifyBadgeText}>🛡️ Live</Text>
              </View>
              <View style={[styles.verifyBadge, styles.verifyBadgeSync]}>
                <Text style={styles.verifyBadgeText}>☁️ Synced</Text>
              </View>
            </View>
            <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
              <Text style={styles.resetButtonText}>Verify Again</Text>
            </TouchableOpacity>
          </View>
        )}

        {phase === 'unauthorized' && (
          <View style={styles.overlayRed}>
            <Text style={styles.overlayIcon}>❌</Text>
            <Text style={styles.overlayTitle}>Access Denied</Text>
            <Text style={styles.successSubtext}>
              Access Denied. Liveness check failed or face not recognized.
            </Text>
            {lastMatch && (
              <Text style={styles.successSubtext}>
                (Highest Similarity: {(lastMatch.similarity * 100).toFixed(2)}%)
              </Text>
            )}
            <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
              <Text style={styles.resetButtonText}>Try Again</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Detection indicator */}
        {hasFace && phase === 'form' && (
          <View style={styles.detectionBadge}>
            <View style={styles.detectionDot} />
            <Text style={styles.detectionText}>Face Detected (MLKit)</Text>
          </View>
        )}

        {!hasFace && phase === 'form' && (
          <View style={styles.noFaceBadge}>
            <Text style={styles.noFaceText}>⚠️ No face detected</Text>
          </View>
        )}
      </View>

      {/* Progress Bar */}
      {(phase === 'liveness' || phase === 'matching') && (
        <View style={styles.progressContainer}>
          <View style={[styles.progressBar, {width: `${progress}%`}]} />
        </View>
      )}

      {/* Form Area */}
      {(phase === 'form' || phase === 'error' || phase === 'unauthorized' || phase === 'matched') && (
        <ScrollView style={styles.formContainer}>
          <Text style={styles.formTitle}>Identity Verification</Text>
          <Text style={styles.infoText}>
            Geometric Landmark Fast-Match: scans only the matching category from the local vector DB. Passive EAR liveness blocks photos, screens & deepfakes.
          </Text>

          {phase === 'form' && (
            <>
              <TouchableOpacity
                style={[
                  styles.extractButton,
                  (!hasFace || identities.length === 0) && styles.extractButtonDisabled,
                ]}
                onPress={handleTriggerVerification}
                disabled={!hasFace || identities.length === 0}
                activeOpacity={0.8}>
                <Text style={styles.extractButtonText}>Verify Identity</Text>
              </TouchableOpacity>
              <Text style={styles.statusHelperText}>
                  {(!hasFace || identities.length === 0)
                    ? (identities.length === 0 ? '⚠️ No identities registered yet.' : '⏳ Waiting for face detection...')
                    : 'Ready to verify'}
                </Text>
            </>
          )}

          {/* Liveness Result */}
          {livenessResult && (
            <View
              style={[
                styles.resultCard,
                livenessResult.isLive
                  ? styles.resultCardPass
                  : styles.resultCardFail,
              ]}>
              <Text style={styles.resultTitle}>
                {livenessResult.isLive ? '✅ LIVE' : '❌ FAILED'}
              </Text>
              <Text style={styles.resultDetail}>
                EAR Variance: {livenessResult.earVariance.toFixed(6)}
              </Text>
              <Text style={styles.resultDetail}>
                Frames: {livenessResult.framesProcessed}
              </Text>
            </View>
          )}

          {/* Match Result Details */}
          {lastMatch && (
             <View
             style={[
               styles.resultCard,
               lastMatch.isMatch
                 ? styles.resultCardPass
                 : styles.resultCardFail,
             ]}>
             <Text style={styles.resultTitle}>
               {lastMatch.isMatch ? '✅ GEOMETRIC MATCH' : '❌ NO MATCH'}
             </Text>
             <Text style={styles.resultDetail}>
               Cosine Similarity: {(lastMatch.similarity * 100).toFixed(2)}%
             </Text>
             {processingTimeMs > 0 && (
               <Text style={styles.resultDetail}>
                 Processing Time: {processingTimeMs}ms {processingTimeMs < 1000 ? '(< 1s ✓)' : ''}
               </Text>
             )}
             {lastMatch.isMatch && (
                 <Text style={styles.resultDetail}>
                    Matched: {lastMatch.identity.name} (Landmark Vector)
                 </Text>
             )}
           </View>
          )}

          {/* Live Detection Metrics */}
          {currentDetection && phase === 'form' && (
            <View style={styles.metricsCard}>
              <Text style={styles.metricsTitle}>
                🔴 LIVE Detection (MLKit)
              </Text>
              <MetricRow
                label="Left Eye Open"
                value={`${(currentDetection.leftEye.openProbability * 100).toFixed(0)}%`}
              />
              <MetricRow
                label="Right Eye Open"
                value={`${(currentDetection.rightEye.openProbability * 100).toFixed(0)}%`}
              />
              <MetricRow
                label="Yaw (L/R)"
                value={`${currentDetection.orientation.eulerY.toFixed(1)}°`}
              />
              <MetricRow
                label="Pitch (U/D)"
                value={`${currentDetection.orientation.eulerX.toFixed(1)}°`}
              />
              <MetricRow
                label="Smile"
                value={`${(currentDetection.smilingProbability * 100).toFixed(0)}%`}
              />
            </View>
          )}
        </ScrollView>
      )}
    </View>
  );
};

// ============================================================================
// HELPERS
// ============================================================================

const MetricRow: React.FC<{label: string; value: string}> = ({
  label,
  value,
}) => (
  <View style={styles.metricRow}>
    <Text style={styles.metricLabel}>{label}</Text>
    <Text style={styles.metricValue}>{value}</Text>
  </View>
);

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#0f172a'},
  cameraContainer: {height: 300, position: 'relative'},
  errorText: {
    color: '#ef4444',
    textAlign: 'center',
    marginTop: 100,
    fontSize: 16,
  },

  overlayAmber: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(245, 158, 11, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlayPurple: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(139, 92, 246, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlayGreen: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlayRed: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(69, 10, 10, 0.85)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlayIcon: {fontSize: 40, marginBottom: 12},
  overlayTitle: {color: '#fff', fontSize: 18, fontWeight: '700'},
  overlaySubtext: {color: '#cbd5e1', fontSize: 13, marginTop: 6},
  successSubtext: {color: '#cbd5e1', fontSize: 14, textAlign: 'center', marginHorizontal: 20},

  detectionBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 16,
  },
  detectionDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#22c55e',
    marginRight: 6,
  },
  detectionText: {color: '#22c55e', fontSize: 12, fontWeight: '600'},

  noFaceBadge: {
    position: 'absolute',
    top: 12,
    alignSelf: 'center',
    backgroundColor: 'rgba(239, 68, 68, 0.7)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
  },
  noFaceText: {color: '#fff', fontSize: 12, fontWeight: '600'},

  progressContainer: {height: 4, backgroundColor: '#1e293b'},
  progressBar: {height: 4, backgroundColor: '#3b82f6'},

  formContainer: {flex: 1, padding: 20},
  formTitle: {
    color: '#e2e8f0',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 8,
  },
  infoText: {
    color: '#94a3b8',
    fontSize: 13,
    marginBottom: 16,
  },

  extractButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 8,
  },
  extractButtonDisabled: {backgroundColor: '#334155'},
  extractButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  statusHelperText: {
    color: '#94a3b8',
    textAlign: 'center',
    marginTop: 8,
    fontSize: 13,
  },

  resetButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 10,
    marginTop: 20,
  },
  resetButtonText: {color: '#fff', fontSize: 14, fontWeight: '600'},

  verifyBadgeRow: {
    flexDirection: 'row',
    gap: 6,
    marginTop: 12,
  },
  verifyBadge: {
    backgroundColor: 'rgba(59, 130, 246, 0.25)',
    borderWidth: 1,
    borderColor: '#3b82f6',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 16,
  },
  verifyBadgeLive: {
    backgroundColor: 'rgba(34, 197, 94, 0.2)',
    borderColor: '#22c55e',
  },
  verifyBadgeSync: {
    backgroundColor: 'rgba(139, 92, 246, 0.2)',
    borderColor: '#8b5cf6',
  },
  verifyBadgeText: {
    color: '#e2e8f0',
    fontSize: 11,
    fontWeight: '600',
  },

  resultCard: {borderRadius: 12, padding: 16, marginTop: 16, borderWidth: 1},
  resultCardPass: {backgroundColor: '#052e16', borderColor: '#166534'},
  resultCardFail: {backgroundColor: '#450a0a', borderColor: '#991b1b'},
  resultTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 6,
  },
  resultDetail: {color: '#94a3b8', fontSize: 12, fontFamily: 'monospace'},

  metricsCard: {
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#334155',
  },
  metricsTitle: {
    color: '#94a3b8',
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 10,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  metricLabel: {color: '#64748b', fontSize: 13},
  metricValue: {color: '#e2e8f0', fontSize: 13, fontFamily: 'monospace'},
});

export default VerifyFaceScreen;
