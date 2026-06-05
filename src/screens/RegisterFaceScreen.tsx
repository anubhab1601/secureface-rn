/**
 * SecureFace AI (React Native) - Register Face Screen
 * Uses REAL MLKit face detection via useFaceDetectorOutput.
 */

import React, {useRef, useState, useCallback, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
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
import {DatabaseManager} from '../services/databaseManager';
import {OfflineSyncQueue} from '../services/offlineSyncQueue';
import type {FaceEmbedding, Identity} from '../types/face';

// ============================================================================
// TYPES
// ============================================================================

type RegisterPhase =
  | 'init'
  | 'form'
  | 'liveness'
  | 'extracting'
  | 'saving'
  | 'success'
  | 'error';

// ============================================================================
// VALIDATION
// ============================================================================

function validateEmployeeId(id: string): string | null {
  if (!id.trim()) return 'Employee ID is required';
  if (id.trim().length < 2) return 'Must be at least 2 characters';
  if (!/^[a-zA-Z0-9_-]+$/.test(id.trim()))
    return 'Only letters, numbers, hyphens, underscores';
  return null;
}

function validateFullName(name: string): string | null {
  if (!name.trim()) return 'Full name is required';
  if (name.trim().length < 2) return 'Must be at least 2 characters';
  return null;
}

// ============================================================================
// COMPONENT
// ============================================================================

const RegisterFaceScreen: React.FC = () => {
  const device = useCameraDevice('front');
  const {hasPermission, requestPermission} = useCameraPermission();

  // Zustand
  const registerIdentity = useFaceStore(s => s.registerIdentity);
  const currentDetection = useFaceStore(s => s.currentDetection);
  const setStatus = useFaceStore(s => s.setStatus);

  // Local state
  const [phase, setPhase] = useState<RegisterPhase>('init');
  const [employeeId, setEmployeeId] = useState('');
  const [fullName, setFullName] = useState('');
  const [employeeIdError, setEmployeeIdError] = useState<string | null>(null);
  const [fullNameError, setFullNameError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
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
            setPhase('form');
            setStatus('scanning');
            Alert.alert(
              'Liveness Failed',
              `Could not confirm live presence.\nEAR Variance: ${result.earVariance.toFixed(6)}\nFrames: ${result.framesProcessed}`,
            );
          } else {
            // Proceed to save
            completeRegistration(result.embedding);
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
        enableAntiSpoofing: false,
        enableEmbeddingExtraction: true,
        frameSkip: 1,
      });
      setPhase('form');
    };
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasPermission]);

  // ── FORM HANDLERS ─────────────────────────────────────────────────────
  const handleEmployeeIdChange = (val: string) => {
    setEmployeeId(val);
    setEmployeeIdError(val ? validateEmployeeId(val) : null);
  };

  const handleFullNameChange = (val: string) => {
    setFullName(val);
    setFullNameError(val ? validateFullName(val) : null);
  };

  const isFormValid =
    validateEmployeeId(employeeId) === null &&
    validateFullName(fullName) === null;
  const hasFace = !!currentDetection;

  // ── REGISTRATION FLOW ─────────────────────────────────────────────────
  const completeRegistration = useCallback(
    (embedding: FaceEmbedding | null) => {
      if (!embedding) {
        setPhase('form');
        Alert.alert('Extraction Failed', 'Could not extract face embedding.');
        return;
      }

      setPhase('saving');
      setStatus('registering');

      const identity: Identity = {
        id: `identity_${employeeId.trim()}_${Date.now()}`,
        name: fullName.trim(),
        embedding,
        registeredAt: Date.now(),
        metadata: {employeeId: employeeId.trim()},
      };

      DatabaseManager.savePerson(identity, employeeId.trim());
      registerIdentity(identity);

      try {
        OfflineSyncQueue.enqueueRegistrationEvent(
          identity,
          employeeId.trim(),
        ).catch(() => {});
      } catch (e) {
        console.warn('[Register] Sync enqueue failed:', e);
      }

      setPhase('success');
      setStatus('success');
    },
    [employeeId, fullName, registerIdentity, setStatus],
  );

  const handleTriggerExtraction = useCallback(() => {
    if (!isFormValid) {
      Alert.alert('Form Incomplete', 'Fill in Employee ID and Full Name.');
      return;
    }
    if (!hasFace) {
      Alert.alert('No Face', 'Position your face in the camera frame.');
      return;
    }

    setPhase('liveness');
    setStatus('detecting');
    setProgress(0);
    livenessEmbeddingRef.current = null;

    // Start real liveness window — accumulates EAR values over 1.5s
    startLivenessWindow(1500);
  }, [isFormValid, hasFace, setStatus]);

  const handleReset = () => {
    setPhase('form');
    setEmployeeId('');
    setFullName('');
    setEmployeeIdError(null);
    setFullNameError(null);
    setProgress(0);
    setLivenessResult(null);
    livenessEmbeddingRef.current = null;
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
          isActive={phase !== 'success'}
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

        {phase === 'saving' && (
          <View style={styles.overlayPurple}>
            <Text style={styles.overlayIcon}>💾</Text>
            <Text style={styles.overlayTitle}>Saving Identity...</Text>
          </View>
        )}

        {phase === 'success' && (
          <View style={styles.overlayGreen}>
            <Text style={styles.overlayIcon}>✅</Text>
            <Text style={styles.overlayTitle}>Registration Complete</Text>
            <Text style={styles.overlaySubtext}>
              {fullName} ({employeeId})
            </Text>
            <View style={styles.successBadgeRow}>
              <View style={styles.successBadge}>
                <Text style={styles.successBadgeText}>~8KB Template</Text>
              </View>
              <View style={[styles.successBadge, styles.successBadgeGreen]}>
                <Text style={styles.successBadgeText}>Self-Learning ✓</Text>
              </View>
            </View>
            <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
              <Text style={styles.resetButtonText}>Register Another</Text>
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
      {phase === 'liveness' && (
        <View style={styles.progressContainer}>
          <View style={[styles.progressBar, {width: `${progress}%`}]} />
        </View>
      )}

      {/* Form */}
      {(phase === 'form' || phase === 'error') && (
        <ScrollView style={styles.formContainer}>
          <Text style={styles.formTitle}>Identity Registration</Text>

          <Text style={styles.inputLabel}>Employee ID</Text>
          <TextInput
            style={[styles.input, employeeIdError ? styles.inputError : null]}
            value={employeeId}
            onChangeText={handleEmployeeIdChange}
            placeholder="e.g. EMP-0042"
            placeholderTextColor="#475569"
            autoCapitalize="none"
            autoCorrect={false}
          />
          {employeeIdError && (
            <Text style={styles.errorLabel}>{employeeIdError}</Text>
          )}

          <Text style={styles.inputLabel}>Full Name</Text>
          <TextInput
            style={[styles.input, fullNameError ? styles.inputError : null]}
            value={fullName}
            onChangeText={handleFullNameChange}
            placeholder="e.g. John Doe"
            placeholderTextColor="#475569"
          />
          {fullNameError && (
            <Text style={styles.errorLabel}>{fullNameError}</Text>
          )}

          <TouchableOpacity
            style={[
              styles.extractButton,
              (!isFormValid || !hasFace) && styles.extractButtonDisabled,
            ]}
            onPress={handleTriggerExtraction}
            disabled={!isFormValid || !hasFace}
            activeOpacity={0.8}>
            <Text style={styles.extractButtonText}>⚡ Register Face</Text>
          </TouchableOpacity>
          {(!isFormValid || !hasFace) && (
            <Text style={styles.statusHelperText}>
              {!hasFace
                ? '⏳ Waiting for face detection...'
                : '⚠️ Please complete the form above.'}
            </Text>
          )}

          {/* Innovation Info */}
          <View style={styles.innovationInfo}>
            <View style={styles.innovationRow}>
              <Text style={styles.innovationDot}>🧬</Text>
              <Text style={styles.innovationText}>
                Geometric landmark vector (~8KB) — not a photo
              </Text>
            </View>
            <View style={styles.innovationRow}>
              <Text style={styles.innovationDot}>🛡️</Text>
              <Text style={styles.innovationText}>
                Passive EAR liveness check blocks spoofing
              </Text>
            </View>
            <View style={styles.innovationRow}>
              <Text style={styles.innovationDot}>☁️</Text>
              <Text style={styles.innovationText}>
                Auto-syncs to AWS when connectivity appears
              </Text>
            </View>
          </View>

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

          {/* Live Detection Metrics */}
          {currentDetection && (
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
  overlayIcon: {fontSize: 40, marginBottom: 12},
  overlayTitle: {color: '#fff', fontSize: 18, fontWeight: '700'},
  overlaySubtext: {color: '#cbd5e1', fontSize: 13, marginTop: 6},

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
    marginBottom: 16,
  },
  inputLabel: {
    color: '#94a3b8',
    fontSize: 13,
    fontWeight: '500',
    marginBottom: 6,
    marginTop: 12,
  },
  input: {
    backgroundColor: '#1e293b',
    borderWidth: 1,
    borderColor: '#334155',
    borderRadius: 10,
    color: '#f1f5f9',
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 15,
  },
  inputError: {borderColor: '#ef4444'},
  errorLabel: {color: '#ef4444', fontSize: 12, marginTop: 4},

  extractButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 24,
  },
  extractButtonDisabled: {backgroundColor: '#334155'},
  extractButtonText: {color: '#fff', fontSize: 15, fontWeight: '700'},

  resetButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 10,
    marginTop: 20,
  },
  resetButtonText: {color: '#fff', fontSize: 14, fontWeight: '600'},

  successBadgeRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 14,
    marginBottom: 4,
  },
  successBadge: {
    backgroundColor: 'rgba(59, 130, 246, 0.25)',
    borderWidth: 1,
    borderColor: '#3b82f6',
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 20,
  },
  successBadgeGreen: {
    backgroundColor: 'rgba(34, 197, 94, 0.2)',
    borderColor: '#22c55e',
  },
  successBadgeText: {
    color: '#e2e8f0',
    fontSize: 11,
    fontWeight: '600',
  },

  innovationInfo: {
    backgroundColor: '#0f172a',
    borderRadius: 10,
    padding: 14,
    marginTop: 18,
    borderWidth: 1,
    borderColor: '#1e3a5f',
  },
  innovationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  innovationDot: {
    fontSize: 14,
    marginRight: 10,
    width: 22,
  },
  innovationText: {
    color: '#64748b',
    fontSize: 12,
    flex: 1,
  },

  statusHelperText: {
    color: '#94a3b8',
    textAlign: 'center',
    marginTop: 8,
    fontSize: 13,
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

export default RegisterFaceScreen;
