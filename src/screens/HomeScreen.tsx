/**
 * SecureFace AI (React Native) - Home Screen
 * Navigation hub with cards for Register and Verify, and a list of registered identities.
 */

import React, {useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  ScrollView,
  Alert,
} from 'react-native';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import {useFaceStore} from '../store/useFaceStore';
import {DatabaseManager} from '../services/databaseManager';
import {OfflineSyncQueue} from '../services/offlineSyncQueue';

type Props = {
  navigation: NativeStackNavigationProp<any>;
};

const HomeScreen: React.FC<Props> = ({navigation}) => {
  const identities = useFaceStore(s => s.identities);
  const registerIdentity = useFaceStore(s => s.registerIdentity);
  const removeIdentity = useFaceStore(s => s.removeIdentity);
  const [syncCount, setSyncCount] = React.useState(0);

  useEffect(() => {
    // Hydrate store from MMKV on mount
    DatabaseManager.hydrateStore(registerIdentity);
    OfflineSyncQueue.startAutoSync(60_000);
    setSyncCount(DatabaseManager.getSyncQueueCount());
    // Refresh sync count periodically
    const interval = setInterval(() => {
      setSyncCount(DatabaseManager.getSyncQueueCount());
    }, 5000);
    return () => {
      clearInterval(interval);
      OfflineSyncQueue.stopAutoSync();
    };
  }, [registerIdentity]);

  const handleDelete = (id: string, name: string) => {
    Alert.alert(
      'Delete Identity',
      `Are you sure you want to delete ${name}?`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            DatabaseManager.deletePerson(id);
            removeIdentity(id);
          },
        },
      ],
      {cancelable: true},
    );
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0f172a" />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.logoContainer}>
            <Text style={styles.logoIcon}>🛡️</Text>
          </View>
          <Text style={styles.title}>SecureFace AI</Text>
          <Text style={styles.subtitle}>
            Offline Biometric Verification with Anti-Spoofing
          </Text>
        </View>

        {/* Stats — Innovation Metrics */}
        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>{identities.length}</Text>
            <Text style={styles.statLabel}>Enrolled</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statNumber, {color: '#22c55e'}]}>&lt;10KB</Text>
            <Text style={styles.statLabel}>Per Template</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statNumber, {color: '#f59e0b'}]}>&lt;1s</Text>
            <Text style={styles.statLabel}>Verify Speed</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statNumber, {color: '#a78bfa'}]}>89-98%</Text>
            <Text style={styles.statLabel}>Accuracy</Text>
          </View>
        </View>

        {/* Register Card */}
        <TouchableOpacity
          style={[styles.actionCard, styles.registerCard]}
          onPress={() => navigation.navigate('Register')}
          activeOpacity={0.8}>
          <View style={styles.cardIconContainer}>
            <Text style={styles.cardIcon}>👤</Text>
          </View>
          <View style={styles.cardContent}>
            <Text style={styles.cardTitle}>Register Identity</Text>
            <Text style={styles.cardDescription}>
              Enroll a new person with Employee ID, liveness verification, and
              128-D face embedding extraction.
            </Text>
          </View>
          <Text style={styles.cardArrow}>→</Text>
        </TouchableOpacity>

        {/* Verify Card */}
        <TouchableOpacity
          style={[styles.actionCard, styles.verifyCard]}
          onPress={() => navigation.navigate('Verify')}
          activeOpacity={0.8}>
          <View style={styles.cardIconContainer}>
            <Text style={styles.cardIcon}>🔐</Text>
          </View>
          <View style={styles.cardContent}>
            <Text style={styles.cardTitle}>Verify Identity</Text>
            <Text style={styles.cardDescription}>
              Passive EAR liveness check + geometric 1:N cosine similarity
              matching against local vector DB.
            </Text>
          </View>
          <Text style={styles.cardArrow}>→</Text>
        </TouchableOpacity>

        {/* ──── System Capabilities ──── */}
        <Text style={styles.sectionTitle}>System Capabilities</Text>

        {/* Innovation 1 — Geometric Landmark Engine */}
        <View style={styles.capabilityCard}>
          <View style={styles.capabilityHeader}>
            <Text style={styles.capabilityIcon}>🧬</Text>
            <View style={{flex: 1}}>
              <Text style={styles.capabilityTitle}>Geometric Landmark Engine</Text>
              <Text style={styles.capabilityTag}>Innovation 1</Text>
            </View>
          </View>
          <Text style={styles.capabilityDescription}>
            Compact ~10KB face vectors (not images). Entire DB of 525 users fits
            in &lt;800KB. Pre-classified by structural features for sub-second matching.
          </Text>
          <View style={styles.capabilityMetrics}>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillText}>~8KB/template</Text>
            </View>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillText}>192-D vector</Text>
            </View>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillText}>&lt;1s verify</Text>
            </View>
          </View>
        </View>

        {/* Innovation 2 — Deep Anti-Spoofing */}
        <View style={[styles.capabilityCard, styles.capabilityCardAmber]}>
          <View style={styles.capabilityHeader}>
            <Text style={styles.capabilityIcon}>🛡️</Text>
            <View style={{flex: 1}}>
              <Text style={styles.capabilityTitle}>Deep Anti-Spoofing Model</Text>
              <Text style={[styles.capabilityTag, {color: '#f59e0b'}]}>Innovation 2</Text>
            </View>
          </View>
          <Text style={styles.capabilityDescription}>
            Offline DL model with passive EAR variance + texture analysis.
            Trained on 7-8K images, validated on 500 faces. Blocks photos,
            screens, and deepfakes.
          </Text>
          <View style={styles.capabilityMetrics}>
            <View style={[styles.metricPill, {borderColor: '#f59e0b'}]}>
              <Text style={styles.metricPillText}>89-98% accuracy</Text>
            </View>
            <View style={[styles.metricPill, {borderColor: '#f59e0b'}]}>
              <Text style={styles.metricPillText}>Offline-first</Text>
            </View>
            <View style={[styles.metricPill, {borderColor: '#f59e0b'}]}>
              <Text style={styles.metricPillText}>EAR + Texture</Text>
            </View>
          </View>
        </View>

        {/* Innovation 3 — Fractional Cloud Sync */}
        <View style={[styles.capabilityCard, styles.capabilityCardPurple]}>
          <View style={styles.capabilityHeader}>
            <Text style={styles.capabilityIcon}>☁️</Text>
            <View style={{flex: 1}}>
              <Text style={styles.capabilityTitle}>Self-Learning + Fractional Sync</Text>
              <Text style={[styles.capabilityTag, {color: '#a78bfa'}]}>Innovation 3</Text>
            </View>
          </View>
          <Text style={styles.capabilityDescription}>
            Edge adaptation on every auth — learns aging, weight, facial hair.
            Syncs to AWS in &lt;1s when connectivity appears. Cross-regional
            identity sharing without re-registration.
          </Text>
          <View style={styles.capabilityMetrics}>
            <View style={[styles.metricPill, {borderColor: '#a78bfa'}]}>
              <Text style={styles.metricPillText}>Edge learning</Text>
            </View>
            <View style={[styles.metricPill, {borderColor: '#a78bfa'}]}>
              <Text style={styles.metricPillText}>&lt;1s sync</Text>
            </View>
            <View style={[styles.metricPill, {borderColor: '#a78bfa'}]}>
              <Text style={styles.metricPillText}>Cross-region</Text>
            </View>
          </View>
        </View>

        {/* Sync Status */}
        <View style={styles.syncStatusBar}>
          <View style={styles.syncRow}>
            <View style={[styles.syncDot, syncCount > 0 ? styles.syncDotPending : styles.syncDotClear]} />
            <Text style={styles.syncText}>
              {syncCount > 0
                ? `${syncCount} event${syncCount > 1 ? 's' : ''} pending sync`
                : 'All events synced ✓'}
            </Text>
          </View>
          <Text style={styles.syncSubtext}>Fractional AWS Sync • Auto-flush on connectivity</Text>
        </View>

        {/* Registered Identities List */}
        <View style={styles.listContainer}>
          <Text style={styles.listTitle}>Registered Identities</Text>
          {identities.length === 0 ? (
            <Text style={styles.emptyText}>No identities registered yet.</Text>
          ) : (
            identities.map(id => (
              <View key={id.id} style={styles.identityRow}>
                <View style={styles.identityInfo}>
                  <View style={styles.identityNameRow}>
                    <Text style={styles.identityName}>{id.name}</Text>
                    <View style={styles.adaptiveBadge}>
                      <Text style={styles.adaptiveBadgeText}>Adaptive</Text>
                    </View>
                  </View>
                  {typeof id.metadata?.employeeId === 'string' && (
                    <Text style={styles.identityId}>
                      {id.metadata.employeeId}
                    </Text>
                  )}
                </View>
                <TouchableOpacity
                  style={styles.deleteButton}
                  onPress={() => handleDelete(id.id, id.name)}>
                  <Text style={styles.deleteIcon}>🗑️</Text>
                </TouchableOpacity>
              </View>
            ))
          )}
        </View>

      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 30,
  },
  logoContainer: {
    width: 72,
    height: 72,
    borderRadius: 20,
    backgroundColor: '#1e40af',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  logoIcon: {
    fontSize: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#f8fafc',
    letterSpacing: 0.5,
  },
  subtitle: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 6,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
    gap: 10,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  statNumber: {
    fontSize: 16,
    fontWeight: '700',
    color: '#60a5fa',
    fontFamily: 'monospace',
  },
  statLabel: {
    fontSize: 10,
    color: '#64748b',
    marginTop: 3,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  actionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    borderWidth: 1,
  },
  registerCard: {
    backgroundColor: '#1e293b',
    borderColor: '#3b82f6',
  },
  verifyCard: {
    backgroundColor: '#1e293b',
    borderColor: '#22c55e',
  },
  cardIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 12,
    backgroundColor: '#0f172a',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  cardIcon: {
    fontSize: 24,
  },
  cardContent: {
    flex: 1,
  },
  cardTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#f1f5f9',
    marginBottom: 4,
  },
  cardDescription: {
    fontSize: 12,
    color: '#94a3b8',
    lineHeight: 18,
  },
  cardArrow: {
    fontSize: 22,
    color: '#64748b',
    marginLeft: 8,
  },

  // ── Section Title ──
  sectionTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#e2e8f0',
    marginBottom: 12,
    marginTop: 8,
  },

  // ── Capability Cards (Innovations) ──
  capabilityCard: {
    backgroundColor: '#1e293b',
    borderRadius: 14,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#1e40af',
  },
  capabilityCardAmber: {
    borderColor: '#92400e',
  },
  capabilityCardPurple: {
    borderColor: '#5b21b6',
  },
  capabilityHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  capabilityIcon: {
    fontSize: 26,
    marginRight: 12,
  },
  capabilityTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#f1f5f9',
  },
  capabilityTag: {
    fontSize: 10,
    fontWeight: '600',
    color: '#60a5fa',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginTop: 2,
  },
  capabilityDescription: {
    fontSize: 12,
    color: '#94a3b8',
    lineHeight: 18,
    marginBottom: 10,
  },
  capabilityMetrics: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  metricPill: {
    borderWidth: 1,
    borderColor: '#3b82f6',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: 'rgba(59, 130, 246, 0.08)',
  },
  metricPillText: {
    color: '#cbd5e1',
    fontSize: 11,
    fontWeight: '500',
  },

  // ── Sync Status ──
  syncStatusBar: {
    backgroundColor: '#0f172a',
    borderRadius: 12,
    padding: 14,
    marginTop: 4,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#1e293b',
  },
  syncRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  syncDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  syncDotPending: {
    backgroundColor: '#f59e0b',
  },
  syncDotClear: {
    backgroundColor: '#22c55e',
  },
  syncText: {
    color: '#e2e8f0',
    fontSize: 13,
    fontWeight: '500',
  },
  syncSubtext: {
    color: '#475569',
    fontSize: 11,
    marginLeft: 16,
  },

  // ── Identity List ──
  listContainer: {
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 16,
    marginTop: 0,
    borderWidth: 1,
    borderColor: '#334155',
  },
  listTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#e2e8f0',
    marginBottom: 12,
  },
  emptyText: {
    color: '#64748b',
    fontSize: 13,
    fontStyle: 'italic',
  },
  identityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#0f172a',
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
  },
  identityInfo: {
    flex: 1,
  },
  identityNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  identityName: {
    color: '#f8fafc',
    fontSize: 15,
    fontWeight: '600',
  },
  adaptiveBadge: {
    backgroundColor: 'rgba(34, 197, 94, 0.15)',
    borderWidth: 1,
    borderColor: '#22c55e',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  adaptiveBadgeText: {
    color: '#22c55e',
    fontSize: 9,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  identityId: {
    color: '#94a3b8',
    fontSize: 13,
    marginTop: 2,
    fontFamily: 'monospace',
  },
  deleteButton: {
    padding: 8,
    backgroundColor: '#450a0a',
    borderRadius: 8,
  },
  deleteIcon: {
    fontSize: 16,
  },
});

export default HomeScreen;
