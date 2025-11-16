#!/usr/bin/env python3
"""
Final Preferences Analysis
Focus on what participants settled on, not continuous tracking data.
Detects discrete adjustment sessions and final stable states.
"""

import re
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def parse_final_preferences(filepath):
    """Extract final/stable preferences from session."""
    data = {
        'volume_history': [],
        'mute_events': 0,
        'pitch_history': [],
        'delay_history': [],
        'saturation_history': [],
        'eq_final_gains': {},  # Final EQ settings per source
        'timestamps': []
    }

    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()

    current_eq_gains = {}  # Track latest EQ for each handle

    for line in lines:
        ts_match = re.search(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', line)
        timestamp = ts_match.group(1) if ts_match else None
        if timestamp:
            data['timestamps'].append(timestamp)

        # Volume
        vol_match = re.search(r'Volume_VolSlider_(\w+):\s*([\d.]+)%', line)
        if vol_match:
            source = vol_match.group(1)
            vol = float(vol_match.group(2))
            data['volume_history'].append({'source': source, 'value': vol, 'ts': timestamp})

        # Mute
        if 'MuteAllAudio: True' in line or ': Muted' in line:
            data['mute_events'] += 1

        # Pitch
        pitch_match = re.search(r'PitchShift_PitchSlider_?(\w+):\s*([\d.]+)x', line)
        if pitch_match:
            source = pitch_match.group(1)
            pitch = float(pitch_match.group(2))
            data['pitch_history'].append({'source': source, 'value': pitch})

        # Delay
        delay_match = re.search(r'Delay_DelaySlider_(\w+):\s*([\d.]+)%', line)
        if delay_match:
            data['delay_history'].append(float(delay_match.group(2)))

        # Saturation
        sat_match = re.search(r'Saturation_SaturationSlider_(\w+):\s*([\d.]+)%', line)
        if sat_match:
            data['saturation_history'].append(float(sat_match.group(2)))

        # EQ - just track the final state per handle
        eq_match = re.search(r'EQ_(\w+):\s*Gain:([\d.-]+)dB,Frequency:([\d.]+)Hz', line)
        if eq_match:
            handle = eq_match.group(1)
            gain = float(eq_match.group(2))
            freq = float(eq_match.group(3))
            current_eq_gains[handle] = {'gain': gain, 'freq': freq}

    data['eq_final_gains'] = current_eq_gains

    return data


def compute_preference_features(raw_data):
    """Compute features based on final preferences and discrete changes."""
    features = {}

    # Session duration
    if len(raw_data['timestamps']) >= 2:
        try:
            first = datetime.strptime(raw_data['timestamps'][0], '%Y/%m/%d %H:%M:%S.%f')
            last = datetime.strptime(raw_data['timestamps'][-1], '%Y/%m/%d %H:%M:%S.%f')
            features['session_duration'] = (last - first).total_seconds() / 60.0
        except:
            features['session_duration'] = 10.0
    else:
        features['session_duration'] = 10.0

    # VOLUME PREFERENCES
    if raw_data['volume_history']:
        # Use last N% of volume readings as "final preference"
        vol_values = [v['value'] for v in raw_data['volume_history']]
        n_final = max(1, len(vol_values) // 5)  # Last 20%
        final_volumes = vol_values[-n_final:]

        features['final_volume'] = np.mean(final_volumes)
        features['final_volume_stability'] = np.std(final_volumes)  # Lower = more stable

        # Overall volume pattern
        features['avg_volume'] = np.mean(vol_values)
        features['volume_range'] = max(vol_values) - min(vol_values)

        # Discrete volume changes (filter out micro-adjustments)
        discrete_changes = 0
        prev_vol = vol_values[0]
        for vol in vol_values[1:]:
            if abs(vol - prev_vol) > 2.0:  # Only count changes > 2%
                discrete_changes += 1
            prev_vol = vol
        features['discrete_volume_changes'] = discrete_changes
        features['volume_changes_per_minute'] = discrete_changes / max(features['session_duration'], 0.1)
    else:
        features['final_volume'] = 50.0
        features['final_volume_stability'] = 0.0
        features['avg_volume'] = 50.0
        features['volume_range'] = 0.0
        features['discrete_volume_changes'] = 0
        features['volume_changes_per_minute'] = 0

    # MUTING
    features['mute_events'] = raw_data['mute_events']
    features['mutes_per_minute'] = raw_data['mute_events'] / max(features['session_duration'], 0.1)

    # PITCH PREFERENCE
    if raw_data['pitch_history']:
        pitches = [p['value'] for p in raw_data['pitch_history']]
        features['final_pitch'] = pitches[-1]
        features['avg_pitch'] = np.mean(pitches)
        features['pitch_deviation_from_natural'] = abs(1.0 - features['final_pitch'])
    else:
        features['final_pitch'] = 1.0
        features['avg_pitch'] = 1.0
        features['pitch_deviation_from_natural'] = 0.0

    # DELAY PREFERENCE
    if raw_data['delay_history']:
        features['final_delay'] = raw_data['delay_history'][-1]
        features['avg_delay'] = np.mean(raw_data['delay_history'])
    else:
        features['final_delay'] = 20.0
        features['avg_delay'] = 20.0

    # SATURATION PREFERENCE
    if raw_data['saturation_history']:
        features['final_saturation'] = raw_data['saturation_history'][-1]
        features['avg_saturation'] = np.mean(raw_data['saturation_history'])
    else:
        features['final_saturation'] = 50.0
        features['avg_saturation'] = 50.0

    # EQ PREFERENCES (final state)
    if raw_data['eq_final_gains']:
        gains = [eq['gain'] for eq in raw_data['eq_final_gains'].values()]
        freqs = [eq['freq'] for eq in raw_data['eq_final_gains'].values()]
        features['final_eq_gain'] = np.mean(gains)
        features['final_eq_freq'] = np.mean(freqs)

        # Categorize frequency preference
        low_count = sum(1 for f in freqs if f < 500)
        high_count = sum(1 for f in freqs if f >= 2000)
        total = len(freqs)
        features['eq_low_freq_preference'] = low_count / total if total > 0 else 0.33
        features['eq_high_freq_preference'] = high_count / total if total > 0 else 0.33
    else:
        features['final_eq_gain'] = 0.0
        features['final_eq_freq'] = 1000.0
        features['eq_low_freq_preference'] = 0.33
        features['eq_high_freq_preference'] = 0.33

    return features


def classify_by_preferences(all_features):
    """Use K-means clustering on preference features."""

    # Select key features for clustering
    feature_keys = [
        'final_volume',
        'volume_changes_per_minute',
        'mutes_per_minute',
        'final_delay',
        'final_saturation',
        'final_pitch',
        'final_eq_gain'
    ]

    # Build feature matrix
    X = []
    valid_participants = []

    for p in all_features:
        # Check if we have enough data
        if p['features']['discrete_volume_changes'] > 0 or p['features']['mute_events'] > 0:
            row = [p['features'][k] for k in feature_keys]
            X.append(row)
            valid_participants.append(p)

    if len(X) < 3:
        return None, "Not enough valid data for clustering"

    X = np.array(X)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Interpret clusters by their centers
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

    # Label clusters based on volume preference (primary indicator)
    cluster_volumes = [(i, centers_original[i][0]) for i in range(3)]
    cluster_volumes.sort(key=lambda x: x[1])

    labels = {}
    labels[cluster_volumes[0][0]] = 'Hypersensitive'  # Lowest volume
    labels[cluster_volumes[1][0]] = 'Typical'         # Middle volume
    labels[cluster_volumes[2][0]] = 'Hyposensitive'   # Highest volume

    # Assign labels
    for i, p in enumerate(valid_participants):
        cluster_id = clusters[i]
        p['classification'] = labels[cluster_id]
        p['cluster_id'] = int(cluster_id)

    return valid_participants, labels, centers_original, feature_keys


def main():
    test_data_dir = Path("/tmp/Neuroverse/Test Data")

    print("="*80)
    print("FINAL PREFERENCES ANALYSIS")
    print("Classifying based on what participants settled on")
    print("="*80)

    # Parse all participants
    all_data = []

    for filepath in sorted(test_data_dir.glob("*.txt")):
        if 'PARTIALLY' in filepath.stem or 'POSTEND' in filepath.stem:
            continue

        participant_id = filepath.stem.upper()
        raw = parse_final_preferences(filepath)
        features = compute_preference_features(raw)

        all_data.append({
            'id': participant_id,
            'features': features
        })

        print(f"Parsed: {participant_id}")
        print(f"  Final Vol: {features['final_volume']:.1f}%, Discrete changes: {features['discrete_volume_changes']}")

    print(f"\nTotal participants: {len(all_data)}")

    # Cluster
    print("\n" + "-"*80)
    print("CLUSTERING BY FINAL PREFERENCES")
    print("-"*80)

    result = classify_by_preferences(all_data)

    if result[0] is None:
        print(f"Error: {result[1]}")
        return

    valid_participants, labels, centers, feature_keys = result

    # Show cluster profiles
    print("\nCLUSTER CENTERS (Original Scale):")
    for cluster_id in range(3):
        label = labels[cluster_id]
        center = centers[cluster_id]
        print(f"\n{label} (Cluster {cluster_id}):")
        for i, key in enumerate(feature_keys):
            print(f"  {key:30s}: {center[i]:.2f}")

    # Individual results
    print("\n" + "-"*80)
    print("INDIVIDUAL CLASSIFICATIONS")
    print("-"*80)

    for p in valid_participants:
        print(f"\n{p['id']}: {p['classification']}")
        f = p['features']
        print(f"  Final Volume: {f['final_volume']:.1f}% (avg: {f['avg_volume']:.1f}%)")
        print(f"  Volume Changes/min: {f['volume_changes_per_minute']:.1f}")
        print(f"  Mutes/min: {f['mutes_per_minute']:.2f}")
        print(f"  Final Delay: {f['final_delay']:.1f}%, Saturation: {f['final_saturation']:.1f}%")
        print(f"  Final Pitch: {f['final_pitch']:.2f}x")
        print(f"  Final EQ Gain: {f['final_eq_gain']:.2f} dB")

    # Summary
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY")
    print("="*80)

    from collections import Counter
    counts = Counter([p['classification'] for p in valid_participants])

    for cls in ['Hypersensitive', 'Typical', 'Hyposensitive']:
        count = counts.get(cls, 0)
        pct = (count / len(valid_participants)) * 100 if valid_participants else 0
        print(f"\n{cls}: {count} ({pct:.1f}%)")

        group = [p for p in valid_participants if p['classification'] == cls]
        if group:
            vol = np.mean([p['features']['final_volume'] for p in group])
            delay = np.mean([p['features']['final_delay'] for p in group])
            sat = np.mean([p['features']['final_saturation'] for p in group])
            pitch = np.mean([p['features']['final_pitch'] for p in group])
            mutes = np.mean([p['features']['mutes_per_minute'] for p in group])

            print(f"  Final Volume: {vol:.1f}%")
            print(f"  Final Delay: {delay:.1f}%")
            print(f"  Final Saturation: {sat:.1f}%")
            print(f"  Final Pitch: {pitch:.2f}x")
            print(f"  Mutes/min: {mutes:.2f}")

    # Save
    output = {
        'method': 'K-Means on Final Preferences',
        'n_participants': len(valid_participants),
        'classification_counts': dict(counts),
        'cluster_centers': {
            labels[i]: dict(zip(feature_keys, centers[i].tolist()))
            for i in range(3)
        },
        'individual_results': [
            {
                'id': p['id'],
                'classification': p['classification'],
                **{k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in p['features'].items()}
            }
            for p in valid_participants
        ]
    }

    with open('/tmp/Neuroverse/final_preferences_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /tmp/Neuroverse/final_preferences_results.json")

    # Key insight
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("This analysis focuses on WHAT participants chose (final preferences)")
    print("rather than HOW they got there (adjustment frequency).")
    print("")
    print("Hypersensitive: Lower volume, lower effects, more natural pitch")
    print("Hyposensitive: Higher volume, higher effects, modified pitch")
    print("Typical: Middle range on all parameters")


if __name__ == "__main__":
    main()
