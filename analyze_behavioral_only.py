#!/usr/bin/env python3
"""
Neuroverse Behavioral-Only Classification
Classifies based purely on behavioral patterns, ignoring questionnaire responses.
Uses K-means clustering on normalized behavioral features.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def parse_participant_file(filepath):
    """Parse Unity debug log for behavioral data only."""
    data = {
        'volume_history': [],
        'mute_events': 0,
        'timestamps': [],
        'total_adjustments': 0,
        'reverb_settings': [],
        'delay_settings': [],
        'saturation_settings': [],
    }

    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()

    for line in lines:
        # Timestamps
        ts_match = re.search(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', line)
        if ts_match:
            data['timestamps'].append(ts_match.group(1))

        # Volume: Volume_VolSlider_Ocean: 66.89528%
        vol_match = re.search(r'Volume_VolSlider_\w+:\s*([\d.]+)%', line)
        if vol_match:
            data['volume_history'].append(float(vol_match.group(1)))
            data['total_adjustments'] += 1

        # Mute events
        if 'MuteAllAudio: True' in line or ': Muted' in line:
            data['mute_events'] += 1

        # Reverb slider: Reverb_ReverbSlider_X: 50%
        reverb_match = re.search(r'Reverb_ReverbSlider_\w+:\s*([\d.]+)%', line)
        if reverb_match:
            data['reverb_settings'].append(float(reverb_match.group(1)))

        # Delay slider
        delay_match = re.search(r'Delay_DelaySlider_\w+:\s*([\d.]+)%', line)
        if delay_match:
            data['delay_settings'].append(float(delay_match.group(1)))

        # Saturation slider
        sat_match = re.search(r'Saturation_SaturationSlider_\w+:\s*([\d.]+)%', line)
        if sat_match:
            data['saturation_settings'].append(float(sat_match.group(1)))

    # Calculate session duration
    if len(data['timestamps']) >= 2:
        try:
            first = datetime.strptime(data['timestamps'][0], '%Y/%m/%d %H:%M:%S.%f')
            last = datetime.strptime(data['timestamps'][-1], '%Y/%m/%d %H:%M:%S.%f')
            data['session_duration_minutes'] = (last - first).total_seconds() / 60.0
        except:
            data['session_duration_minutes'] = 10.0
    else:
        data['session_duration_minutes'] = 10.0

    # Derived metrics
    if data['volume_history']:
        data['avg_volume'] = np.mean(data['volume_history'])
        data['volume_std'] = np.std(data['volume_history'])
        data['volume_range'] = max(data['volume_history']) - min(data['volume_history'])
        data['final_volume'] = data['volume_history'][-1]
    else:
        data['avg_volume'] = 50.0
        data['volume_std'] = 0.0
        data['volume_range'] = 0.0
        data['final_volume'] = 50.0

    # Per-minute metrics
    duration = max(data['session_duration_minutes'], 0.1)
    data['mutes_per_minute'] = data['mute_events'] / duration
    data['adjustments_per_minute'] = data['total_adjustments'] / duration

    # Effect preferences (if available)
    data['avg_reverb'] = np.mean(data['reverb_settings']) if data['reverb_settings'] else 50.0
    data['avg_delay'] = np.mean(data['delay_settings']) if data['delay_settings'] else 20.0
    data['avg_saturation'] = np.mean(data['saturation_settings']) if data['saturation_settings'] else 50.0

    return data


def extract_features(participant_data):
    """Extract feature vector for clustering."""
    return [
        participant_data['avg_volume'],
        participant_data['volume_std'],
        participant_data['volume_range'],
        participant_data['mutes_per_minute'],
        participant_data['adjustments_per_minute'],
        participant_data['avg_reverb'],
        participant_data['avg_delay'],
        participant_data['avg_saturation'],
    ]


def interpret_clusters(cluster_centers, feature_names):
    """
    Interpret cluster centers to assign sensory labels.
    Hypersensitive: Low volume, high adjustment rate, low effects
    Hyposensitive: High volume, low adjustment rate, high effects
    Typical: Middle values
    """
    interpretations = {}

    for i, center in enumerate(cluster_centers):
        scores = {
            'hyper': 0,
            'hypo': 0,
            'typical': 0
        }

        # Volume (index 0) - lower = hyper, higher = hypo
        vol = center[0]
        if vol < -0.3:
            scores['hyper'] += 3
        elif vol > 0.3:
            scores['hypo'] += 3
        else:
            scores['typical'] += 2

        # Volume std (index 1) - higher = hyper (more adjustment)
        if center[1] > 0.3:
            scores['hyper'] += 2
        elif center[1] < -0.3:
            scores['hypo'] += 1
        else:
            scores['typical'] += 1

        # Adjustment rate (index 4) - higher = hyper
        if center[4] > 0.5:
            scores['hyper'] += 3
        elif center[4] < -0.3:
            scores['hypo'] += 2
        else:
            scores['typical'] += 2

        # Mutes per minute (index 3)
        if center[3] > 0.5:
            scores['hyper'] += 2  # Frequent muting = sensitivity
        elif center[3] < -0.3:
            scores['typical'] += 1

        # Reverb (index 5) - higher = hypo (seeking stimulation)
        if center[5] > 0.3:
            scores['hypo'] += 2
        elif center[5] < -0.3:
            scores['hyper'] += 2
        else:
            scores['typical'] += 1

        # Saturation (index 7) - higher = hypo
        if center[7] > 0.3:
            scores['hypo'] += 2
        elif center[7] < -0.3:
            scores['hyper'] += 2
        else:
            scores['typical'] += 1

        # Assign label based on highest score
        label = max(scores, key=scores.get)
        interpretations[i] = {
            'label': label.capitalize(),
            'scores': scores,
            'center': center.tolist()
        }

    return interpretations


def main():
    test_data_dir = Path("/tmp/Neuroverse/Test Data")

    print("="*80)
    print("NEUROVERSE BEHAVIORAL-ONLY CLASSIFICATION")
    print("Using K-Means Clustering on Normalized Behavioral Features")
    print("="*80)

    # Parse all participants
    participants = []
    for filepath in sorted(test_data_dir.glob("*.txt")):
        # Skip metadata files
        if 'PARTIALLY' in filepath.stem or 'POSTEND' in filepath.stem:
            continue

        participant_id = filepath.stem.upper()
        data = parse_participant_file(filepath)

        # Only include if we have behavioral data
        if data['total_adjustments'] > 0 or data['mute_events'] > 0:
            participants.append({
                'id': participant_id,
                'data': data
            })
            print(f"Parsed: {participant_id} (Vol: {data['avg_volume']:.1f}%, Adj/min: {data['adjustments_per_minute']:.1f})")

    print(f"\nTotal participants with behavioral data: {len(participants)}")

    # Extract features
    feature_names = ['avg_volume', 'volume_std', 'volume_range', 'mutes_per_min',
                     'adj_per_min', 'avg_reverb', 'avg_delay', 'avg_saturation']

    X = np.array([extract_features(p['data']) for p in participants])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Show cohort statistics
    print("\n" + "-"*80)
    print("COHORT STATISTICS (Raw Values)")
    print("-"*80)
    for i, name in enumerate(feature_names):
        values = X[:, i]
        print(f"{name:20s}: Mean={np.mean(values):7.2f}, Std={np.std(values):7.2f}, Range=[{np.min(values):.1f} - {np.max(values):.1f}]")

    # K-Means clustering with 3 clusters
    print("\n" + "-"*80)
    print("K-MEANS CLUSTERING (3 groups)")
    print("-"*80)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Interpret clusters
    cluster_interpretations = interpret_clusters(kmeans.cluster_centers_, feature_names)

    # Ensure we have one of each type
    assigned_labels = [cluster_interpretations[i]['label'] for i in range(3)]
    print(f"Cluster interpretations: {assigned_labels}")

    # If duplicate labels, reassign based on characteristics
    if len(set(assigned_labels)) < 3:
        # Sort clusters by volume (center[0])
        sorted_clusters = sorted(range(3), key=lambda i: kmeans.cluster_centers_[i][0])
        cluster_interpretations[sorted_clusters[0]]['label'] = 'Hypersensitive'
        cluster_interpretations[sorted_clusters[1]]['label'] = 'Typical'
        cluster_interpretations[sorted_clusters[2]]['label'] = 'Hyposensitive'

    # Assign classifications
    results = []
    for i, p in enumerate(participants):
        cluster_id = clusters[i]
        classification = cluster_interpretations[cluster_id]['label']

        results.append({
            'id': p['id'],
            'data': p['data'],
            'cluster': int(cluster_id),
            'classification': classification
        })

    # Print cluster profiles
    print("\nCLUSTER PROFILES (Standardized Centers):")
    for cluster_id, interp in cluster_interpretations.items():
        print(f"\n{interp['label'].upper()} (Cluster {cluster_id}):")
        for j, (name, val) in enumerate(zip(feature_names, interp['center'])):
            direction = "↑" if val > 0.3 else "↓" if val < -0.3 else "→"
            print(f"  {name:20s}: {val:+.2f} {direction}")

    # Show individual results
    print("\n" + "-"*80)
    print("INDIVIDUAL CLASSIFICATIONS")
    print("-"*80)

    for result in results:
        p = result['data']
        print(f"\n{result['id']}")
        print(f"  Classification: {result['classification']}")
        print(f"  Volume: {p['avg_volume']:.1f}% (std: {p['volume_std']:.1f})")
        print(f"  Adjustments/min: {p['adjustments_per_minute']:.1f}")
        print(f"  Mutes/min: {p['mutes_per_minute']:.2f}")
        print(f"  Reverb: {p['avg_reverb']:.1f}%, Delay: {p['avg_delay']:.1f}%, Saturation: {p['avg_saturation']:.1f}%")

    # Summary
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY")
    print("="*80)

    from collections import Counter
    counts = Counter([r['classification'] for r in results])
    total = len(results)

    for cls in ['Hypersensitive', 'Typical', 'Hyposensitive']:
        count = counts.get(cls, 0)
        pct = (count / total) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")

        # Show actual averages for this group
        group_data = [r['data'] for r in results if r['classification'] == cls]
        if group_data:
            avg_vol = np.mean([d['avg_volume'] for d in group_data])
            avg_adj = np.mean([d['adjustments_per_minute'] for d in group_data])
            avg_mutes = np.mean([d['mutes_per_minute'] for d in group_data])
            print(f"    Avg Volume: {avg_vol:.1f}%")
            print(f"    Avg Adj/min: {avg_adj:.1f}")
            print(f"    Avg Mutes/min: {avg_mutes:.2f}")

    # Save results
    output = {
        'method': 'K-Means Clustering on Behavioral Features',
        'n_participants': len(results),
        'classification_counts': dict(counts),
        'cluster_profiles': {
            interp['label']: {
                'cluster_id': cid,
                'standardized_center': interp['center'],
                'feature_names': feature_names
            }
            for cid, interp in cluster_interpretations.items()
        },
        'individual_results': [
            {
                'id': r['id'],
                'classification': r['classification'],
                'cluster_id': r['cluster'],
                'avg_volume': r['data']['avg_volume'],
                'volume_std': r['data']['volume_std'],
                'adjustments_per_minute': r['data']['adjustments_per_minute'],
                'mutes_per_minute': r['data']['mutes_per_minute'],
                'avg_reverb': r['data']['avg_reverb'],
                'avg_delay': r['data']['avg_delay'],
                'avg_saturation': r['data']['avg_saturation']
            }
            for r in results
        ]
    }

    with open('/tmp/Neuroverse/behavioral_clustering_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /tmp/Neuroverse/behavioral_clustering_results.json")

    # Recommendations
    print("\n" + "="*80)
    print("INTERPRETATION NOTES")
    print("="*80)
    print("""
This classification uses ONLY behavioral data:
- Volume preferences (what they actually set, not what they said)
- Adjustment frequency (how often they changed settings)
- Muting behavior (how often they needed to mute)
- Effect levels (reverb, delay, saturation settings)

The K-means algorithm finds natural groupings in the data without
imposing predetermined thresholds. The labels (Hyper/Typical/Hypo)
are assigned based on which cluster shows:

- HYPERSENSITIVE: Lower volumes, higher adjustment rates, less effects
- HYPOSENSITIVE: Higher volumes, lower adjustment rates, more effects
- TYPICAL: Middle values across features

This approach is more objective than rule-based classification because
it lets the data define the groups.
""")


if __name__ == "__main__":
    main()
