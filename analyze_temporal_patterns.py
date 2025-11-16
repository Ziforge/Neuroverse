#!/usr/bin/env python3
"""
Temporal Pattern Analysis
Look at how participants adapt over time within their session.
Early vs Late behavior can indicate learning/adaptation patterns.
"""

import re
import json
import numpy as np
from pathlib import Path
from datetime import datetime

def parse_temporal_data(filepath):
    """Parse data with temporal segmentation."""
    data = {
        'volume_events': [],  # (timestamp, value)
        'mute_events': [],    # timestamps
        'session_start': None,
        'session_end': None,
    }

    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()

    first_ts = None
    last_ts = None

    for line in lines:
        ts_match = re.search(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', line)
        if ts_match:
            ts_str = ts_match.group(1)
            try:
                ts = datetime.strptime(ts_str, '%Y/%m/%d %H:%M:%S.%f')
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            except:
                continue

            # Volume
            vol_match = re.search(r'Volume_VolSlider_\w+:\s*([\d.]+)%', line)
            if vol_match:
                data['volume_events'].append((ts, float(vol_match.group(1))))

            # Mute
            if 'MuteAllAudio: True' in line or ': Muted' in line:
                data['mute_events'].append(ts)

    data['session_start'] = first_ts
    data['session_end'] = last_ts

    if first_ts and last_ts:
        data['session_duration'] = (last_ts - first_ts).total_seconds() / 60.0
    else:
        data['session_duration'] = 10.0

    return data


def compute_temporal_features(data):
    """Compute early vs late session features."""
    features = {}

    if not data['session_start'] or not data['session_end']:
        return None

    session_start = data['session_start']
    session_duration_sec = (data['session_end'] - session_start).total_seconds()

    if session_duration_sec < 60:  # Less than 1 minute
        return None

    midpoint_sec = session_duration_sec / 2

    # Split volume events into early and late
    early_volumes = []
    late_volumes = []
    first_third_volumes = []
    last_third_volumes = []

    for ts, vol in data['volume_events']:
        elapsed = (ts - session_start).total_seconds()
        if elapsed < midpoint_sec:
            early_volumes.append(vol)
        else:
            late_volumes.append(vol)

        # Also track thirds
        if elapsed < session_duration_sec / 3:
            first_third_volumes.append(vol)
        elif elapsed > 2 * session_duration_sec / 3:
            last_third_volumes.append(vol)

    if not early_volumes or not late_volumes:
        return None

    features['early_avg_volume'] = np.mean(early_volumes)
    features['late_avg_volume'] = np.mean(late_volumes)
    features['volume_change'] = features['late_avg_volume'] - features['early_avg_volume']

    features['early_volume_std'] = np.std(early_volumes)
    features['late_volume_std'] = np.std(late_volumes)
    features['stability_change'] = features['early_volume_std'] - features['late_volume_std']

    if first_third_volumes and last_third_volumes:
        features['first_third_volume'] = np.mean(first_third_volumes)
        features['last_third_volume'] = np.mean(last_third_volumes)
        features['exploration_to_settling'] = features['last_third_volume'] - features['first_third_volume']
    else:
        features['first_third_volume'] = features['early_avg_volume']
        features['last_third_volume'] = features['late_avg_volume']
        features['exploration_to_settling'] = features['volume_change']

    # Muting patterns
    early_mutes = 0
    late_mutes = 0
    for ts in data['mute_events']:
        elapsed = (ts - session_start).total_seconds()
        if elapsed < midpoint_sec:
            early_mutes += 1
        else:
            late_mutes += 1

    half_duration_min = (session_duration_sec / 2) / 60
    features['early_mutes_per_min'] = early_mutes / max(half_duration_min, 0.1)
    features['late_mutes_per_min'] = late_mutes / max(half_duration_min, 0.1)
    features['muting_change'] = features['late_mutes_per_min'] - features['early_mutes_per_min']

    features['session_duration'] = data['session_duration']

    return features


def classify_adaptation_pattern(features):
    """
    Classify the adaptation pattern based on temporal changes.
    """
    patterns = []

    # Volume trajectory
    if features['volume_change'] > 5:
        patterns.append('VOLUME_INCREASE (seeking more stimulation)')
    elif features['volume_change'] < -5:
        patterns.append('VOLUME_DECREASE (reducing stimulation)')
    else:
        patterns.append('VOLUME_STABLE (consistent preference)')

    # Stability change
    if features['stability_change'] > 3:
        patterns.append('SETTLING_DOWN (less exploration over time)')
    elif features['stability_change'] < -3:
        patterns.append('INCREASING_SEARCH (more exploration over time)')
    else:
        patterns.append('CONSISTENT_EXPLORATION')

    # Muting pattern
    if features['muting_change'] > 0.3:
        patterns.append('FATIGUE (more muting later)')
    elif features['muting_change'] < -0.3:
        patterns.append('ADAPTATION (less muting later)')
    else:
        patterns.append('STABLE_MUTING')

    # Combined interpretation
    if 'VOLUME_DECREASE' in patterns[0] and 'SETTLING_DOWN' in patterns[1]:
        interpretation = 'HYPERSENSITIVE_PATTERN: Reducing stimulation, finding safe zone'
    elif 'VOLUME_INCREASE' in patterns[0] and 'SETTLING_DOWN' in patterns[1]:
        interpretation = 'HYPOSENSITIVE_PATTERN: Increasing stimulation until satisfied'
    elif 'STABLE' in patterns[0] and 'SETTLING_DOWN' in patterns[1]:
        interpretation = 'TYPICAL_PATTERN: Quick preference identification'
    elif 'FATIGUE' in patterns[2]:
        interpretation = 'SENSORY_FATIGUE: Possible overstimulation over time'
    elif 'ADAPTATION' in patterns[2]:
        interpretation = 'POSITIVE_ADAPTATION: Becoming more comfortable'
    else:
        interpretation = 'MIXED_PATTERN: No clear adaptation trajectory'

    return patterns, interpretation


def main():
    test_data_dir = Path("/tmp/Neuroverse/Test Data")

    print("="*80)
    print("TEMPORAL PATTERN ANALYSIS")
    print("How participants adapt over their session")
    print("="*80)

    participants = []

    for filepath in sorted(test_data_dir.glob("*.txt")):
        if 'PARTIALLY' in filepath.stem or 'POSTEND' in filepath.stem:
            continue

        participant_id = filepath.stem.upper()
        data = parse_temporal_data(filepath)
        features = compute_temporal_features(data)

        if features:
            patterns, interpretation = classify_adaptation_pattern(features)
            participants.append({
                'id': participant_id,
                'features': features,
                'patterns': patterns,
                'interpretation': interpretation
            })
            print(f"Parsed: {participant_id} (Duration: {features['session_duration']:.1f} min)")

    print(f"\nTotal valid participants: {len(participants)}")

    # Summary statistics
    print("\n" + "="*80)
    print("COHORT TEMPORAL PATTERNS")
    print("="*80)

    vol_changes = [p['features']['volume_change'] for p in participants]
    stability_changes = [p['features']['stability_change'] for p in participants]
    muting_changes = [p['features']['muting_change'] for p in participants]

    print(f"\nVolume Change (Late - Early):")
    print(f"  Mean: {np.mean(vol_changes):+.1f}% (Negative = decreasing)")
    print(f"  Range: [{np.min(vol_changes):+.1f}% to {np.max(vol_changes):+.1f}%]")

    print(f"\nStability Change (Early STD - Late STD):")
    print(f"  Mean: {np.mean(stability_changes):+.1f} (Positive = settling down)")
    print(f"  Range: [{np.min(stability_changes):+.1f} to {np.max(stability_changes):+.1f}]")

    print(f"\nMuting Change (Late/min - Early/min):")
    print(f"  Mean: {np.mean(muting_changes):+.2f} (Positive = more muting later)")
    print(f"  Range: [{np.min(muting_changes):+.2f} to {np.max(muting_changes):+.2f}]")

    # Individual patterns
    print("\n" + "="*80)
    print("INDIVIDUAL TEMPORAL PATTERNS")
    print("="*80)

    for p in participants:
        print(f"\n{p['id']}")
        print(f"  Early Volume: {p['features']['early_avg_volume']:.1f}% → Late: {p['features']['late_avg_volume']:.1f}%")
        print(f"  Volume Change: {p['features']['volume_change']:+.1f}%")
        print(f"  Early STD: {p['features']['early_volume_std']:.1f} → Late: {p['features']['late_volume_std']:.1f}")
        print(f"  Stability Change: {p['features']['stability_change']:+.1f}")
        print(f"  Early Mutes/min: {p['features']['early_mutes_per_min']:.2f} → Late: {p['features']['late_mutes_per_min']:.2f}")
        print(f"  Patterns: {', '.join(p['patterns'])}")
        print(f"  Interpretation: {p['interpretation']}")

    # Group by interpretation
    print("\n" + "="*80)
    print("ADAPTATION PATTERN SUMMARY")
    print("="*80)

    from collections import Counter
    interpretations = Counter([p['interpretation'] for p in participants])

    for pattern, count in interpretations.most_common():
        pct = count / len(participants) * 100
        print(f"\n{pattern}")
        print(f"  {count} participants ({pct:.1f}%)")

        group = [p for p in participants if p['interpretation'] == pattern]
        avg_vol_change = np.mean([p['features']['volume_change'] for p in group])
        avg_early_vol = np.mean([p['features']['early_avg_volume'] for p in group])
        avg_late_vol = np.mean([p['features']['late_avg_volume'] for p in group])

        print(f"  Avg Early Vol: {avg_early_vol:.1f}%")
        print(f"  Avg Late Vol: {avg_late_vol:.1f}%")
        print(f"  Avg Change: {avg_vol_change:+.1f}%")

    # Save results
    output = {
        'method': 'Temporal Pattern Analysis',
        'n_participants': len(participants),
        'cohort_summary': {
            'mean_volume_change': float(np.mean(vol_changes)),
            'mean_stability_change': float(np.mean(stability_changes)),
            'mean_muting_change': float(np.mean(muting_changes))
        },
        'pattern_distribution': dict(interpretations),
        'individual_results': [
            {
                'id': p['id'],
                **{k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in p['features'].items()},
                'patterns': p['patterns'],
                'interpretation': p['interpretation']
            }
            for p in participants
        ]
    }

    with open('/tmp/Neuroverse/temporal_patterns_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /tmp/Neuroverse/temporal_patterns_results.json")


if __name__ == "__main__":
    main()
