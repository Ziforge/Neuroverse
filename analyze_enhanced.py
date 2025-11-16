#!/usr/bin/env python3
"""
Enhanced Neuroverse Analysis
Extracts ALL available behavioral features including:
- EQ adjustments (gain, frequency preferences)
- Pitch shift patterns
- Temporal dynamics (adjustment speed, hesitation)
- Session phases (early vs late behavior)
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def parse_enhanced_features(filepath):
    """Extract comprehensive features from Unity logs."""
    data = {
        'volume_history': [],
        'mute_events': 0,
        'timestamps': [],
        'eq_adjustments': [],  # (timestamp, gain, frequency)
        'pitch_shifts': [],    # (timestamp, multiplier)
        'delay_values': [],
        'saturation_values': [],
    }

    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()

    for line in lines:
        # Timestamp
        ts_match = re.search(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', line)
        timestamp = ts_match.group(1) if ts_match else None

        # Volume
        vol_match = re.search(r'Volume_VolSlider_\w+:\s*([\d.]+)%', line)
        if vol_match:
            data['volume_history'].append((timestamp, float(vol_match.group(1))))

        # Mutes
        if 'MuteAllAudio: True' in line or ': Muted' in line:
            data['mute_events'] += 1

        # EQ - Gain:2.088618dB,Frequency:88.46996Hz
        eq_match = re.search(r'EQ_\w+:\s*Gain:([\d.-]+)dB,Frequency:([\d.]+)Hz', line)
        if eq_match:
            data['eq_adjustments'].append({
                'timestamp': timestamp,
                'gain': float(eq_match.group(1)),
                'frequency': float(eq_match.group(2))
            })

        # Pitch shift - 0.5734807x
        pitch_match = re.search(r'PitchShift_\w+:\s*([\d.]+)x', line)
        if pitch_match:
            data['pitch_shifts'].append((timestamp, float(pitch_match.group(1))))

        # Delay
        delay_match = re.search(r'Delay_DelaySlider_\w+:\s*([\d.]+)%', line)
        if delay_match:
            data['delay_values'].append(float(delay_match.group(1)))

        # Saturation
        sat_match = re.search(r'Saturation_SaturationSlider_\w+:\s*([\d.]+)%', line)
        if sat_match:
            data['saturation_values'].append(float(sat_match.group(1)))

        if timestamp:
            data['timestamps'].append(timestamp)

    return data


def compute_advanced_features(raw_data):
    """Compute advanced behavioral metrics."""
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

    duration = max(features['session_duration'], 0.1)

    # Volume metrics
    if raw_data['volume_history']:
        volumes = [v[1] for v in raw_data['volume_history']]
        features['avg_volume'] = np.mean(volumes)
        features['volume_std'] = np.std(volumes)
        features['volume_range'] = max(volumes) - min(volumes)
        features['final_volume'] = volumes[-1]
        features['initial_volume'] = volumes[0]
        features['volume_change'] = features['final_volume'] - features['initial_volume']

        # Volume trajectory (increasing/decreasing over time)
        if len(volumes) > 10:
            first_half = np.mean(volumes[:len(volumes)//2])
            second_half = np.mean(volumes[len(volumes)//2:])
            features['volume_trajectory'] = second_half - first_half
        else:
            features['volume_trajectory'] = 0

        # Adjustment magnitude (how big are the changes?)
        if len(volumes) > 1:
            diffs = np.abs(np.diff(volumes))
            features['avg_adjustment_magnitude'] = np.mean(diffs)
            features['max_adjustment_magnitude'] = np.max(diffs)
        else:
            features['avg_adjustment_magnitude'] = 0
            features['max_adjustment_magnitude'] = 0

    else:
        features['avg_volume'] = 50.0
        features['volume_std'] = 0.0
        features['volume_range'] = 0.0
        features['final_volume'] = 50.0
        features['initial_volume'] = 50.0
        features['volume_change'] = 0
        features['volume_trajectory'] = 0
        features['avg_adjustment_magnitude'] = 0
        features['max_adjustment_magnitude'] = 0

    # Muting behavior
    features['mute_events'] = raw_data['mute_events']
    features['mutes_per_minute'] = raw_data['mute_events'] / duration

    # EQ preferences
    if raw_data['eq_adjustments']:
        gains = [eq['gain'] for eq in raw_data['eq_adjustments']]
        freqs = [eq['frequency'] for eq in raw_data['eq_adjustments']]

        features['avg_eq_gain'] = np.mean(gains)
        features['max_eq_gain'] = np.max(gains)
        features['eq_adjustment_count'] = len(gains)

        # Frequency preference analysis
        low_freq_count = sum(1 for f in freqs if f < 500)
        mid_freq_count = sum(1 for f in freqs if 500 <= f < 2000)
        high_freq_count = sum(1 for f in freqs if f >= 2000)

        total_eq = len(freqs)
        features['low_freq_focus'] = low_freq_count / total_eq if total_eq > 0 else 0.33
        features['mid_freq_focus'] = mid_freq_count / total_eq if total_eq > 0 else 0.33
        features['high_freq_focus'] = high_freq_count / total_eq if total_eq > 0 else 0.33

        # Dominant frequency range
        features['avg_eq_frequency'] = np.mean(freqs)
    else:
        features['avg_eq_gain'] = 0
        features['max_eq_gain'] = 0
        features['eq_adjustment_count'] = 0
        features['low_freq_focus'] = 0.33
        features['mid_freq_focus'] = 0.33
        features['high_freq_focus'] = 0.33
        features['avg_eq_frequency'] = 1000

    # Pitch shift behavior
    if raw_data['pitch_shifts']:
        pitches = [p[1] for p in raw_data['pitch_shifts']]
        features['avg_pitch_shift'] = np.mean(pitches)
        features['pitch_shift_range'] = max(pitches) - min(pitches)
        features['final_pitch_shift'] = pitches[-1]

        # Deviation from natural (1.0x)
        features['pitch_deviation'] = abs(1.0 - features['avg_pitch_shift'])
    else:
        features['avg_pitch_shift'] = 1.0
        features['pitch_shift_range'] = 0
        features['final_pitch_shift'] = 1.0
        features['pitch_deviation'] = 0

    # Delay and saturation
    features['avg_delay'] = np.mean(raw_data['delay_values']) if raw_data['delay_values'] else 50.0
    features['avg_saturation'] = np.mean(raw_data['saturation_values']) if raw_data['saturation_values'] else 50.0

    # Total interaction intensity
    total_adjustments = len(raw_data['volume_history']) + len(raw_data['eq_adjustments']) + len(raw_data['pitch_shifts'])
    features['total_adjustments'] = total_adjustments
    features['adjustments_per_minute'] = total_adjustments / duration

    return features


def create_sensory_profile(features_dict):
    """
    Create detailed sensory profile based on all features.
    Returns classification and detailed characteristics.
    """
    profile = {}

    # Volume sensitivity score (-1 to +1)
    # Negative = hypersensitive (low volume), Positive = hyposensitive (high volume)
    vol_score = (features_dict['avg_volume'] - 55) / 25  # Normalized around 55%
    profile['volume_sensitivity'] = np.clip(vol_score, -1, 1)

    # Adjustment anxiety score (higher = more anxious/sensitive)
    adj_rate = features_dict['adjustments_per_minute']
    vol_std = features_dict['volume_std']
    profile['adjustment_anxiety'] = min((adj_rate / 50) + (vol_std / 30), 2)

    # Effect seeking score (higher = seeks more stimulation)
    effect_score = (
        (features_dict['avg_delay'] - 30) / 40 +
        (features_dict['avg_saturation'] - 40) / 30 +
        (features_dict['avg_eq_gain'] / 3)
    ) / 3
    profile['effect_seeking'] = np.clip(effect_score, -1, 1)

    # Frequency sensitivity (high freq focus = sensitive to treble)
    freq_score = features_dict['high_freq_focus'] - features_dict['low_freq_focus']
    profile['high_frequency_sensitivity'] = freq_score

    # Pitch modification preference
    profile['pitch_modification'] = features_dict['pitch_deviation']

    # Muting behavior (higher = more muting = potentially more sensitive)
    profile['muting_tendency'] = min(features_dict['mutes_per_minute'] / 0.5, 2)

    # Overall classification
    hyper_score = 0
    hypo_score = 0
    typical_score = 0

    # Volume
    if profile['volume_sensitivity'] < -0.3:
        hyper_score += 3
    elif profile['volume_sensitivity'] > 0.3:
        hypo_score += 3
    else:
        typical_score += 2

    # Adjustment anxiety
    if profile['adjustment_anxiety'] > 1.2:
        hyper_score += 2
    elif profile['adjustment_anxiety'] < 0.5:
        hypo_score += 1
    else:
        typical_score += 1

    # Effect seeking
    if profile['effect_seeking'] > 0.3:
        hypo_score += 2
    elif profile['effect_seeking'] < -0.3:
        hyper_score += 2
    else:
        typical_score += 1

    # High frequency sensitivity
    if profile['high_frequency_sensitivity'] > 0.2:
        hypo_score += 1  # Likes treble = seeking stimulation
    elif profile['high_frequency_sensitivity'] < -0.2:
        hyper_score += 1  # Avoids treble = sensitive

    # Muting
    if profile['muting_tendency'] > 1.0:
        hyper_score += 2
    elif profile['muting_tendency'] < 0.3:
        typical_score += 1

    scores = {
        'Hypersensitive': hyper_score,
        'Hyposensitive': hypo_score,
        'Typical': typical_score
    }

    profile['classification'] = max(scores, key=scores.get)
    profile['classification_scores'] = scores
    profile['confidence'] = max(scores.values()) / sum(scores.values()) * 100

    return profile


def main():
    test_data_dir = Path("/tmp/Neuroverse/Test Data")

    print("="*80)
    print("ENHANCED NEUROVERSE ANALYSIS")
    print("Including EQ, Pitch Shift, and Temporal Dynamics")
    print("="*80)

    participants = []

    for filepath in sorted(test_data_dir.glob("*.txt")):
        if 'PARTIALLY' in filepath.stem or 'POSTEND' in filepath.stem:
            continue

        participant_id = filepath.stem.upper()
        raw_data = parse_enhanced_features(filepath)
        features = compute_advanced_features(raw_data)

        if features['total_adjustments'] > 5:  # Minimum interaction threshold
            participants.append({
                'id': participant_id,
                'features': features
            })
            print(f"Parsed: {participant_id}")
            print(f"  Vol: {features['avg_volume']:.1f}%, EQ adj: {features['eq_adjustment_count']}, Pitch: {features['avg_pitch_shift']:.2f}x")

    print(f"\nTotal valid participants: {len(participants)}")

    # Create sensory profiles
    print("\n" + "="*80)
    print("INDIVIDUAL SENSORY PROFILES")
    print("="*80)

    results = []
    for p in participants:
        profile = create_sensory_profile(p['features'])

        results.append({
            'id': p['id'],
            'features': p['features'],
            'profile': profile
        })

        print(f"\n{p['id']}")
        print(f"  Classification: {profile['classification']} ({profile['confidence']:.1f}% confidence)")
        print(f"  Scores: {profile['classification_scores']}")
        print(f"  Sensory Profile:")
        print(f"    - Volume Sensitivity: {profile['volume_sensitivity']:+.2f} (- = hyper, + = hypo)")
        print(f"    - Adjustment Anxiety: {profile['adjustment_anxiety']:.2f} (higher = more anxious)")
        print(f"    - Effect Seeking: {profile['effect_seeking']:+.2f} (higher = seeks stimulation)")
        print(f"    - High Freq Sensitivity: {profile['high_frequency_sensitivity']:+.2f}")
        print(f"    - Muting Tendency: {profile['muting_tendency']:.2f}")

    # Summary
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY")
    print("="*80)

    from collections import Counter
    counts = Counter([r['profile']['classification'] for r in results])

    for cls in ['Hypersensitive', 'Typical', 'Hyposensitive']:
        count = counts.get(cls, 0)
        pct = (count / len(results)) * 100
        print(f"\n{cls}: {count} ({pct:.1f}%)")

        group = [r for r in results if r['profile']['classification'] == cls]
        if group:
            avg_vol = np.mean([r['features']['avg_volume'] for r in group])
            avg_adj = np.mean([r['features']['adjustments_per_minute'] for r in group])
            avg_eq_gain = np.mean([r['features']['avg_eq_gain'] for r in group])
            avg_pitch = np.mean([r['features']['avg_pitch_shift'] for r in group])

            print(f"  Avg Volume: {avg_vol:.1f}%")
            print(f"  Avg Adj/min: {avg_adj:.1f}")
            print(f"  Avg EQ Gain: {avg_eq_gain:.2f} dB")
            print(f"  Avg Pitch Shift: {avg_pitch:.2f}x")

    # Additional insights
    print("\n" + "="*80)
    print("ADDITIONAL INSIGHTS")
    print("="*80)

    # EQ frequency preferences
    print("\nEQ Frequency Focus by Classification:")
    for cls in ['Hypersensitive', 'Typical', 'Hyposensitive']:
        group = [r for r in results if r['profile']['classification'] == cls]
        if group:
            low = np.mean([r['features']['low_freq_focus'] for r in group])
            mid = np.mean([r['features']['mid_freq_focus'] for r in group])
            high = np.mean([r['features']['high_freq_focus'] for r in group])
            print(f"  {cls}: Low={low:.2f}, Mid={mid:.2f}, High={high:.2f}")

    # Volume trajectory
    print("\nVolume Trajectory (change over session):")
    for cls in ['Hypersensitive', 'Typical', 'Hyposensitive']:
        group = [r for r in results if r['profile']['classification'] == cls]
        if group:
            traj = np.mean([r['features']['volume_trajectory'] for r in group])
            print(f"  {cls}: {traj:+.1f}% (+ = increased volume over time)")

    # Save results
    output = {
        'method': 'Enhanced Multi-Feature Analysis',
        'n_participants': len(results),
        'classification_counts': dict(counts),
        'individual_results': [
            {
                'id': r['id'],
                'classification': r['profile']['classification'],
                'confidence': r['profile']['confidence'],
                'sensory_profile': {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in r['profile'].items()
                },
                'features': {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in r['features'].items()
                }
            }
            for r in results
        ]
    }

    with open('/tmp/Neuroverse/enhanced_analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /tmp/Neuroverse/enhanced_analysis_results.json")


if __name__ == "__main__":
    main()
