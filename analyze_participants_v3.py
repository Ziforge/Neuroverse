#!/usr/bin/env python3
"""
Neuroverse Participant Analysis v3
Improved classification using:
- Normalized metrics (per-minute rates)
- Cohort-relative comparisons instead of static thresholds
- Z-scores for each behavioral metric
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

def parse_participant_file(filepath):
    """Parse Unity debug log file for participant data."""
    data = {
        'volume_history': [],
        'mute_events': 0,
        'eq_settings': [],
        'reverb_settings': [],
        'delay_settings': [],
        'saturation_settings': [],
        'pitch_shift_settings': [],
        'questionnaire': {},
        'timestamps': [],
        'total_adjustments': 0
    }

    with open(filepath, 'r', errors='ignore') as f:
        content = f.read()

    lines = content.split('\n')

    for line in lines:
        # Extract timestamps
        timestamp_match = re.search(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', line)
        if timestamp_match:
            data['timestamps'].append(timestamp_match.group(1))

        # Track volume changes - format: Volume_VolSlider_Ocean: 66.89528%
        volume_match = re.search(r'Volume_VolSlider_\w+:\s*([\d.]+)%', line)
        if volume_match:
            data['volume_history'].append(float(volume_match.group(1)))
            data['total_adjustments'] += 1

        # Track mute events - count each Muted or True event
        if 'Mute' in line:
            # Count each mute action (MuteAllAudio: True or Mute_MuteToggle_X: Muted)
            if 'MuteAllAudio: True' in line or ': Muted' in line:
                data['mute_events'] += 1

        # Track questionnaire responses
        for option_type in ['SoundEnvs', 'SoundTexture', 'Saturation', 'Delay', 'Reverb', 'PitchShift']:
            pattern = rf'{option_type}:\s*Option_(\d+)'
            match = re.search(pattern, line)
            if match:
                key = option_type.lower().replace('soundenvs', 'sound_env').replace('soundtexture', 'sound_texture').replace('pitchshift', 'pitch_shift')
                data['questionnaire'][key] = int(match.group(1))

        # Track EQ adjustments
        for band in ['Bass', 'Mid', 'Treble']:
            pattern = rf'{band}\s*EQ:\s*([\d.]+)'
            match = re.search(pattern, line)
            if match:
                data[f'{band.lower()}_eq'] = float(match.group(1))
                data['total_adjustments'] += 1

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

    # Calculate behavioral metrics
    if data['volume_history']:
        data['avg_volume'] = np.mean(data['volume_history'])
        data['volume_variance'] = np.var(data['volume_history'])
        data['volume_range'] = max(data['volume_history']) - min(data['volume_history'])
    else:
        data['avg_volume'] = 50.0
        data['volume_variance'] = 0.0
        data['volume_range'] = 0.0

    # Normalized metrics (per minute)
    duration = max(data['session_duration_minutes'], 0.1)
    data['mutes_per_minute'] = data['mute_events'] / duration
    data['adjustments_per_minute'] = data['total_adjustments'] / duration

    return data


def calculate_cohort_statistics(all_participants):
    """Calculate cohort-wide statistics for relative comparisons."""
    cohort_stats = {}

    metrics = [
        'avg_volume', 'volume_variance', 'volume_range',
        'mutes_per_minute', 'adjustments_per_minute',
        'mute_events', 'total_adjustments', 'session_duration_minutes'
    ]

    for metric in metrics:
        values = [p['data'][metric] for p in all_participants if metric in p['data']]
        if values:
            cohort_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }

    # Calculate questionnaire response distributions
    questionnaire_stats = defaultdict(lambda: defaultdict(int))
    for p in all_participants:
        for key, value in p['data']['questionnaire'].items():
            questionnaire_stats[key][value] += 1

    cohort_stats['questionnaire_distributions'] = dict(questionnaire_stats)

    return cohort_stats


def classify_participant_relative(participant_data, cohort_stats):
    """
    Classify participant using cohort-relative metrics.
    Uses z-scores to compare each participant to the group average.
    """
    scores = {
        'hypersensitive': 0,
        'hyposensitive': 0,
        'typical': 0
    }

    evidence = []

    # 1. Volume relative to cohort (z-score based)
    if 'avg_volume' in cohort_stats:
        vol_z = (participant_data['avg_volume'] - cohort_stats['avg_volume']['mean']) / max(cohort_stats['avg_volume']['std'], 0.1)

        if vol_z < -0.5:  # Below average volume
            scores['hypersensitive'] += 2.5
            evidence.append(f"Volume below cohort avg (z={vol_z:.2f})")
        elif vol_z > 0.5:  # Above average volume
            scores['hyposensitive'] += 2.5
            evidence.append(f"Volume above cohort avg (z={vol_z:.2f})")
        else:
            scores['typical'] += 2.0
            evidence.append(f"Volume near cohort avg (z={vol_z:.2f})")

    # 2. Volume variance (stability)
    if 'volume_variance' in cohort_stats:
        var_z = (participant_data['volume_variance'] - cohort_stats['volume_variance']['mean']) / max(cohort_stats['volume_variance']['std'], 0.1)

        if var_z > 0.5:  # High variance - more adjustment
            scores['hypersensitive'] += 1.5  # Sensitive to finding right level
            evidence.append(f"High volume variance (z={var_z:.2f})")
        elif var_z < -0.5:
            scores['hyposensitive'] += 1.0
            evidence.append(f"Stable volume preference (z={var_z:.2f})")
        else:
            scores['typical'] += 1.0

    # 3. Adjustment rate relative to cohort
    if 'adjustments_per_minute' in cohort_stats:
        adj_z = (participant_data['adjustments_per_minute'] - cohort_stats['adjustments_per_minute']['mean']) / max(cohort_stats['adjustments_per_minute']['std'], 0.1)

        if adj_z > 0.8:  # Very frequent adjustments
            scores['hypersensitive'] += 2.0
            evidence.append(f"High adjustment rate (z={adj_z:.2f})")
        elif adj_z < -0.8:  # Infrequent adjustments
            scores['hyposensitive'] += 1.5
            evidence.append(f"Low adjustment rate (z={adj_z:.2f})")
        else:
            scores['typical'] += 1.5

    # 4. Muting behavior relative to cohort
    if 'mutes_per_minute' in cohort_stats:
        mute_z = (participant_data['mutes_per_minute'] - cohort_stats['mutes_per_minute']['mean']) / max(cohort_stats['mutes_per_minute']['std'], 0.1)

        if mute_z > 0.5:  # More muting than average
            scores['hypersensitive'] += 1.5
            evidence.append(f"Above-average muting (z={mute_z:.2f})")
        elif mute_z < -0.5:
            scores['hyposensitive'] += 1.0
            evidence.append(f"Below-average muting (z={mute_z:.2f})")
        else:
            scores['typical'] += 1.0

    # 5. Questionnaire responses - strongest indicator (weight 3x)
    questionnaire = participant_data['questionnaire']

    # Sound environment preference
    if 'sound_env' in questionnaire:
        opt = questionnaire['sound_env']
        if opt == 1:
            scores['hypersensitive'] += 3.0
            evidence.append("Sound env: Natural/quiet (Option 1)")
        elif opt == 3:
            scores['hyposensitive'] += 3.0
            evidence.append("Sound env: Stimulating/rich (Option 3)")
        else:
            scores['typical'] += 2.5
            evidence.append("Sound env: Balanced (Option 2)")

    # Sound texture preference
    if 'sound_texture' in questionnaire:
        opt = questionnaire['sound_texture']
        if opt == 1:
            scores['hypersensitive'] += 3.0
            evidence.append("Texture: Warm/soft (Option 1)")
        elif opt == 3:
            scores['hyposensitive'] += 3.0
            evidence.append("Texture: Bright/sharp (Option 3)")
        else:
            scores['typical'] += 2.5
            evidence.append("Texture: Balanced (Option 2)")

    # Saturation preference
    if 'saturation' in questionnaire:
        opt = questionnaire['saturation']
        if opt == 1:
            scores['hypersensitive'] += 2.5
            evidence.append("Saturation: Minimal (Option 1)")
        elif opt == 3:
            scores['hyposensitive'] += 2.5
            evidence.append("Saturation: Rich/full (Option 3)")
        else:
            scores['typical'] += 2.0

    # Reverb preference
    if 'reverb' in questionnaire:
        opt = questionnaire['reverb']
        if opt == 1:
            scores['hypersensitive'] += 2.5
            evidence.append("Reverb: Dry (Option 1)")
        elif opt == 3:
            scores['hyposensitive'] += 2.5
            evidence.append("Reverb: Wet/spacious (Option 3)")
        else:
            scores['typical'] += 2.0

    # Delay preference
    if 'delay' in questionnaire:
        opt = questionnaire['delay']
        if opt == 1:
            scores['hypersensitive'] += 2.0
            evidence.append("Delay: None/minimal (Option 1)")
        elif opt == 3:
            scores['hyposensitive'] += 2.0
            evidence.append("Delay: Extended (Option 3)")
        else:
            scores['typical'] += 1.5

    # Pitch shift preference
    if 'pitch_shift' in questionnaire:
        opt = questionnaire['pitch_shift']
        if opt == 1:
            scores['hypersensitive'] += 2.0
            evidence.append("Pitch: Natural/unchanged (Option 1)")
        elif opt == 3:
            scores['hyposensitive'] += 2.0
            evidence.append("Pitch: Processed/altered (Option 3)")
        else:
            scores['typical'] += 1.5

    # Determine classification
    max_score = max(scores.values())
    classification = max(scores.items(), key=lambda x: x[1])[0]

    # Calculate confidence as percentage of total evidence
    total_score = sum(scores.values())
    confidence = (max_score / total_score) * 100 if total_score > 0 else 33.3

    # Certainty level based on score separation
    sorted_scores = sorted(scores.values(), reverse=True)
    separation = sorted_scores[0] - sorted_scores[1]

    if separation > 5:
        certainty = 'High'
    elif separation > 2:
        certainty = 'Medium'
    else:
        certainty = 'Low'

    return {
        'classification': classification.capitalize(),
        'scores': scores,
        'confidence': confidence,
        'certainty': certainty,
        'evidence': evidence,
        'max_score': max_score
    }


def generate_sound_wheel_mapping(classification_results):
    """Generate Sound Wheel attributes based on actual cohort data."""

    profiles = {
        'Hypersensitive': [],
        'Hyposensitive': [],
        'Typical': []
    }

    for result in classification_results:
        cls = result['classification']['classification']
        profiles[cls].append(result['data'])

    sound_wheel = {}

    for profile_name, participants in profiles.items():
        if not participants:
            continue

        # Calculate actual averages for this profile group
        avg_vol = np.mean([p['avg_volume'] for p in participants])
        avg_mutes = np.mean([p['mutes_per_minute'] for p in participants])
        avg_adj = np.mean([p['adjustments_per_minute'] for p in participants])
        avg_var = np.mean([p['volume_variance'] for p in participants])

        # Map to Sound Wheel based on actual group characteristics
        sound_wheel[profile_name] = {
            'Loudness': round(avg_vol, 1),
            'Attack': 'Imprecise' if avg_var > 300 else ('Precise' if avg_var < 100 else 'Moderate'),
            'Punch': 'Weak' if avg_vol < 50 else ('Strong' if avg_vol > 65 else 'Moderate'),
            'Treble_Strength': round((avg_vol / 10), 1),  # Scale 0-10
            'Bass_Depth': round((avg_vol / 12), 1),
            'Timbral_Balance': 'Dark/Warm' if avg_vol < 50 else ('Bright' if avg_vol > 65 else 'Neutral'),
            'Depth': 'Shallow' if avg_vol < 50 else ('Deep' if avg_vol > 65 else 'Moderate'),
            'Width': 'Small' if avg_vol < 50 else ('Large' if avg_vol > 65 else 'Moderate'),
            'Reverberance': 'Dry' if avg_adj > 35 else ('Wet' if avg_adj < 20 else 'Moderate'),
            'Presence': round(avg_vol / 10, 1),
            'Detail': 'Simple' if avg_vol < 50 else ('Rich' if avg_vol > 65 else 'Moderate'),
            'sample_size': len(participants),
            'avg_volume': avg_vol,
            'avg_mutes_per_min': avg_mutes,
            'avg_adjustments_per_min': avg_adj
        }

    return sound_wheel


def main():
    test_data_dir = Path("/tmp/Neuroverse/Test Data")

    # Parse all participants first
    print("="*80)
    print("NEUROVERSE PARTICIPANT ANALYSIS v3 - COHORT-RELATIVE CLASSIFICATION")
    print("="*80)

    all_participants = []

    for filepath in sorted(test_data_dir.glob("*.txt")):
        participant_id = filepath.stem.upper()
        data = parse_participant_file(filepath)
        all_participants.append({
            'id': participant_id,
            'data': data
        })
        print(f"Parsed: {participant_id}")

    print(f"\nTotal participants: {len(all_participants)}")

    # Calculate cohort statistics
    print("\n" + "-"*80)
    print("COHORT STATISTICS (Reference for Classification)")
    print("-"*80)

    cohort_stats = calculate_cohort_statistics(all_participants)

    for metric, stats_dict in cohort_stats.items():
        if metric != 'questionnaire_distributions' and isinstance(stats_dict, dict):
            print(f"\n{metric}:")
            print(f"  Mean: {stats_dict['mean']:.2f}")
            print(f"  Std:  {stats_dict['std']:.2f}")
            print(f"  Median: {stats_dict['median']:.2f}")
            print(f"  IQR: [{stats_dict['q25']:.2f} - {stats_dict['q75']:.2f}]")

    # Classify each participant relative to cohort
    print("\n" + "-"*80)
    print("INDIVIDUAL CLASSIFICATIONS (Relative to Cohort)")
    print("-"*80)

    classification_results = []

    for participant in all_participants:
        classification = classify_participant_relative(participant['data'], cohort_stats)

        classification_results.append({
            'id': participant['id'],
            'data': participant['data'],
            'classification': classification
        })

        print(f"\n{participant['id']}")
        print(f"  Classification: {classification['classification']} (Confidence: {classification['confidence']:.1f}%, Certainty: {classification['certainty']})")
        print(f"  Scores: Hyper={classification['scores']['hypersensitive']:.1f}, Hypo={classification['scores']['hyposensitive']:.1f}, Typical={classification['scores']['typical']:.1f}")
        print(f"  Behavioral: Vol={participant['data']['avg_volume']:.1f}%, Adj/min={participant['data']['adjustments_per_minute']:.1f}, Mutes/min={participant['data']['mutes_per_minute']:.2f}")
        print(f"  Key Evidence:")
        for evidence in classification['evidence'][:5]:  # Top 5 pieces of evidence
            print(f"    - {evidence}")

    # Summary statistics
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY (COHORT-RELATIVE)")
    print("="*80)

    counts = defaultdict(int)
    for result in classification_results:
        counts[result['classification']['classification']] += 1

    total = len(classification_results)
    for cls in ['Hypersensitive', 'Typical', 'Hyposensitive']:
        count = counts[cls]
        pct = (count / total) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")

    # Generate Sound Wheel mapping based on actual data
    print("\n" + "="*80)
    print("SOUND WHEEL MAPPING (Based on Actual Group Characteristics)")
    print("="*80)

    sound_wheel = generate_sound_wheel_mapping(classification_results)

    for profile_name, attributes in sound_wheel.items():
        print(f"\n{profile_name.upper()} (n={attributes['sample_size']}):")
        print(f"  Actual Group Averages:")
        print(f"    - Volume: {attributes['avg_volume']:.1f}%")
        print(f"    - Adjustments/min: {attributes['avg_adjustments_per_min']:.1f}")
        print(f"    - Mutes/min: {attributes['avg_mutes_per_min']:.2f}")
        print(f"  Sound Wheel Attributes:")
        for attr, value in attributes.items():
            if attr not in ['sample_size', 'avg_volume', 'avg_mutes_per_min', 'avg_adjustments_per_min']:
                print(f"    - {attr.replace('_', ' ')}: {value}")

    # Save results
    results_data = {
        'cohort_statistics': {k: v for k, v in cohort_stats.items() if k != 'questionnaire_distributions'},
        'classification_counts': dict(counts),
        'individual_results': [
            {
                'id': r['id'],
                'classification': r['classification']['classification'],
                'confidence': r['classification']['confidence'],
                'certainty': r['classification']['certainty'],
                'scores': r['classification']['scores'],
                'avg_volume': r['data']['avg_volume'],
                'adjustments_per_minute': r['data']['adjustments_per_minute'],
                'mutes_per_minute': r['data']['mutes_per_minute'],
                'session_duration': r['data']['session_duration_minutes']
            }
            for r in classification_results
        ],
        'sound_wheel_mapping': sound_wheel
    }

    with open('/tmp/Neuroverse/analysis_v3_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n\nResults saved to /tmp/Neuroverse/analysis_v3_results.json")

    # Compare with v2 results
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS IN v3:")
    print("="*80)
    print("1. Uses z-scores relative to cohort mean instead of static thresholds")
    print("2. Normalizes behavioral metrics per minute (accounts for session duration)")
    print("3. Sound Wheel attributes derived from actual group characteristics")
    print("4. Considers volume variance as indicator of sensory sensitivity")
    print("5. Provides detailed evidence chain for each classification")


if __name__ == "__main__":
    main()
