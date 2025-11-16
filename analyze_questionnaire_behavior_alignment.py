#!/usr/bin/env python3
"""
Cross-Validation Analysis: Questionnaire vs Actual Behavior
Compare what participants SAID they prefer with what they ACTUALLY did.
Identify alignment/mismatches to improve classification confidence.
"""

import re
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def parse_all_data(filepath):
    """Parse both questionnaire and behavioral data."""
    data = {
        'questionnaire': {},
        'volume_history': [],
        'mute_events': 0,
        'timestamps': [],
        'delay_values': [],
        'saturation_values': [],
        'pitch_values': [],
        'reverb_selected': None,  # Option 1/2/3
    }

    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()

    for line in lines:
        # Timestamps
        ts_match = re.search(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', line)
        if ts_match:
            data['timestamps'].append(ts_match.group(1))

        # Questionnaire responses
        for q_type in ['SoundEnvs', 'SoundTexture', 'Saturation', 'Delay', 'Reverb', 'PitchShift']:
            match = re.search(rf'{q_type}:\s*Option_(\d+)', line)
            if match:
                key = q_type.lower().replace('soundenvs', 'sound_env').replace('soundtexture', 'sound_texture').replace('pitchshift', 'pitch_shift')
                data['questionnaire'][key] = int(match.group(1))

        # Volume behavior
        vol_match = re.search(r'Volume_VolSlider_\w+:\s*([\d.]+)%', line)
        if vol_match:
            data['volume_history'].append(float(vol_match.group(1)))

        # Mutes
        if 'MuteAllAudio: True' in line or ': Muted' in line:
            data['mute_events'] += 1

        # Delay behavior
        delay_match = re.search(r'Delay_DelaySlider_\w+:\s*([\d.]+)%', line)
        if delay_match:
            data['delay_values'].append(float(delay_match.group(1)))

        # Saturation behavior
        sat_match = re.search(r'Saturation_SaturationSlider_\w+:\s*([\d.]+)%', line)
        if sat_match:
            data['saturation_values'].append(float(sat_match.group(1)))

        # Pitch behavior
        pitch_match = re.search(r'PitchShift_PitchSlider_?\w+:\s*([\d.]+)x', line)
        if pitch_match:
            data['pitch_values'].append(float(pitch_match.group(1)))

    # Compute behavioral summaries
    if data['volume_history']:
        data['final_volume'] = data['volume_history'][-1]
        data['avg_volume'] = np.mean(data['volume_history'])
        # Last 20% for stability
        n = max(1, len(data['volume_history']) // 5)
        data['settled_volume'] = np.mean(data['volume_history'][-n:])
    else:
        data['final_volume'] = 50.0
        data['avg_volume'] = 50.0
        data['settled_volume'] = 50.0

    if data['delay_values']:
        data['final_delay'] = data['delay_values'][-1]
        data['avg_delay'] = np.mean(data['delay_values'])
    else:
        data['final_delay'] = 20.0
        data['avg_delay'] = 20.0

    if data['saturation_values']:
        data['final_saturation'] = data['saturation_values'][-1]
        data['avg_saturation'] = np.mean(data['saturation_values'])
    else:
        data['final_saturation'] = 50.0
        data['avg_saturation'] = 50.0

    if data['pitch_values']:
        data['final_pitch'] = data['pitch_values'][-1]
        data['avg_pitch'] = np.mean(data['pitch_values'])
    else:
        data['final_pitch'] = 1.0
        data['avg_pitch'] = 1.0

    # Session duration
    if len(data['timestamps']) >= 2:
        try:
            first = datetime.strptime(data['timestamps'][0], '%Y/%m/%d %H:%M:%S.%f')
            last = datetime.strptime(data['timestamps'][-1], '%Y/%m/%d %H:%M:%S.%f')
            data['session_duration'] = (last - first).total_seconds() / 60.0
        except:
            data['session_duration'] = 10.0
    else:
        data['session_duration'] = 10.0

    data['mutes_per_minute'] = data['mute_events'] / max(data['session_duration'], 0.1)

    return data


def interpret_questionnaire_option(question, option):
    """
    Interpret what each questionnaire option likely means.
    Option 1 = Conservative/Sensitive
    Option 2 = Balanced/Neutral
    Option 3 = Stimulating/Seeking
    """
    interpretations = {
        'sound_env': {
            1: ('Natural/Quiet', 'hyper'),
            2: ('Balanced', 'typical'),
            3: ('Stimulating/Rich', 'hypo')
        },
        'sound_texture': {
            1: ('Warm/Soft', 'hyper'),
            2: ('Balanced', 'typical'),
            3: ('Bright/Sharp', 'hypo')
        },
        'saturation': {
            1: ('Minimal/Clean', 'hyper'),
            2: ('Moderate', 'typical'),
            3: ('Rich/Full', 'hypo')
        },
        'delay': {
            1: ('None/Minimal', 'hyper'),
            2: ('Moderate', 'typical'),
            3: ('Extended', 'hypo')
        },
        'reverb': {
            1: ('Dry', 'hyper'),
            2: ('Moderate', 'typical'),
            3: ('Wet/Spacious', 'hypo')
        },
        'pitch_shift': {
            1: ('Natural/Unchanged', 'hyper'),
            2: ('Slight modification', 'typical'),
            3: ('Processed/Altered', 'hypo')
        }
    }

    if question in interpretations and option in interpretations[question]:
        return interpretations[question][option]
    return ('Unknown', 'typical')


def classify_behavior(data, cohort_stats):
    """
    Classify based purely on behavior within cohort context.
    Returns behavioral profile scores.
    """
    scores = {'hyper': 0, 'typical': 0, 'hypo': 0}

    # Volume (most important)
    vol_z = (data['settled_volume'] - cohort_stats['volume_mean']) / max(cohort_stats['volume_std'], 1)
    if vol_z < -0.5:
        scores['hyper'] += 3
    elif vol_z > 0.5:
        scores['hypo'] += 3
    else:
        scores['typical'] += 2

    # Muting (higher = more sensitive)
    mute_z = (data['mutes_per_minute'] - cohort_stats['mute_mean']) / max(cohort_stats['mute_std'], 0.1)
    if mute_z > 0.5:
        scores['hyper'] += 2
    elif mute_z < -0.3:
        scores['typical'] += 1
    else:
        scores['typical'] += 1

    # Delay preference
    delay_z = (data['final_delay'] - cohort_stats['delay_mean']) / max(cohort_stats['delay_std'], 1)
    if delay_z > 0.5:
        scores['hypo'] += 1
    elif delay_z < -0.5:
        scores['hyper'] += 1
    else:
        scores['typical'] += 1

    # Saturation preference
    sat_z = (data['final_saturation'] - cohort_stats['sat_mean']) / max(cohort_stats['sat_std'], 1)
    if sat_z > 0.5:
        scores['hypo'] += 1
    elif sat_z < -0.5:
        scores['hyper'] += 1
    else:
        scores['typical'] += 1

    return scores


def main():
    test_data_dir = Path("/tmp/Neuroverse/Test Data")

    print("="*80)
    print("QUESTIONNAIRE vs BEHAVIOR ALIGNMENT ANALYSIS")
    print("Cross-validating self-report with actual actions")
    print("="*80)

    # Parse all participants
    participants = []
    for filepath in sorted(test_data_dir.glob("*.txt")):
        if 'PARTIALLY' in filepath.stem or 'POSTEND' in filepath.stem:
            continue

        participant_id = filepath.stem.upper()
        data = parse_all_data(filepath)

        if data['questionnaire'] and len(data['volume_history']) > 0:
            participants.append({
                'id': participant_id,
                'data': data
            })
            print(f"Parsed: {participant_id} (Questionnaire: {len(data['questionnaire'])} responses)")

    print(f"\nTotal valid participants: {len(participants)}")

    # Calculate cohort statistics for behavioral normalization
    cohort_stats = {
        'volume_mean': np.mean([p['data']['settled_volume'] for p in participants]),
        'volume_std': np.std([p['data']['settled_volume'] for p in participants]),
        'mute_mean': np.mean([p['data']['mutes_per_minute'] for p in participants]),
        'mute_std': np.std([p['data']['mutes_per_minute'] for p in participants]),
        'delay_mean': np.mean([p['data']['final_delay'] for p in participants]),
        'delay_std': np.std([p['data']['final_delay'] for p in participants]),
        'sat_mean': np.mean([p['data']['final_saturation'] for p in participants]),
        'sat_std': np.std([p['data']['final_saturation'] for p in participants]),
    }

    print(f"\nCohort Baseline:")
    print(f"  Volume: {cohort_stats['volume_mean']:.1f}% (±{cohort_stats['volume_std']:.1f})")
    print(f"  Delay: {cohort_stats['delay_mean']:.1f}% (±{cohort_stats['delay_std']:.1f})")
    print(f"  Saturation: {cohort_stats['sat_mean']:.1f}% (±{cohort_stats['sat_std']:.1f})")
    print(f"  Mutes/min: {cohort_stats['mute_mean']:.2f} (±{cohort_stats['mute_std']:.2f})")

    # Analyze each participant
    print("\n" + "="*80)
    print("INDIVIDUAL ALIGNMENT ANALYSIS")
    print("="*80)

    results = []
    alignment_scores = []

    for p in participants:
        participant_id = p['id']
        data = p['data']

        print(f"\n{participant_id}")
        print("-" * 40)

        # Questionnaire profile
        questionnaire_scores = {'hyper': 0, 'typical': 0, 'hypo': 0}
        print("Questionnaire Responses:")
        for q, opt in data['questionnaire'].items():
            desc, tendency = interpret_questionnaire_option(q, opt)
            questionnaire_scores[tendency] += 1
            print(f"  {q}: Option {opt} -> {desc} ({tendency})")

        q_classification = max(questionnaire_scores, key=questionnaire_scores.get)
        print(f"  -> Self-Report Classification: {q_classification.upper()}")
        print(f"     Scores: {questionnaire_scores}")

        # Behavioral profile
        behavior_scores = classify_behavior(data, cohort_stats)
        b_classification = max(behavior_scores, key=behavior_scores.get)

        print(f"\nActual Behavior:")
        print(f"  Settled Volume: {data['settled_volume']:.1f}%")
        print(f"  Final Delay: {data['final_delay']:.1f}%")
        print(f"  Final Saturation: {data['final_saturation']:.1f}%")
        print(f"  Mutes/min: {data['mutes_per_minute']:.2f}")
        print(f"  -> Behavioral Classification: {b_classification.upper()}")
        print(f"     Scores: {behavior_scores}")

        # Alignment check
        if q_classification == b_classification:
            alignment = "ALIGNED ✓"
            alignment_score = 1.0
        elif (q_classification == 'hyper' and b_classification == 'hypo') or \
             (q_classification == 'hypo' and b_classification == 'hyper'):
            alignment = "MISMATCHED ✗"
            alignment_score = 0.0
        else:
            alignment = "PARTIALLY ALIGNED ~"
            alignment_score = 0.5

        print(f"\n  Alignment: {alignment}")
        alignment_scores.append(alignment_score)

        # Combined classification with confidence
        combined_scores = {
            'hyper': questionnaire_scores['hyper'] * 2 + behavior_scores['hyper'] * 3,
            'typical': questionnaire_scores['typical'] * 2 + behavior_scores['typical'] * 3,
            'hypo': questionnaire_scores['hypo'] * 2 + behavior_scores['hypo'] * 3
        }
        combined_class = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[combined_class] / sum(combined_scores.values()) * 100

        print(f"  Combined Classification: {combined_class.upper()} ({confidence:.1f}% confidence)")

        results.append({
            'id': participant_id,
            'questionnaire_class': q_classification,
            'behavioral_class': b_classification,
            'combined_class': combined_class,
            'alignment': alignment,
            'confidence': confidence,
            'questionnaire_scores': questionnaire_scores,
            'behavior_scores': behavior_scores,
            'combined_scores': combined_scores,
            'settled_volume': data['settled_volume'],
            'final_delay': data['final_delay'],
            'final_saturation': data['final_saturation'],
            'mutes_per_minute': data['mutes_per_minute']
        })

    # Summary
    print("\n" + "="*80)
    print("ALIGNMENT SUMMARY")
    print("="*80)

    aligned_count = sum(1 for s in alignment_scores if s == 1.0)
    partial_count = sum(1 for s in alignment_scores if s == 0.5)
    mismatch_count = sum(1 for s in alignment_scores if s == 0.0)

    print(f"Total Participants: {len(results)}")
    print(f"  Fully Aligned: {aligned_count} ({aligned_count/len(results)*100:.1f}%)")
    print(f"  Partially Aligned: {partial_count} ({partial_count/len(results)*100:.1f}%)")
    print(f"  Mismatched: {mismatch_count} ({mismatch_count/len(results)*100:.1f}%)")
    print(f"\nOverall Alignment Score: {np.mean(alignment_scores)*100:.1f}%")

    # Final combined classification
    print("\n" + "="*80)
    print("FINAL COMBINED CLASSIFICATION")
    print("="*80)

    from collections import Counter
    combined_counts = Counter([r['combined_class'] for r in results])

    for cls in ['hyper', 'typical', 'hypo']:
        count = combined_counts.get(cls, 0)
        pct = (count / len(results)) * 100
        label = {'hyper': 'Hypersensitive', 'typical': 'Typical', 'hypo': 'Hyposensitive'}[cls]
        print(f"\n{label}: {count} ({pct:.1f}%)")

        group = [r for r in results if r['combined_class'] == cls]
        if group:
            vol = np.mean([r['settled_volume'] for r in group])
            delay = np.mean([r['final_delay'] for r in group])
            sat = np.mean([r['final_saturation'] for r in group])
            conf = np.mean([r['confidence'] for r in group])
            print(f"  Avg Settled Volume: {vol:.1f}%")
            print(f"  Avg Final Delay: {delay:.1f}%")
            print(f"  Avg Final Saturation: {sat:.1f}%")
            print(f"  Avg Confidence: {conf:.1f}%")

    # Save results
    output = {
        'method': 'Questionnaire-Behavior Cross-Validation',
        'cohort_stats': cohort_stats,
        'overall_alignment': np.mean(alignment_scores) * 100,
        'classification_counts': {
            'Hypersensitive': combined_counts.get('hyper', 0),
            'Typical': combined_counts.get('typical', 0),
            'Hyposensitive': combined_counts.get('hypo', 0)
        },
        'individual_results': results
    }

    with open('/tmp/Neuroverse/questionnaire_behavior_alignment.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nResults saved to /tmp/Neuroverse/questionnaire_behavior_alignment.json")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. ALIGNMENT indicates how well self-report matches behavior
   - High alignment = consistent participant (more reliable classification)
   - Low alignment = possible misunderstanding or compensation behavior

2. COMBINED CLASSIFICATION weights behavior MORE than questionnaire
   - Behavior is objective (what they did)
   - Questionnaire is subjective (what they think they prefer)

3. CONFIDENCE increases when:
   - Questionnaire and behavior agree
   - Scores are strongly skewed to one category
   - Pattern is consistent across multiple indicators
""")


if __name__ == "__main__":
    main()
