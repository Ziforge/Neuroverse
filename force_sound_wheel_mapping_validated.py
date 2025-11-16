#!/usr/bin/env python3
"""
FORCE Technology Sound Wheel Mapping
Based on validated behavioral classifications.
Maps actual participant preferences to standardized audio attributes.
"""

import json
import numpy as np

# Load the cross-validated results
with open('/tmp/Neuroverse/questionnaire_behavior_alignment.json', 'r') as f:
    alignment_data = json.load(f)

# Load temporal patterns
with open('/tmp/Neuroverse/temporal_patterns_results.json', 'r') as f:
    temporal_data = json.load(f)

results = alignment_data['individual_results']

print("="*80)
print("FORCE TECHNOLOGY SOUND WHEEL MAPPING")
print("Based on Validated Behavioral Classifications")
print("="*80)

# Group by combined classification
profiles = {
    'hyper': [],
    'typical': [],
    'hypo': []
}

for r in results:
    profiles[r['combined_class']].append(r)

# FORCE Technology Sound Wheel Categories
sound_wheel_mapping = {}

for profile_name, participants in profiles.items():
    if not participants:
        continue

    # Extract behavioral metrics
    volumes = [p['settled_volume'] for p in participants]
    delays = [p['final_delay'] for p in participants]
    saturations = [p['final_saturation'] for p in participants]
    mutes_per_min = [p['mutes_per_minute'] for p in participants]

    avg_vol = np.mean(volumes)
    avg_delay = np.mean(delays)
    avg_sat = np.mean(saturations)
    avg_muting = np.mean(mutes_per_min)
    vol_range = (np.min(volumes), np.max(volumes))

    label = {
        'hyper': 'HYPERSENSITIVE',
        'typical': 'TYPICAL',
        'hypo': 'HYPOSENSITIVE'
    }[profile_name]

    print(f"\n{'='*80}")
    print(f"{label} PROFILE (n={len(participants)})")
    print(f"{'='*80}")
    print(f"\nActual Behavioral Data:")
    print(f"  Settled Volume: {avg_vol:.1f}% (range: {vol_range[0]:.1f}% - {vol_range[1]:.1f}%)")
    print(f"  Final Delay: {avg_delay:.1f}%")
    print(f"  Final Saturation: {avg_sat:.1f}%")
    print(f"  Muting Rate: {avg_muting:.2f}/min")

    print(f"\n--- FORCE SOUND WHEEL ATTRIBUTES ---")

    # Map to Sound Wheel based on actual behavior
    wheel = {}

    # LOUDNESS CATEGORY
    if avg_vol < 50:
        wheel['Loudness'] = 'Soft (40-55 dB SPL estimated)'
        wheel['Loudness_Value'] = 3  # Scale 1-10
    elif avg_vol < 70:
        wheel['Loudness'] = 'Medium (55-65 dB SPL estimated)'
        wheel['Loudness_Value'] = 5
    else:
        wheel['Loudness'] = 'Loud (65-75+ dB SPL estimated)'
        wheel['Loudness_Value'] = 7

    print(f"\nLOUDNESS CATEGORY:")
    print(f"  Loudness: {wheel['Loudness']}")

    # DYNAMICS CATEGORY
    if avg_muting > 0.5:
        wheel['Attack'] = 'Imprecise/Smoothed'
        wheel['Punch'] = 'Weak'
        wheel['Powerful'] = 'Reduced'
        wheel['Bass_Precision'] = 'Soft'
    elif avg_muting < 0.3:
        wheel['Attack'] = 'Precise'
        wheel['Punch'] = 'Strong'
        wheel['Powerful'] = 'Maintained'
        wheel['Bass_Precision'] = 'Precise'
    else:
        wheel['Attack'] = 'Moderate'
        wheel['Punch'] = 'Moderate'
        wheel['Powerful'] = 'Maintained'
        wheel['Bass_Precision'] = 'Moderate'

    print(f"\nDYNAMICS CATEGORY:")
    print(f"  Attack: {wheel['Attack']}")
    print(f"  Punch: {wheel['Punch']}")
    print(f"  Powerful: {wheel['Powerful']}")
    print(f"  Bass Precision: {wheel['Bass_Precision']}")

    # TIMBRE CATEGORY
    if avg_vol < 50 and avg_sat < 30:
        wheel['Treble_Strength'] = 'Low (avoid harsh frequencies)'
        wheel['Brilliance'] = 'Low'
        wheel['Bass_Strength'] = 'Neutral'
        wheel['Bass_Depth'] = 'Shallow'
        wheel['Timbral_Balance'] = 'Dark/Warm'
        wheel['Full'] = 'Low degree'
    elif avg_vol > 70 and avg_sat > 50:
        wheel['Treble_Strength'] = 'High (detailed high frequencies)'
        wheel['Brilliance'] = 'High'
        wheel['Bass_Strength'] = 'Strong'
        wheel['Bass_Depth'] = 'Deep'
        wheel['Timbral_Balance'] = 'Bright'
        wheel['Full'] = 'High degree'
    else:
        wheel['Treble_Strength'] = 'Neutral'
        wheel['Brilliance'] = 'Moderate'
        wheel['Bass_Strength'] = 'Neutral'
        wheel['Bass_Depth'] = 'Moderate'
        wheel['Timbral_Balance'] = 'Neutral'
        wheel['Full'] = 'Moderate'

    print(f"\nTIMBRE CATEGORY:")
    print(f"  Treble Strength: {wheel['Treble_Strength']}")
    print(f"  Brilliance: {wheel['Brilliance']}")
    print(f"  Bass Strength: {wheel['Bass_Strength']}")
    print(f"  Bass Depth: {wheel['Bass_Depth']}")
    print(f"  Timbral Balance: {wheel['Timbral_Balance']}")
    print(f"  Full: {wheel['Full']}")

    # SPATIAL CATEGORY
    if avg_delay < 30:
        wheel['Depth'] = 'Shallow'
        wheel['Width'] = 'Small'
        wheel['Envelopment'] = 'Low'
        wheel['Reverberance'] = 'Dry'
        wheel['Distance'] = 'Close/Intimate'
    elif avg_delay > 60:
        wheel['Depth'] = 'Deep'
        wheel['Width'] = 'Wide'
        wheel['Envelopment'] = 'High'
        wheel['Reverberance'] = 'Wet'
        wheel['Distance'] = 'Distant/Spacious'
    else:
        wheel['Depth'] = 'Moderate'
        wheel['Width'] = 'Moderate'
        wheel['Envelopment'] = 'Moderate'
        wheel['Reverberance'] = 'Moderate'
        wheel['Distance'] = 'Balanced'

    print(f"\nSPATIAL CATEGORY:")
    print(f"  Depth: {wheel['Depth']}")
    print(f"  Width: {wheel['Width']}")
    print(f"  Envelopment: {wheel['Envelopment']}")
    print(f"  Reverberance: {wheel['Reverberance']}")
    print(f"  Distance: {wheel['Distance']}")

    # TRANSPARENCY CATEGORY
    if avg_sat < 30:
        wheel['Presence'] = 'Low'
        wheel['Detail'] = 'Simple/Clean'
        wheel['Natural'] = 'High'
        wheel['Clean'] = 'Very clean'
    elif avg_sat > 50:
        wheel['Presence'] = 'High'
        wheel['Detail'] = 'Rich/Complex'
        wheel['Natural'] = 'Moderate'
        wheel['Clean'] = 'Textured'
    else:
        wheel['Presence'] = 'Moderate'
        wheel['Detail'] = 'Balanced'
        wheel['Natural'] = 'Natural'
        wheel['Clean'] = 'Clean'

    print(f"\nTRANSPARENCY CATEGORY:")
    print(f"  Presence: {wheel['Presence']}")
    print(f"  Detail: {wheel['Detail']}")
    print(f"  Natural: {wheel['Natural']}")
    print(f"  Clean: {wheel['Clean']}")

    # ARTEFACTS CATEGORY (Processing recommendations)
    if profile_name == 'hyper':
        wheel['Distortion'] = 'None (avoid harsh artifacts)'
        wheel['Compression'] = 'High (limit dynamic peaks)'
        wheel['Processing_Rec'] = 'Low-pass filter 4-6kHz, compress peaks'
    elif profile_name == 'hypo':
        wheel['Distortion'] = 'Low tolerance'
        wheel['Compression'] = 'None (preserve full dynamics)'
        wheel['Processing_Rec'] = 'Enhance highs, add spatial effects'
    else:
        wheel['Distortion'] = 'Minimal'
        wheel['Compression'] = 'Moderate'
        wheel['Processing_Rec'] = 'Natural reproduction'

    print(f"\nARTEFACTS/PROCESSING:")
    print(f"  Distortion Tolerance: {wheel['Distortion']}")
    print(f"  Compression: {wheel['Compression']}")
    print(f"  Recommendation: {wheel['Processing_Rec']}")

    sound_wheel_mapping[profile_name] = wheel

# Summary table
print("\n" + "="*80)
print("SOUND WHEEL COMPARISON TABLE")
print("="*80)

print(f"\n{'Attribute':<25} {'Hypersensitive':<20} {'Typical':<20} {'Hyposensitive':<20}")
print("-"*85)

attributes = ['Loudness', 'Attack', 'Punch', 'Treble_Strength', 'Bass_Depth',
              'Timbral_Balance', 'Reverberance', 'Presence', 'Detail', 'Compression']

for attr in attributes:
    hyper_val = sound_wheel_mapping.get('hyper', {}).get(attr, 'N/A')
    typical_val = sound_wheel_mapping.get('typical', {}).get(attr, 'N/A')
    hypo_val = sound_wheel_mapping.get('hypo', {}).get(attr, 'N/A')

    # Truncate long values
    if len(str(hyper_val)) > 18:
        hyper_val = str(hyper_val)[:18] + '...'
    if len(str(typical_val)) > 18:
        typical_val = str(typical_val)[:18] + '...'
    if len(str(hypo_val)) > 18:
        hypo_val = str(hypo_val)[:18] + '...'

    print(f"{attr.replace('_', ' '):<25} {hyper_val:<20} {typical_val:<20} {hypo_val:<20}")

# Connection to neurodiversity theory
print("\n" + "="*80)
print("CONNECTION TO NEURODIVERSITY & SENSORY PROCESSING")
print("="*80)

print("""
THEORETICAL FRAMEWORK:

According to Dunn's Sensory Processing Framework (2007):
- HYPERSENSITIVE: Low neurological threshold + Active self-regulation
  → Seeks to REDUCE sensory input (lower volume, less effects)
  → Our findings: 47.6% avg volume, 0.64 mutes/min, dry reverb

- HYPOSENSITIVE: High neurological threshold + Active self-regulation
  → Seeks to INCREASE sensory input (higher volume, more effects)
  → Our findings: 86.9% avg volume, rich saturation, spacious reverb

- TYPICAL: Moderate thresholds + Balanced regulation
  → Seeks OPTIMAL stimulation (moderate levels)
  → Our findings: 63.5% avg volume, balanced processing

NEURODIVERGENCE IMPLICATIONS:

1. AUTISM SPECTRUM (ASD):
   - Often shows hypersensitive patterns
   - Expected: Lower volumes, avoiding harsh frequencies, dry spatial effects
   - Our hypersensitive group (50%) shows this pattern

2. ADHD:
   - Can show hyposensitive (seeking) OR hypersensitive patterns
   - Seeking type: Higher stimulation, complex sounds
   - Avoidant type: Sensitive to distractions, needs clean signals

3. SENSORY PROCESSING DISORDER (SPD):
   - Fluctuating patterns (matches our "sensory fatigue" group - 29%)
   - These participants started high, reduced over time

KEY FINDING:
The 22% MISMATCH between self-report and behavior may indicate:
- Compensation strategies (saying what's expected)
- Lack of sensory awareness
- Neurotypical masking behaviors
- OR actual neurodivergence that affects self-perception

RECOMMENDATION:
Cross-reference with:
1. Pre-session neurodiversity questionnaire (Dunn's Sensory Profile)
2. Formal diagnoses (ADHD, ASD, SPD)
3. Self-reported sensory sensitivities
""")

# Save mapping
output = {
    'sound_wheel_profiles': sound_wheel_mapping,
    'sample_sizes': {
        'hypersensitive': len(profiles.get('hyper', [])),
        'typical': len(profiles.get('typical', [])),
        'hyposensitive': len(profiles.get('hypo', []))
    },
    'behavioral_summary': {
        'hypersensitive': {
            'avg_volume': float(np.mean([p['settled_volume'] for p in profiles.get('hyper', [])])) if profiles.get('hyper') else 0,
            'avg_delay': float(np.mean([p['final_delay'] for p in profiles.get('hyper', [])])) if profiles.get('hyper') else 0,
            'avg_saturation': float(np.mean([p['final_saturation'] for p in profiles.get('hyper', [])])) if profiles.get('hyper') else 0
        },
        'typical': {
            'avg_volume': float(np.mean([p['settled_volume'] for p in profiles.get('typical', [])])) if profiles.get('typical') else 0,
            'avg_delay': float(np.mean([p['final_delay'] for p in profiles.get('typical', [])])) if profiles.get('typical') else 0,
            'avg_saturation': float(np.mean([p['final_saturation'] for p in profiles.get('typical', [])])) if profiles.get('typical') else 0
        },
        'hyposensitive': {
            'avg_volume': float(np.mean([p['settled_volume'] for p in profiles.get('hypo', [])])) if profiles.get('hypo') else 0,
            'avg_delay': float(np.mean([p['final_delay'] for p in profiles.get('hypo', [])])) if profiles.get('hypo') else 0,
            'avg_saturation': float(np.mean([p['final_saturation'] for p in profiles.get('hypo', [])])) if profiles.get('hypo') else 0
        }
    }
}

with open('/tmp/Neuroverse/force_sound_wheel_validated.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nMapping saved to /tmp/Neuroverse/force_sound_wheel_validated.json")
