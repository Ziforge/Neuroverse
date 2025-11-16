#!/usr/bin/env python3
"""
Neuroverse Participant Data Analysis v2
Enhanced classification with refined weighting and visualizations
"""

import os
import re
from collections import defaultdict
from pathlib import Path
import json

class ParticipantAnalyzerV2:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.participants = []

    def parse_file(self, filepath):
        """Parse a single participant file with enhanced metrics."""
        data = {
            'questionnaire_responses': {},
            'volume_settings': defaultdict(list),  # Per source
            'eq_settings': [],
            'reverb_settings': [],
            'delay_settings': [],
            'pitch_settings': [],
            'saturation_settings': [],
            'mute_events': {'global': 0, 'individual': 0},
            'total_adjustments': 0,
            'session_duration_minutes': 0,
            'timestamps': [],
            'adjustment_timeline': []  # Track when adjustments happen
        }

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Extract timestamp
                time_match = re.match(r'(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d+)', line)
                if time_match:
                    data['timestamps'].append(time_match.group(1))

                # Parse questionnaire responses
                if 'SoundEnvs: Option_' in line:
                    opt = re.search(r'Option_(\d)', line)
                    if opt:
                        data['questionnaire_responses']['sound_env'] = int(opt.group(1))

                if 'SoundTexture: Option_' in line:
                    opt = re.search(r'Option_(\d)', line)
                    if opt:
                        data['questionnaire_responses']['sound_texture'] = int(opt.group(1))

                if 'Saturation: Option_' in line:
                    opt = re.search(r'Option_(\d)', line)
                    if opt:
                        data['questionnaire_responses']['saturation'] = int(opt.group(1))

                if 'Delay: Option_' in line:
                    opt = re.search(r'Option_(\d)', line)
                    if opt:
                        data['questionnaire_responses']['delay'] = int(opt.group(1))

                if 'Reverb: Option_' in line:
                    opt = re.search(r'Option_(\d)', line)
                    if opt:
                        data['questionnaire_responses']['reverb'] = int(opt.group(1))

                if 'PitchShift: Option_' in line:
                    opt = re.search(r'Option_(\d)', line)
                    if opt:
                        data['questionnaire_responses']['pitch_shift'] = int(opt.group(1))

                # Parse volume adjustments per source
                vol_match = re.search(r'Volume_VolSlider_(\w+): ([\d.]+)%', line)
                if vol_match:
                    source = vol_match.group(1)
                    value = float(vol_match.group(2))
                    data['volume_settings'][source].append(value)
                    data['total_adjustments'] += 1
                    data['adjustment_timeline'].append(('volume', source, value))

                # Parse EQ settings
                eq_match = re.search(r'EQ_(\w+): Gain:([\d.]+)dB,Frequency:(\d+)Hz', line)
                if eq_match:
                    data['eq_settings'].append({
                        'source': eq_match.group(1),
                        'gain': float(eq_match.group(2)),
                        'freq': int(eq_match.group(3))
                    })

                # Parse reverb with source
                rev_match = re.search(r'Reverb_(\w+): Decay:([\d.]+)ms,Room:([-\d]+)mB', line)
                if rev_match:
                    data['reverb_settings'].append({
                        'source': rev_match.group(1),
                        'decay': float(rev_match.group(2)),
                        'room': int(rev_match.group(3))
                    })

                # Parse delay per source
                delay_match = re.search(r'Delay_DelaySlider_(\w+): (\d+)%', line)
                if delay_match:
                    data['delay_settings'].append({
                        'source': delay_match.group(1),
                        'value': int(delay_match.group(2))
                    })

                # Parse pitch per source
                pitch_match = re.search(r'PitchShift_PitchSlider[_]*(\w*): ([\d.]+)x', line)
                if pitch_match:
                    data['pitch_settings'].append(float(pitch_match.group(2)))

                # Parse saturation
                sat_match = re.search(r'Saturation_SaturationSlider_(\w+): (\d+)%', line)
                if sat_match:
                    data['saturation_settings'].append(int(sat_match.group(2)))

                # Count mute events
                if 'MuteAllAudio: True' in line:
                    data['mute_events']['global'] += 1
                elif 'Muted' in line and 'MuteToggle' in line:
                    data['mute_events']['individual'] += 1

        # Calculate session duration
        if len(data['timestamps']) >= 2:
            from datetime import datetime
            try:
                start = datetime.strptime(data['timestamps'][0][:19], '%Y/%m/%d %H:%M:%S')
                end = datetime.strptime(data['timestamps'][-1][:19], '%Y/%m/%d %H:%M:%S')
                data['session_duration_minutes'] = (end - start).total_seconds() / 60
            except:
                pass

        return data

    def calculate_sensory_score_v2(self, data):
        """Enhanced scoring with refined weights based on research literature."""
        score = {
            'hyper_indicators': 0,
            'hypo_indicators': 0,
            'typical_indicators': 0
        }

        weights = {
            'questionnaire': 2.0,  # Self-reported preferences
            'behavioral': 1.5,     # Actual behavior
            'final_state': 1.0     # End configuration
        }

        q_responses = data['questionnaire_responses']

        # === QUESTIONNAIRE ANALYSIS (Weight: 2.0) ===

        # Sound Environment preference (critical indicator)
        if q_responses.get('sound_env') == 1:  # Highly sensitive
            score['hyper_indicators'] += 3 * weights['questionnaire']
        elif q_responses.get('sound_env') == 2:  # Moderate
            score['typical_indicators'] += 2 * weights['questionnaire']
        elif q_responses.get('sound_env') == 3:  # Under-responsive/seeking
            score['hypo_indicators'] += 3 * weights['questionnaire']

        # Sound Texture preference
        if q_responses.get('sound_texture') == 1:  # Warm/soft
            score['hyper_indicators'] += 2 * weights['questionnaire']
        elif q_responses.get('sound_texture') == 2:  # Balanced
            score['typical_indicators'] += 2 * weights['questionnaire']
        elif q_responses.get('sound_texture') == 3:  # Bright/sharp
            score['hypo_indicators'] += 2 * weights['questionnaire']

        # Saturation tolerance (strong indicator)
        if q_responses.get('saturation') == 1:  # Overwhelming
            score['hyper_indicators'] += 3 * weights['questionnaire']
        elif q_responses.get('saturation') == 2:  # Tolerate mild
            score['typical_indicators'] += 2 * weights['questionnaire']
        elif q_responses.get('saturation') == 3:  # Enjoy it
            score['hypo_indicators'] += 3 * weights['questionnaire']

        # Delay preference
        if q_responses.get('delay') == 1:  # Prefer dry
            score['hyper_indicators'] += 1.5 * weights['questionnaire']
        elif q_responses.get('delay') == 2:  # Subtle OK
            score['typical_indicators'] += 1.5 * weights['questionnaire']
        elif q_responses.get('delay') == 3:  # Enjoy pronounced
            score['hypo_indicators'] += 1.5 * weights['questionnaire']

        # Reverb preference
        if q_responses.get('reverb') == 1:  # Flat sound
            score['hyper_indicators'] += 2 * weights['questionnaire']
        elif q_responses.get('reverb') == 2:  # Light reverb
            score['typical_indicators'] += 2 * weights['questionnaire']
        elif q_responses.get('reverb') == 3:  # Rich reverb
            score['hypo_indicators'] += 2 * weights['questionnaire']

        # Pitch shift preference
        if q_responses.get('pitch_shift') == 1:  # Natural/no shift
            score['hyper_indicators'] += 1 * weights['questionnaire']
        elif q_responses.get('pitch_shift') == 2:  # Moderate
            score['typical_indicators'] += 1 * weights['questionnaire']
        elif q_responses.get('pitch_shift') in [3, 4]:  # Custom/variable
            score['hypo_indicators'] += 1 * weights['questionnaire']

        # === BEHAVIORAL ANALYSIS (Weight: 1.5) ===

        # Volume analysis across all sources
        all_volumes = []
        for source, volumes in data['volume_settings'].items():
            all_volumes.extend(volumes)

        if all_volumes:
            avg_volume = sum(all_volumes) / len(all_volumes)
            final_volumes = [vols[-1] for vols in data['volume_settings'].values() if vols]
            avg_final = sum(final_volumes) / len(final_volumes) if final_volumes else 50

            # Average volume preference
            if avg_volume < 35:
                score['hyper_indicators'] += 4 * weights['behavioral']
            elif avg_volume < 50:
                score['typical_indicators'] += 3 * weights['behavioral']
            elif avg_volume < 70:
                score['hypo_indicators'] += 2 * weights['behavioral']
            else:
                score['hypo_indicators'] += 4 * weights['behavioral']

            # Final volume (settled preference)
            if avg_final < 40:
                score['hyper_indicators'] += 2 * weights['final_state']
            elif avg_final < 60:
                score['typical_indicators'] += 2 * weights['final_state']
            else:
                score['hypo_indicators'] += 2 * weights['final_state']

            # Volume variance (seeking behavior)
            if len(all_volumes) > 10:
                variance = sum((v - avg_volume)**2 for v in all_volumes) / len(all_volumes)
                if variance > 400:  # High variance = seeking/adjusting
                    score['hypo_indicators'] += 1.5 * weights['behavioral']
                elif variance < 100:  # Low variance = stable preference
                    score['typical_indicators'] += 1 * weights['behavioral']

        # Muting behavior analysis
        total_mutes = data['mute_events']['global'] + data['mute_events']['individual']
        if data['session_duration_minutes'] > 0:
            mutes_per_minute = total_mutes / data['session_duration_minutes']

            if mutes_per_minute > 0.8:  # Very frequent muting
                score['hyper_indicators'] += 4 * weights['behavioral']
            elif mutes_per_minute > 0.3:  # Moderate muting
                score['hyper_indicators'] += 2 * weights['behavioral']
            elif mutes_per_minute < 0.1:  # Rarely mutes
                score['hypo_indicators'] += 1 * weights['behavioral']

        # Adjustment frequency patterns
        if data['session_duration_minutes'] > 0:
            adj_per_min = data['total_adjustments'] / data['session_duration_minutes']

            if adj_per_min > 60:  # Very high = sensory seeking
                score['hypo_indicators'] += 3 * weights['behavioral']
            elif adj_per_min > 30:
                score['typical_indicators'] += 1 * weights['behavioral']
            elif adj_per_min < 15:  # Low adjustment = avoidance or satisfaction
                score['hyper_indicators'] += 1 * weights['behavioral']

        # EQ frequency preference
        if data['eq_settings']:
            freqs = [s['freq'] for s in data['eq_settings']]
            avg_freq = sum(freqs) / len(freqs)

            if avg_freq > 5000:  # High freq boost = bright preference
                score['hypo_indicators'] += 2 * weights['behavioral']
            elif avg_freq < 1000:  # Low freq = warm preference
                score['hyper_indicators'] += 2 * weights['behavioral']
            else:
                score['typical_indicators'] += 1 * weights['behavioral']

        # Delay settings behavior
        if data['delay_settings']:
            avg_delay = sum(d['value'] for d in data['delay_settings']) / len(data['delay_settings'])
            if avg_delay < 15:
                score['hyper_indicators'] += 1 * weights['behavioral']
            elif avg_delay > 40:
                score['hypo_indicators'] += 1 * weights['behavioral']

        # Saturation behavior
        if data['saturation_settings']:
            avg_sat = sum(data['saturation_settings']) / len(data['saturation_settings'])
            if avg_sat < 30:
                score['hyper_indicators'] += 1.5 * weights['behavioral']
            elif avg_sat > 60:
                score['hypo_indicators'] += 1.5 * weights['behavioral']

        return score

    def classify_participant(self, score):
        """Classify with confidence levels and sub-scores."""
        total = sum(score.values())
        if total == 0:
            return 'Typical', 0.33, score

        hyper_pct = score['hyper_indicators'] / total
        hypo_pct = score['hypo_indicators'] / total
        typical_pct = score['typical_indicators'] / total

        # Determine classification
        if hyper_pct > hypo_pct and hyper_pct > typical_pct:
            classification = 'Hypersensitive'
            confidence = hyper_pct
        elif hypo_pct > hyper_pct and hypo_pct > typical_pct:
            classification = 'Hyposensitive'
            confidence = hypo_pct
        else:
            classification = 'Typical'
            confidence = typical_pct

        # Add certainty level
        if confidence > 0.6:
            certainty = 'High'
        elif confidence > 0.45:
            certainty = 'Medium'
        else:
            certainty = 'Low'

        return classification, confidence, certainty

    def map_to_sound_wheel_detailed(self, classification, data):
        """Detailed Sound Wheel mapping with numerical values."""
        # Base profiles
        profiles = {
            'Hypersensitive': {
                'Loudness': {'value': 'Soft', 'dB_range': '30-40', 'scale': 2},
                'Attack': {'value': 'Imprecise', 'scale': 3},
                'Bass_Precision': {'value': 'Moderate', 'scale': 5},
                'Punch': {'value': 'Weak', 'scale': 2},
                'Treble_Strength': {'value': 'A little', 'scale': 3},
                'Brilliance': {'value': 'Low', 'scale': 3},
                'Bass_Strength': {'value': 'Neutral', 'scale': 5},
                'Bass_Depth': {'value': 'A little', 'scale': 3},
                'Timbral_Balance': {'value': 'Dark', 'scale': 2},
                'Depth': {'value': 'Shallow', 'scale': 3},
                'Width': {'value': 'Small', 'scale': 3},
                'Envelopment': {'value': 'Small', 'scale': 3},
                'Reverberance': {'value': 'Dry', 'scale': 2},
                'Clarity': {'value': 'Clear', 'scale': 7},
                'Presence': {'value': 'A little', 'scale': 3},
                'Detail': {'value': 'Simple', 'scale': 3},
                'Distortion': {'value': 'A little', 'scale': 2},
                'Compression': {'value': 'A lot', 'scale': 8}
            },
            'Hyposensitive': {
                'Loudness': {'value': 'Loud', 'dB_range': '65-75+', 'scale': 8},
                'Attack': {'value': 'Precise', 'scale': 8},
                'Bass_Precision': {'value': 'Precise', 'scale': 8},
                'Punch': {'value': 'Strong', 'scale': 8},
                'Treble_Strength': {'value': 'A lot', 'scale': 8},
                'Brilliance': {'value': 'High', 'scale': 8},
                'Bass_Strength': {'value': 'A lot', 'scale': 8},
                'Bass_Depth': {'value': 'A lot', 'scale': 8},
                'Timbral_Balance': {'value': 'Bright', 'scale': 8},
                'Depth': {'value': 'Deep', 'scale': 8},
                'Width': {'value': 'Large', 'scale': 8},
                'Envelopment': {'value': 'Large', 'scale': 8},
                'Reverberance': {'value': 'Wet', 'scale': 8},
                'Clarity': {'value': 'Clear', 'scale': 7},
                'Presence': {'value': 'A lot', 'scale': 8},
                'Detail': {'value': 'Rich', 'scale': 8},
                'Distortion': {'value': 'Moderate', 'scale': 5},
                'Compression': {'value': 'A little', 'scale': 2}
            },
            'Typical': {
                'Loudness': {'value': 'Medium', 'dB_range': '50-60', 'scale': 5},
                'Attack': {'value': 'Moderate', 'scale': 5},
                'Bass_Precision': {'value': 'Moderate', 'scale': 5},
                'Punch': {'value': 'Moderate', 'scale': 5},
                'Treble_Strength': {'value': 'Neutral', 'scale': 5},
                'Brilliance': {'value': 'Moderate', 'scale': 5},
                'Bass_Strength': {'value': 'Neutral', 'scale': 5},
                'Bass_Depth': {'value': 'Neutral', 'scale': 5},
                'Timbral_Balance': {'value': 'Neutral', 'scale': 5},
                'Depth': {'value': 'Moderate', 'scale': 5},
                'Width': {'value': 'Moderate', 'scale': 5},
                'Envelopment': {'value': 'Moderate', 'scale': 5},
                'Reverberance': {'value': 'Moderate', 'scale': 5},
                'Clarity': {'value': 'Clear', 'scale': 7},
                'Presence': {'value': 'Moderate', 'scale': 5},
                'Detail': {'value': 'Moderate', 'scale': 5},
                'Distortion': {'value': 'A little', 'scale': 3},
                'Compression': {'value': 'Neutral', 'scale': 5}
            }
        }

        profile = profiles[classification].copy()

        # Customize based on actual data
        all_volumes = []
        for vols in data['volume_settings'].values():
            all_volumes.extend(vols)

        if all_volumes:
            avg_vol = sum(all_volumes) / len(all_volumes)
            if avg_vol < 30:
                profile['Loudness']['value'] = f'Very Soft ({avg_vol:.0f}%)'
                profile['Loudness']['dB_range'] = '25-35'
                profile['Loudness']['scale'] = 1
            elif avg_vol < 45:
                profile['Loudness']['value'] = f'Soft ({avg_vol:.0f}%)'
                profile['Loudness']['dB_range'] = '35-45'
                profile['Loudness']['scale'] = 3
            elif avg_vol < 60:
                profile['Loudness']['value'] = f'Medium ({avg_vol:.0f}%)'
                profile['Loudness']['dB_range'] = '50-60'
                profile['Loudness']['scale'] = 5
            elif avg_vol < 75:
                profile['Loudness']['value'] = f'Loud ({avg_vol:.0f}%)'
                profile['Loudness']['dB_range'] = '60-70'
                profile['Loudness']['scale'] = 7
            else:
                profile['Loudness']['value'] = f'Very Loud ({avg_vol:.0f}%)'
                profile['Loudness']['dB_range'] = '70+'
                profile['Loudness']['scale'] = 9

        return profile

    def analyze_all_participants(self):
        """Analyze all participants with enhanced metrics."""
        results = []

        for filepath in sorted(self.data_dir.glob('participant*.txt')):
            if 'Extra' in filepath.name or 'partially' in filepath.name:
                continue

            participant_id = filepath.stem
            print(f"Analyzing {participant_id}...")

            data = self.parse_file(filepath)
            score = self.calculate_sensory_score_v2(data)
            classification, confidence, certainty = self.classify_participant(score)
            sound_wheel = self.map_to_sound_wheel_detailed(classification, data)

            # Calculate additional metrics
            all_volumes = []
            for vols in data['volume_settings'].values():
                all_volumes.extend(vols)
            avg_vol = sum(all_volumes) / len(all_volumes) if all_volumes else 0

            results.append({
                'id': participant_id,
                'classification': classification,
                'confidence': confidence,
                'certainty': certainty,
                'score': score,
                'sound_wheel': sound_wheel,
                'metrics': {
                    'avg_volume': avg_vol,
                    'mute_events': data['mute_events']['global'] + data['mute_events']['individual'],
                    'session_minutes': data['session_duration_minutes'],
                    'total_adjustments': data['total_adjustments'],
                    'questionnaire': data['questionnaire_responses'],
                    'adj_per_minute': data['total_adjustments'] / max(data['session_duration_minutes'], 0.1)
                }
            })

        self.participants = results
        return results

    def generate_visualization_data(self):
        """Generate data for matplotlib visualizations."""
        if not self.participants:
            self.analyze_all_participants()

        viz_data = {
            'classification_counts': {'Hypersensitive': 0, 'Hyposensitive': 0, 'Typical': 0},
            'volume_by_class': {'Hypersensitive': [], 'Hyposensitive': [], 'Typical': []},
            'mutes_by_class': {'Hypersensitive': [], 'Hyposensitive': [], 'Typical': []},
            'adj_rate_by_class': {'Hypersensitive': [], 'Hyposensitive': [], 'Typical': []},
            'confidence_distribution': [],
            'questionnaire_patterns': defaultdict(lambda: defaultdict(int)),
            'sound_wheel_scales': {'Hypersensitive': {}, 'Hyposensitive': {}, 'Typical': {}}
        }

        for p in self.participants:
            cls = p['classification']
            viz_data['classification_counts'][cls] += 1
            viz_data['volume_by_class'][cls].append(p['metrics']['avg_volume'])
            viz_data['mutes_by_class'][cls].append(p['metrics']['mute_events'])
            viz_data['adj_rate_by_class'][cls].append(p['metrics']['adj_per_minute'])
            viz_data['confidence_distribution'].append({
                'id': p['id'],
                'class': cls,
                'confidence': p['confidence']
            })

            # Questionnaire patterns
            for key, val in p['metrics']['questionnaire'].items():
                viz_data['questionnaire_patterns'][key][f"{cls}_{val}"] += 1

        # Average sound wheel scales per class
        for cls in ['Hypersensitive', 'Hyposensitive', 'Typical']:
            cls_participants = [p for p in self.participants if p['classification'] == cls]
            if cls_participants:
                sample_wheel = cls_participants[0]['sound_wheel']
                for attr, data in sample_wheel.items():
                    viz_data['sound_wheel_scales'][cls][attr] = data['scale']

        return viz_data

    def save_visualization_data(self):
        """Save data for external visualization."""
        viz_data = self.generate_visualization_data()

        with open("/tmp/Neuroverse/visualization_data.json", 'w') as f:
            json.dump(viz_data, f, indent=2, default=str)

        return viz_data


def main():
    analyzer = ParticipantAnalyzerV2("/tmp/Neuroverse/Test Data")

    print("Running enhanced analysis...")
    results = analyzer.analyze_all_participants()

    # Save visualization data
    viz_data = analyzer.save_visualization_data()

    # Print summary
    print("\n" + "="*80)
    print("ENHANCED CLASSIFICATION RESULTS")
    print("="*80)

    counts = viz_data['classification_counts']
    total = sum(counts.values())

    print(f"\nTotal Participants: {total}")
    for cls, count in counts.items():
        print(f"  {cls}: {count} ({count/total*100:.1f}%)")

    print("\n" + "-"*80)
    print("INDIVIDUAL RESULTS")
    print("-"*80)

    for p in results:
        print(f"\n{p['id']}")
        print(f"  Classification: {p['classification']} ({p['certainty']} confidence: {p['confidence']:.1%})")
        print(f"  Scores: H={p['score']['hyper_indicators']:.1f}, Hypo={p['score']['hypo_indicators']:.1f}, T={p['score']['typical_indicators']:.1f}")
        print(f"  Avg Volume: {p['metrics']['avg_volume']:.1f}%")
        print(f"  Mutes: {p['metrics']['mute_events']}, Adj/min: {p['metrics']['adj_per_minute']:.1f}")

    print(f"\n\nVisualization data saved to: /tmp/Neuroverse/visualization_data.json")

    return results, viz_data


if __name__ == "__main__":
    main()
