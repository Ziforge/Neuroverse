#!/usr/bin/env python3
"""
Neuroverse Participant Data Analysis
Classifies participants into Hypersensitive, Hyposensitive, or Typical groups
Maps results to FORCE Technology Sound Wheel attributes
"""

import os
import re
from collections import defaultdict
from pathlib import Path

class ParticipantAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.participants = {}

    def parse_file(self, filepath):
        """Parse a single participant file and extract key metrics."""
        data = {
            'questionnaire_responses': {},
            'volume_settings': [],
            'eq_settings': [],
            'reverb_settings': [],
            'delay_settings': [],
            'pitch_settings': [],
            'saturation_settings': [],
            'mute_events': 0,
            'total_adjustments': 0,
            'session_duration_minutes': 0
        }

        timestamps = []

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Extract timestamp
                time_match = re.match(r'(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', line)
                if time_match:
                    timestamps.append(time_match.group(1))

                # Parse questionnaire responses (initial choices)
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

                # Parse volume adjustments
                vol_match = re.search(r'Volume_VolSlider_\w+: ([\d.]+)%', line)
                if vol_match:
                    data['volume_settings'].append(float(vol_match.group(1)))
                    data['total_adjustments'] += 1

                # Parse EQ settings
                eq_match = re.search(r'EQ_\w+: Gain:([\d.]+)dB,Frequency:(\d+)Hz', line)
                if eq_match:
                    data['eq_settings'].append({
                        'gain': float(eq_match.group(1)),
                        'freq': int(eq_match.group(2))
                    })

                # Parse reverb
                rev_match = re.search(r'Reverb_\w+: Decay:([\d.]+)ms,Room:([-\d]+)mB', line)
                if rev_match:
                    data['reverb_settings'].append({
                        'decay': float(rev_match.group(1)),
                        'room': int(rev_match.group(2))
                    })

                # Parse delay
                delay_match = re.search(r'Delay_\w+: (\d+)%', line)
                if delay_match:
                    data['delay_settings'].append(int(delay_match.group(1)))

                # Parse pitch
                pitch_match = re.search(r'PitchShift_\w+: ([\d.]+)x', line)
                if pitch_match:
                    data['pitch_settings'].append(float(pitch_match.group(1)))

                # Parse saturation
                sat_match = re.search(r'Saturation_\w+: (\d+)%', line)
                if sat_match:
                    data['saturation_settings'].append(int(sat_match.group(1)))

                # Count mute events
                if 'MuteAllAudio: True' in line or 'Muted' in line:
                    data['mute_events'] += 1

        # Calculate session duration
        if len(timestamps) >= 2:
            from datetime import datetime
            try:
                start = datetime.strptime(timestamps[0], '%Y/%m/%d %H:%M:%S')
                end = datetime.strptime(timestamps[-1], '%Y/%m/%d %H:%M:%S')
                data['session_duration_minutes'] = (end - start).total_seconds() / 60
            except:
                pass

        return data

    def calculate_sensory_score(self, data):
        """Calculate a sensory profile score based on participant data."""
        score = {
            'hyper_indicators': 0,
            'hypo_indicators': 0,
            'typical_indicators': 0
        }

        # 1. Questionnaire responses analysis
        q_responses = data['questionnaire_responses']

        # Sound Environment preference
        if q_responses.get('sound_env') == 1:  # Quiet/sensitive
            score['hyper_indicators'] += 2
        elif q_responses.get('sound_env') == 2:  # Moderate
            score['typical_indicators'] += 2
        elif q_responses.get('sound_env') == 3:  # Loud/stimulating
            score['hypo_indicators'] += 2

        # Sound Texture preference
        if q_responses.get('sound_texture') == 1:  # Warm/soft
            score['hyper_indicators'] += 1
        elif q_responses.get('sound_texture') == 2:  # Balanced
            score['typical_indicators'] += 1
        elif q_responses.get('sound_texture') == 3:  # Bright/sharp
            score['hypo_indicators'] += 1

        # Saturation tolerance
        if q_responses.get('saturation') == 1:  # Overwhelming
            score['hyper_indicators'] += 2
        elif q_responses.get('saturation') == 2:  # Tolerate mild
            score['typical_indicators'] += 1
        elif q_responses.get('saturation') == 3:  # Enjoy it
            score['hypo_indicators'] += 2

        # Delay preference
        if q_responses.get('delay') == 1:  # Prefer dry
            score['hyper_indicators'] += 1
        elif q_responses.get('delay') == 2:  # Subtle OK
            score['typical_indicators'] += 1
        elif q_responses.get('delay') == 3:  # Enjoy pronounced
            score['hypo_indicators'] += 1

        # Reverb preference
        if q_responses.get('reverb') == 1:  # Flat sound
            score['hyper_indicators'] += 1
        elif q_responses.get('reverb') == 2:  # Light reverb
            score['typical_indicators'] += 1
        elif q_responses.get('reverb') == 3:  # Rich reverb
            score['hypo_indicators'] += 1

        # 2. Behavioral analysis

        # Volume preferences
        if data['volume_settings']:
            avg_volume = sum(data['volume_settings']) / len(data['volume_settings'])
            final_volume = data['volume_settings'][-1] if data['volume_settings'] else 50

            if avg_volume < 40:
                score['hyper_indicators'] += 3
            elif avg_volume < 60:
                score['typical_indicators'] += 2
            else:
                score['hypo_indicators'] += 3

            # Check for seeking behavior (high final volume)
            if final_volume > 80:
                score['hypo_indicators'] += 2
            elif final_volume < 30:
                score['hyper_indicators'] += 2

        # Muting frequency (normalized by session duration)
        if data['session_duration_minutes'] > 0:
            mutes_per_minute = data['mute_events'] / data['session_duration_minutes']
            if mutes_per_minute > 0.5:  # High muting frequency
                score['hyper_indicators'] += 3
            elif mutes_per_minute > 0.1:
                score['typical_indicators'] += 1

        # Adjustment frequency (exploration vs. stability)
        if data['session_duration_minutes'] > 0:
            adjustments_per_minute = data['total_adjustments'] / data['session_duration_minutes']
            if adjustments_per_minute > 50:  # High seeking behavior
                score['hypo_indicators'] += 2
            elif adjustments_per_minute < 10:  # Low seeking
                score['hyper_indicators'] += 1

        # EQ preferences
        if data['eq_settings']:
            avg_freq = sum(s['freq'] for s in data['eq_settings']) / len(data['eq_settings'])
            if avg_freq > 4000:  # High frequency preference (bright)
                score['hypo_indicators'] += 1
            elif avg_freq < 500:  # Low frequency preference (warm)
                score['hyper_indicators'] += 1
            else:
                score['typical_indicators'] += 1

        return score

    def classify_participant(self, score):
        """Classify participant based on sensory score."""
        total = sum(score.values())
        if total == 0:
            return 'Typical', 0.33

        hyper_pct = score['hyper_indicators'] / total
        hypo_pct = score['hypo_indicators'] / total
        typical_pct = score['typical_indicators'] / total

        if hyper_pct > hypo_pct and hyper_pct > typical_pct:
            return 'Hypersensitive', hyper_pct
        elif hypo_pct > hyper_pct and hypo_pct > typical_pct:
            return 'Hyposensitive', hypo_pct
        else:
            return 'Typical', typical_pct

    def map_to_sound_wheel(self, classification, data):
        """Map sensory profile to FORCE Technology Sound Wheel attributes."""
        sound_wheel_profile = {}

        if classification == 'Hypersensitive':
            sound_wheel_profile = {
                'Loudness': 'Soft',
                'Attack': 'Imprecise (smoothed)',
                'Bass Precision': 'Moderate',
                'Punch': 'Weak',
                'Treble Strength': 'A little',
                'Brilliance': 'Low',
                'Tinny': 'A little',
                'Midrange Strength': 'A little',
                'Bass Strength': 'Neutral',
                'Bass Depth': 'A little',
                'Timbral Balance': 'Dark/Warm',
                'Full': 'Low degree',
                'Depth': 'Shallow',
                'Width': 'Small',
                'Envelopment': 'Small',
                'Reverberance': 'Dry',
                'Clarity': 'Clear (but filtered)',
                'Presence': 'A little',
                'Detail': 'Simple',
                'Natural': 'A lot',
                'Distortion': 'A little',
                'Compression': 'A lot (limited dynamics)'
            }
        elif classification == 'Hyposensitive':
            sound_wheel_profile = {
                'Loudness': 'Loud',
                'Attack': 'Precise',
                'Bass Precision': 'Precise',
                'Punch': 'Strong',
                'Treble Strength': 'A lot',
                'Brilliance': 'High',
                'Tinny': 'A lot',
                'Midrange Strength': 'A lot',
                'Bass Strength': 'A lot',
                'Bass Depth': 'A lot',
                'Timbral Balance': 'Bright',
                'Full': 'High degree',
                'Depth': 'Deep',
                'Width': 'Large',
                'Envelopment': 'Large',
                'Reverberance': 'Wet',
                'Clarity': 'Clear',
                'Presence': 'A lot',
                'Detail': 'Rich',
                'Natural': 'Neutral',
                'Distortion': 'Moderate',
                'Compression': 'A little (full dynamics)'
            }
        else:  # Typical
            sound_wheel_profile = {
                'Loudness': 'Neutral',
                'Attack': 'Moderate',
                'Bass Precision': 'Moderate',
                'Punch': 'Moderate',
                'Treble Strength': 'Neutral',
                'Brilliance': 'Moderate',
                'Tinny': 'Neutral',
                'Midrange Strength': 'Neutral',
                'Bass Strength': 'Neutral',
                'Bass Depth': 'Neutral',
                'Timbral Balance': 'Neutral',
                'Full': 'Moderate',
                'Depth': 'Moderate',
                'Width': 'Moderate',
                'Envelopment': 'Moderate',
                'Reverberance': 'Moderate',
                'Clarity': 'Clear',
                'Presence': 'Moderate',
                'Detail': 'Moderate',
                'Natural': 'A lot',
                'Distortion': 'A little',
                'Compression': 'Neutral'
            }

        # Customize based on actual participant data
        if data['volume_settings']:
            avg_vol = sum(data['volume_settings']) / len(data['volume_settings'])
            if avg_vol < 30:
                sound_wheel_profile['Loudness'] = 'Very Soft (30dB)'
            elif avg_vol < 50:
                sound_wheel_profile['Loudness'] = f'Soft ({avg_vol:.0f}%)'
            elif avg_vol < 70:
                sound_wheel_profile['Loudness'] = f'Medium ({avg_vol:.0f}%)'
            else:
                sound_wheel_profile['Loudness'] = f'Loud ({avg_vol:.0f}%)'

        return sound_wheel_profile

    def analyze_all_participants(self):
        """Analyze all participant files in the data directory."""
        results = []

        for filepath in sorted(self.data_dir.glob('participant*.txt')):
            if 'Extra' in filepath.name or 'partially' in filepath.name:
                continue  # Skip metadata files

            participant_id = filepath.stem
            print(f"Analyzing {participant_id}...")

            data = self.parse_file(filepath)
            score = self.calculate_sensory_score(data)
            classification, confidence = self.classify_participant(score)
            sound_wheel = self.map_to_sound_wheel(classification, data)

            results.append({
                'id': participant_id,
                'classification': classification,
                'confidence': confidence,
                'score': score,
                'sound_wheel': sound_wheel,
                'raw_data': {
                    'avg_volume': sum(data['volume_settings']) / len(data['volume_settings']) if data['volume_settings'] else 0,
                    'mute_events': data['mute_events'],
                    'session_minutes': data['session_duration_minutes'],
                    'total_adjustments': data['total_adjustments'],
                    'questionnaire': data['questionnaire_responses']
                }
            })

        self.participants = results
        return results

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        if not self.participants:
            self.analyze_all_participants()

        report = []
        report.append("=" * 80)
        report.append("NEUROVERSE PARTICIPANT SENSORY PROFILE ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        hyper_count = sum(1 for p in self.participants if p['classification'] == 'Hypersensitive')
        hypo_count = sum(1 for p in self.participants if p['classification'] == 'Hyposensitive')
        typical_count = sum(1 for p in self.participants if p['classification'] == 'Typical')

        report.append(f"Total Participants: {len(self.participants)}")
        report.append(f"  Hypersensitive: {hyper_count} ({hyper_count/len(self.participants)*100:.1f}%)")
        report.append(f"  Hyposensitive: {hypo_count} ({hypo_count/len(self.participants)*100:.1f}%)")
        report.append(f"  Typical: {typical_count} ({typical_count/len(self.participants)*100:.1f}%)")
        report.append("")

        # Individual results
        report.append("-" * 80)
        report.append("INDIVIDUAL PARTICIPANT CLASSIFICATIONS")
        report.append("-" * 80)

        for p in self.participants:
            report.append(f"\n{p['id'].upper()}")
            report.append(f"  Classification: {p['classification']} (Confidence: {p['confidence']:.1%})")
            report.append(f"  Sensory Score: Hyper={p['score']['hyper_indicators']}, "
                         f"Hypo={p['score']['hypo_indicators']}, "
                         f"Typical={p['score']['typical_indicators']}")
            report.append(f"  Avg Volume: {p['raw_data']['avg_volume']:.1f}%")
            report.append(f"  Mute Events: {p['raw_data']['mute_events']}")
            report.append(f"  Session Duration: {p['raw_data']['session_minutes']:.1f} min")
            report.append(f"  Total Adjustments: {p['raw_data']['total_adjustments']}")

            if p['raw_data']['questionnaire']:
                report.append(f"  Questionnaire Responses: {p['raw_data']['questionnaire']}")

        # Sound Wheel Mapping Summary
        report.append("\n" + "=" * 80)
        report.append("FORCE TECHNOLOGY SOUND WHEEL MAPPING")
        report.append("=" * 80)

        for profile_type in ['Hypersensitive', 'Hyposensitive', 'Typical']:
            participants_of_type = [p for p in self.participants if p['classification'] == profile_type]
            if participants_of_type:
                report.append(f"\n{profile_type.upper()} PROFILE Sound Wheel Characteristics:")
                sample = participants_of_type[0]['sound_wheel']
                for attr, value in sample.items():
                    report.append(f"  {attr}: {value}")

        return '\n'.join(report)

    def generate_latex_table(self):
        """Generate LaTeX table for the dissertation."""
        if not self.participants:
            self.analyze_all_participants()

        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Participant Sensory Profile Classifications}")
        latex.append(r"\label{tab:classifications}")
        latex.append(r"\begin{tabular}{lcccccc}")
        latex.append(r"\hline")
        latex.append(r"\textbf{Participant} & \textbf{Class} & \textbf{Conf.} & \textbf{Avg Vol} & \textbf{Mutes} & \textbf{Adj.} & \textbf{Duration} \\")
        latex.append(r"\hline")

        for p in self.participants:
            class_short = p['classification'][:5]
            latex.append(f"{p['id']} & {class_short} & {p['confidence']:.0%} & "
                        f"{p['raw_data']['avg_volume']:.0f}\\% & {p['raw_data']['mute_events']} & "
                        f"{p['raw_data']['total_adjustments']} & {p['raw_data']['session_minutes']:.1f}m \\\\")

        latex.append(r"\hline")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        return '\n'.join(latex)


def main():
    analyzer = ParticipantAnalyzer("/tmp/Neuroverse/Test Data")

    print("Analyzing participant data...")
    analyzer.analyze_all_participants()

    # Generate and print report
    report = analyzer.generate_report()
    print(report)

    # Save report
    with open("/tmp/Neuroverse/analysis_report.txt", 'w') as f:
        f.write(report)

    # Generate LaTeX table
    latex_table = analyzer.generate_latex_table()
    with open("/tmp/Neuroverse/latex_classification_table.tex", 'w') as f:
        f.write(latex_table)

    print("\n\nReport saved to: /tmp/Neuroverse/analysis_report.txt")
    print("LaTeX table saved to: /tmp/Neuroverse/latex_classification_table.tex")


if __name__ == "__main__":
    main()
