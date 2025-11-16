#!/usr/bin/env python3
"""
Create visualizations for Neuroverse analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def load_data():
    with open("/tmp/Neuroverse/visualization_data.json", 'r') as f:
        return json.load(f)

def plot_classification_distribution(data):
    """Pie chart of sensory profile distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    counts = data['classification_counts']
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=90,
        textprops={'fontsize': 12}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Participant Sensory Profile Distribution\n(n=20)', fontsize=14, fontweight='bold')

    # Add count annotations
    legend_labels = [f'{l}: {s} participants' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='lower left', bbox_to_anchor=(0, -0.1))

    plt.tight_layout()
    plt.savefig('/tmp/Neuroverse/fig_classification_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('/tmp/Neuroverse/fig_classification_distribution.pdf', bbox_inches='tight')
    plt.close()

def plot_volume_comparison(data):
    """Box plot comparing volume preferences by classification."""
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = ['Hypersensitive', 'Typical', 'Hyposensitive']
    colors = ['#FF6B6B', '#45B7D1', '#4ECDC4']

    box_data = [data['volume_by_class'][cls] for cls in classes]

    bp = ax.boxplot(box_data, labels=classes, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (cls, color) in enumerate(zip(classes, colors), 1):
        y = data['volume_by_class'][cls]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, color='black', s=30, zorder=3)

    # Add reference lines for dB ranges
    ax.axhline(y=35, color='red', linestyle='--', alpha=0.5, label='Hyper threshold (35%)')
    ax.axhline(y=65, color='green', linestyle='--', alpha=0.5, label='Hypo threshold (65%)')

    ax.set_ylabel('Average Volume (%)', fontsize=12)
    ax.set_title('Volume Preferences by Sensory Profile', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    # Add sample sizes
    for i, cls in enumerate(classes, 1):
        n = len(data['volume_by_class'][cls])
        ax.text(i, ax.get_ylim()[0] - 5, f'n={n}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('/tmp/Neuroverse/fig_volume_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('/tmp/Neuroverse/fig_volume_comparison.pdf', bbox_inches='tight')
    plt.close()

def plot_behavioral_metrics(data):
    """Bar chart comparing key behavioral metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    classes = ['Hypersensitive', 'Typical', 'Hyposensitive']
    colors = ['#FF6B6B', '#45B7D1', '#4ECDC4']

    # Muting behavior
    ax = axes[0]
    means = [np.mean(data['mutes_by_class'][cls]) if data['mutes_by_class'][cls] else 0
             for cls in classes]
    stds = [np.std(data['mutes_by_class'][cls]) if len(data['mutes_by_class'][cls]) > 1 else 0
            for cls in classes]

    bars = ax.bar(classes, means, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(classes, means, yerr=stds, fmt='none', color='black', capsize=5)
    ax.set_ylabel('Mean Mute Events')
    ax.set_title('Muting Frequency')

    # Add values on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    # Adjustment rate
    ax = axes[1]
    means = [np.mean(data['adj_rate_by_class'][cls]) if data['adj_rate_by_class'][cls] else 0
             for cls in classes]
    stds = [np.std(data['adj_rate_by_class'][cls]) if len(data['adj_rate_by_class'][cls]) > 1 else 0
            for cls in classes]

    bars = ax.bar(classes, means, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(classes, means, yerr=stds, fmt='none', color='black', capsize=5)
    ax.set_ylabel('Mean Adjustments/Minute')
    ax.set_title('Parameter Adjustment Rate')

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    # Volume means
    ax = axes[2]
    means = [np.mean(data['volume_by_class'][cls]) if data['volume_by_class'][cls] else 0
             for cls in classes]
    stds = [np.std(data['volume_by_class'][cls]) if len(data['volume_by_class'][cls]) > 1 else 0
            for cls in classes]

    bars = ax.bar(classes, means, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(classes, means, yerr=stds, fmt='none', color='black', capsize=5)
    ax.set_ylabel('Mean Volume (%)')
    ax.set_title('Average Volume Preference')

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Behavioral Metrics by Sensory Profile', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/tmp/Neuroverse/fig_behavioral_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('/tmp/Neuroverse/fig_behavioral_metrics.pdf', bbox_inches='tight')
    plt.close()

def plot_sound_wheel_radar(data):
    """Radar chart comparing Sound Wheel attributes across profiles."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    categories = ['Loudness', 'Attack', 'Punch', 'Treble_Strength', 'Bass_Depth',
                  'Timbral_Balance', 'Depth', 'Width', 'Reverberance', 'Presence', 'Detail']

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    colors = {'Hypersensitive': '#FF6B6B', 'Typical': '#45B7D1', 'Hyposensitive': '#4ECDC4'}

    for cls in ['Hypersensitive', 'Typical', 'Hyposensitive']:
        if cls in data['sound_wheel_scales'] and data['sound_wheel_scales'][cls]:
            values = [data['sound_wheel_scales'][cls].get(cat, 5) for cat in categories]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=cls, color=colors[cls])
            ax.fill(angles, values, alpha=0.25, color=colors[cls])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], size=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8])
    ax.set_yticklabels(['Low', 'Med-Low', 'Med-High', 'High'], size=8)
    ax.grid(True)

    ax.set_title('Sound Wheel Profile Comparison\n(FORCE Technology Attributes)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig('/tmp/Neuroverse/fig_sound_wheel_radar.png', dpi=300, bbox_inches='tight')
    plt.savefig('/tmp/Neuroverse/fig_sound_wheel_radar.pdf', bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(data):
    """Histogram of classification confidence scores."""
    fig, ax = plt.subplots(figsize=(10, 6))

    confidences = {
        'Hypersensitive': [],
        'Typical': [],
        'Hyposensitive': []
    }

    for p in data['confidence_distribution']:
        confidences[p['class']].append(p['confidence'] * 100)

    colors = ['#FF6B6B', '#45B7D1', '#4ECDC4']
    classes = ['Hypersensitive', 'Typical', 'Hyposensitive']

    x = np.arange(len(data['confidence_distribution']))
    width = 0.8

    # Sort by confidence
    sorted_data = sorted(data['confidence_distribution'], key=lambda x: x['confidence'])

    for i, p in enumerate(sorted_data):
        color = colors[classes.index(p['class'])]
        ax.bar(i, p['confidence'] * 100, color=color, alpha=0.7, edgecolor='black')

    ax.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='High Confidence (60%)')
    ax.axhline(y=45, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence (45%)')

    ax.set_xlabel('Participants (sorted by confidence)')
    ax.set_ylabel('Classification Confidence (%)')
    ax.set_title('Classification Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend()

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.7, label=cls)
                      for c, cls in zip(colors, classes)]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='upper left')

    plt.tight_layout()
    plt.savefig('/tmp/Neuroverse/fig_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('/tmp/Neuroverse/fig_confidence_distribution.pdf', bbox_inches='tight')
    plt.close()

def plot_questionnaire_heatmap(data):
    """Heatmap of questionnaire responses by classification."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create matrix for heatmap
    questions = ['sound_env', 'sound_texture', 'saturation', 'delay', 'reverb', 'pitch_shift']
    classes = ['Hypersensitive', 'Typical', 'Hyposensitive']

    # Count option distributions per class
    matrix = []
    row_labels = []

    for q in questions:
        for opt in [1, 2, 3]:
            row = []
            for cls in classes:
                key = f"{cls}_{opt}"
                count = data['questionnaire_patterns'].get(q, {}).get(key, 0)
                row.append(count)
            matrix.append(row)
            row_labels.append(f"{q.replace('_', ' ').title()} - Opt {opt}")

    matrix = np.array(matrix)

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(row_labels, fontsize=9)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(classes)):
            text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Questionnaire Response Patterns by Sensory Profile', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im)
    cbar.set_label('Number of Participants', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig('/tmp/Neuroverse/fig_questionnaire_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('/tmp/Neuroverse/fig_questionnaire_heatmap.pdf', bbox_inches='tight')
    plt.close()

def main():
    print("Loading visualization data...")
    data = load_data()

    print("Creating classification distribution chart...")
    plot_classification_distribution(data)

    print("Creating volume comparison boxplot...")
    plot_volume_comparison(data)

    print("Creating behavioral metrics comparison...")
    plot_behavioral_metrics(data)

    print("Creating Sound Wheel radar chart...")
    plot_sound_wheel_radar(data)

    print("Creating confidence distribution plot...")
    plot_confidence_distribution(data)

    print("Creating questionnaire heatmap...")
    plot_questionnaire_heatmap(data)

    print("\nAll visualizations saved to /tmp/Neuroverse/")
    print("Files created:")
    for f in Path("/tmp/Neuroverse").glob("fig_*.png"):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
