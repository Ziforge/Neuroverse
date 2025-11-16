#!/usr/bin/env python3
"""
Create visualization showing questionnaire vs behavior alignment.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/tmp/Neuroverse/questionnaire_behavior_alignment.json', 'r') as f:
    data = json.load(f)

results = data['individual_results']

# Create figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Scatter plot: Questionnaire vs Behavioral classification
ax = axes[0, 0]

# Map classifications to numbers
class_map = {'hyper': 0, 'typical': 1, 'hypo': 2}
class_labels = ['Hypersensitive', 'Typical', 'Hyposensitive']

q_scores = [class_map[r['questionnaire_class']] for r in results]
b_scores = [class_map[r['behavioral_class']] for r in results]

# Add jitter
jitter = np.random.normal(0, 0.1, len(q_scores))
ax.scatter([q + jitter[i] for i, q in enumerate(q_scores)],
           [b + jitter[i] for i, b in enumerate(b_scores)],
           s=100, alpha=0.6, c=[r['confidence'] for r in results],
           cmap='RdYlGn', edgecolors='black')

# Diagonal line (perfect alignment)
ax.plot([-0.5, 2.5], [-0.5, 2.5], 'k--', alpha=0.3, label='Perfect Alignment')

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(class_labels, rotation=45)
ax.set_yticklabels(class_labels)
ax.set_xlabel('Self-Reported Classification (Questionnaire)')
ax.set_ylabel('Behavioral Classification (Actions)')
ax.set_title('Questionnaire vs Behavior Alignment')
plt.colorbar(ax.collections[0], ax=ax, label='Confidence %')
ax.grid(True, alpha=0.3)

# Add count annotations
for i in range(3):
    for j in range(3):
        count = sum(1 for r in results if class_map[r['questionnaire_class']] == i and class_map[r['behavioral_class']] == j)
        if count > 0:
            ax.text(i, j, str(count), ha='center', va='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. Bar chart: Alignment distribution
ax = axes[0, 1]

aligned = sum(1 for r in results if r['questionnaire_class'] == r['behavioral_class'])
partial = sum(1 for r in results if 'PARTIALLY' in r['alignment'])
mismatch = sum(1 for r in results if 'MISMATCHED' in r['alignment'])

bars = ax.bar(['Aligned\n(Same Class)', 'Partial\n(Adjacent)', 'Mismatched\n(Opposite)'],
              [aligned, partial, mismatch],
              color=['#4CAF50', '#FFC107', '#F44336'],
              edgecolor='black', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height)} ({height/len(results)*100:.0f}%)',
           ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Number of Participants')
ax.set_title('Self-Report to Behavior Alignment')
ax.set_ylim(0, max(aligned, partial, mismatch) * 1.2)

# 3. Volume distribution by combined classification
ax = axes[1, 0]

classes = ['hyper', 'typical', 'hypo']
colors = ['#FF6B6B', '#45B7D1', '#4ECDC4']
labels = ['Hypersensitive', 'Typical', 'Hyposensitive']

for i, (cls, color, label) in enumerate(zip(classes, colors, labels)):
    volumes = [r['settled_volume'] for r in results if r['combined_class'] == cls]
    if volumes:
        bp = ax.boxplot([volumes], positions=[i], widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        # Add individual points
        x = np.random.normal(i, 0.05, len(volumes))
        ax.scatter(x, volumes, color='black', alpha=0.6, s=30, zorder=3)
        # Add mean annotation
        ax.text(i, np.mean(volumes) + 3, f'μ={np.mean(volumes):.1f}%',
               ha='center', va='bottom', fontsize=10)

ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel('Settled Volume (%)')
ax.set_title('Volume Distribution by Combined Classification')
ax.grid(True, alpha=0.3, axis='y')

# 4. Confidence by alignment type
ax = axes[1, 1]

aligned_conf = [r['confidence'] for r in results if 'ALIGNED ✓' in r['alignment']]
partial_conf = [r['confidence'] for r in results if 'PARTIALLY' in r['alignment']]
mismatch_conf = [r['confidence'] for r in results if 'MISMATCHED' in r['alignment']]

data_to_plot = []
labels_plot = []
colors_plot = []

if aligned_conf:
    data_to_plot.append(aligned_conf)
    labels_plot.append(f'Aligned\n(n={len(aligned_conf)})')
    colors_plot.append('#4CAF50')
if partial_conf:
    data_to_plot.append(partial_conf)
    labels_plot.append(f'Partial\n(n={len(partial_conf)})')
    colors_plot.append('#FFC107')
if mismatch_conf:
    data_to_plot.append(mismatch_conf)
    labels_plot.append(f'Mismatched\n(n={len(mismatch_conf)})')
    colors_plot.append('#F44336')

bp = ax.boxplot(data_to_plot, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_plot):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels(labels_plot)
ax.set_ylabel('Classification Confidence (%)')
ax.set_title('Confidence by Alignment Type')
ax.grid(True, alpha=0.3, axis='y')

# Add means
for i, data in enumerate(data_to_plot, 1):
    ax.text(i, np.mean(data) + 2, f'{np.mean(data):.1f}%', ha='center', fontsize=9)

plt.suptitle('Neuroverse: Questionnaire vs Behavior Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/Neuroverse/fig_alignment_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('/tmp/Neuroverse/fig_alignment_analysis.pdf', bbox_inches='tight')
plt.close()

print("Alignment visualization saved!")
print("  - fig_alignment_analysis.png")
print("  - fig_alignment_analysis.pdf")
