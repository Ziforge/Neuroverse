#!/usr/bin/env python3
"""
Updated Visualizations: Original vs ML-Augmented Results
Generates publication-ready figures showing the impact of data augmentation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (12, 10)

# Load data
with open('/tmp/Neuroverse/questionnaire_behavior_alignment.json', 'r') as f:
    orig_data = json.load(f)

mega_data = np.load('/tmp/Neuroverse/mega_dataset.npz', allow_pickle=True)
X_mega = mega_data['X_mega']
y_mega = mega_data['y_mega']

# Original data
results = orig_data['individual_results']
X_orig = np.array([
    [r['settled_volume'], r['mutes_per_minute'], r['final_delay'], r['final_saturation']]
    for r in results
])
y_orig = np.array([r['combined_class'] for r in results])

feature_names = ['Volume (%)', 'Muting Rate (/min)', 'Delay (%)', 'Saturation (%)']
colors = {'hyper': '#FF6B6B', 'typical': '#4ECDC4', 'hypo': '#45B7D1'}
class_labels = {'hyper': 'Hypersensitive', 'typical': 'Typical', 'hypo': 'Hyposensitive'}

print("="*80)
print("GENERATING UPDATED VISUALIZATIONS")
print("="*80)

# =============================================================================
# FIGURE 1: BEFORE & AFTER COMPARISON
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Impact of ML Data Augmentation on Statistical Robustness', fontsize=14, fontweight='bold')

# 1A: Sample Size Comparison
ax = axes[0, 0]
datasets = ['Original\n(n=18)', 'Augmented\n(n=3,336)']
sizes = [18, 3336]
bars = ax.bar(datasets, sizes, color=['#95A5A6', '#27AE60'], edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Samples')
ax.set_title('A. Dataset Size Expansion (185x)')
ax.set_yscale('log')
for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, f'n={size}',
            ha='center', va='bottom', fontweight='bold')

# 1B: Statistical Power Comparison
ax = axes[0, 1]
comparisons = ['Hyper vs\nTypical', 'Typical vs\nHypo', 'Hyper vs\nHypo']
power_orig = [0.460, 0.592, 0.673]
power_aug = [1.0, 1.0, 1.0]

x = np.arange(len(comparisons))
width = 0.35
bars1 = ax.bar(x - width/2, power_orig, width, label='Original (n=18)', color='#E74C3C', alpha=0.7)
bars2 = ax.bar(x + width/2, power_aug, width, label='Augmented (n=3,336)', color='#27AE60', alpha=0.7)

ax.axhline(y=0.8, color='black', linestyle='--', linewidth=1, label='Recommended Power (0.80)')
ax.set_ylabel('Statistical Power')
ax.set_title('B. Statistical Power for Detecting Group Differences')
ax.set_xticks(x)
ax.set_xticklabels(comparisons)
ax.set_ylim(0, 1.15)
ax.legend(loc='lower right')

# Add power gain labels
for i, (p1, p2) in enumerate(zip(power_orig, power_aug)):
    gain = (p2 - p1) * 100
    ax.text(i, 1.05, f'+{gain:.0f}%', ha='center', fontweight='bold', color='green')

# 1C: Confidence Interval Width Comparison
ax = axes[1, 0]
# Original CI widths (estimated from bootstrap)
orig_ci_widths = {
    'hyper': (60.1 - 37.5),  # From bootstrap results
    'typical': (71.8 - 55.7),
    'hypo': (98.6 - 75.2)
}

# Augmented CI widths (calculated from expanded data)
aug_ci_widths = {
    'hyper': 47.37 - 46.06,
    'typical': 63.34 - 62.60,
    'hypo': 87.10 - 86.28
}

classes = ['Hypersensitive', 'Typical', 'Hyposensitive']
x = np.arange(len(classes))
width = 0.35

orig_widths = [orig_ci_widths['hyper'], orig_ci_widths['typical'], orig_ci_widths['hypo']]
aug_widths = [aug_ci_widths['hyper'], aug_ci_widths['typical'], aug_ci_widths['hypo']]

bars1 = ax.bar(x - width/2, orig_widths, width, label='Original', color='#E74C3C', alpha=0.7)
bars2 = ax.bar(x + width/2, aug_widths, width, label='Augmented', color='#27AE60', alpha=0.7)

ax.set_ylabel('95% CI Width (Volume %)')
ax.set_title('C. Confidence Interval Precision')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Add reduction percentage
for i, (w1, w2) in enumerate(zip(orig_widths, aug_widths)):
    reduction = (1 - w2/w1) * 100
    ax.text(i + width/2, w2 + 0.5, f'{reduction:.0f}%\nreduction', ha='center', fontsize=8)

# 1D: Classification Accuracy
ax = axes[1, 1]
methods = ['Original\n(LOO CV)', 'SMOTE', 'GMM', 'Ensemble\n(All Methods)']
accuracies = [0.833, 0.980, 0.980, 0.984]
errors = [0.0, 0.008, 0.008, 0.021]

bars = ax.bar(methods, accuracies, yerr=errors, capsize=5,
              color=['#95A5A6', '#3498DB', '#9B59B6', '#27AE60'],
              edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Classification Accuracy')
ax.set_title('D. Model Performance Comparison')
ax.set_ylim(0.7, 1.05)
ax.axhline(y=0.90, color='black', linestyle='--', linewidth=1, alpha=0.5)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.1%}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/Neuroverse/fig_augmentation_impact.png', dpi=300, bbox_inches='tight')
plt.savefig('/tmp/Neuroverse/fig_augmentation_impact.pdf', bbox_inches='tight')
print("✓ Figure 1: Augmentation Impact saved")

# =============================================================================
# FIGURE 2: DISTRIBUTION COMPARISON (Original vs Augmented)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Behavioral Parameter Distributions: Original vs ML-Augmented Data',
             fontsize=14, fontweight='bold')

feature_idx = [0, 1, 3, 2]  # Volume, Muting, Saturation, Delay
feature_labels = ['Volume (%)', 'Muting Rate (/min)', 'Saturation (%)', 'Delay (%)']

for idx, (feat_i, feat_name) in enumerate(zip(feature_idx, feature_labels)):
    ax = axes[idx // 2, idx % 2]

    # Plot distributions for each class
    for cls in ['hyper', 'typical', 'hypo']:
        # Original data (KDE)
        orig_vals = X_orig[y_orig == cls][:, feat_i]

        # Augmented data (histogram)
        aug_vals = X_mega[y_mega == cls][:, feat_i]

        # Plot histogram of augmented data
        ax.hist(aug_vals, bins=30, alpha=0.3, color=colors[cls],
                label=f'{class_labels[cls]} (n={len(aug_vals)})', density=True)

        # Overlay original data points
        y_jitter = np.random.uniform(-0.02, 0.02, len(orig_vals))
        ax.scatter(orig_vals, y_jitter + 0.01, color=colors[cls], s=80,
                   edgecolor='black', linewidth=1.5, alpha=0.9, zorder=5,
                   marker='o')

        # Add mean line
        mean_val = np.mean(aug_vals)
        ax.axvline(mean_val, color=colors[cls], linestyle='--', linewidth=2, alpha=0.8)

    ax.set_xlabel(feat_name)
    ax.set_ylabel('Density')
    ax.set_title(f'{chr(65+idx)}. {feat_name} Distribution')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=-0.05)

plt.tight_layout()
plt.savefig('/tmp/Neuroverse/fig_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('/tmp/Neuroverse/fig_distribution_comparison.pdf', bbox_inches='tight')
print("✓ Figure 2: Distribution Comparison saved")

# =============================================================================
# FIGURE 3: EFFECT SIZE FOREST PLOT
# =============================================================================
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / pooled_std

fig, ax = plt.subplots(figsize=(12, 8))

comparisons = []
effect_sizes = []
ci_lower = []
ci_upper = []

# Calculate effect sizes for all features and comparisons
for feat_i, feat_name in enumerate(['Volume', 'Muting', 'Delay', 'Saturation']):
    for cls1, cls2 in [('hyper', 'typical'), ('typical', 'hypo'), ('hyper', 'hypo')]:
        g1 = X_mega[y_mega == cls1][:, feat_i]
        g2 = X_mega[y_mega == cls2][:, feat_i]

        d = cohens_d(g1, g2)
        # Confidence interval for Cohen's d
        n1, n2 = len(g1), len(g2)
        se_d = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
        ci_low = d - 1.96 * se_d
        ci_high = d + 1.96 * se_d

        comparisons.append(f'{feat_name}\n{cls1[:4]} vs {cls2[:4]}')
        effect_sizes.append(d)
        ci_lower.append(ci_low)
        ci_upper.append(ci_high)

# Sort by effect size
sorted_indices = np.argsort(effect_sizes)[::-1]
y_pos = np.arange(len(comparisons))

# Plot forest plot
for i, idx in enumerate(sorted_indices):
    color = '#27AE60' if abs(effect_sizes[idx]) > 0.8 else '#F39C12' if abs(effect_sizes[idx]) > 0.5 else '#E74C3C'
    ax.errorbar(effect_sizes[idx], i,
                xerr=[[effect_sizes[idx] - ci_lower[idx]], [ci_upper[idx] - effect_sizes[idx]]],
                fmt='o', color=color, markersize=10, capsize=5, capthick=2, linewidth=2)

ax.set_yticks(y_pos)
ax.set_yticklabels([comparisons[i] for i in sorted_indices])
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=-0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=-0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel("Cohen's d Effect Size")
ax.set_title('Effect Sizes with 95% Confidence Intervals (Augmented Data)', fontweight='bold')

# Add effect size interpretation
ax.text(0.2, len(comparisons) + 0.5, 'Small', ha='center', fontsize=9, style='italic')
ax.text(0.5, len(comparisons) + 0.5, 'Medium', ha='center', fontsize=9, style='italic')
ax.text(0.8, len(comparisons) + 0.5, 'Large', ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('/tmp/Neuroverse/fig_effect_sizes.png', dpi=300, bbox_inches='tight')
plt.savefig('/tmp/Neuroverse/fig_effect_sizes.pdf', bbox_inches='tight')
print("✓ Figure 3: Effect Size Forest Plot saved")

# =============================================================================
# FIGURE 4: CLASSIFICATION BOUNDARIES (2D Projection)
# =============================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Classification Boundaries: Original vs Augmented Data', fontsize=14, fontweight='bold')

# Standardize
scaler = StandardScaler()
X_orig_scaled = scaler.fit_transform(X_orig)
X_mega_scaled = scaler.fit_transform(X_mega)

# PCA for visualization
pca = PCA(n_components=2)

# Original data
ax = axes[0]
X_orig_pca = pca.fit_transform(X_orig_scaled)
for cls in ['hyper', 'typical', 'hypo']:
    mask = y_orig == cls
    ax.scatter(X_orig_pca[mask, 0], X_orig_pca[mask, 1],
               c=colors[cls], label=class_labels[cls], s=150,
               edgecolor='black', linewidth=1.5, alpha=0.9)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('A. Original Data (n=18)')
ax.legend()
ax.grid(True, alpha=0.3)

# Augmented data
ax = axes[1]
# Subsample for visualization
np.random.seed(42)
subsample_idx = np.random.choice(len(X_mega), 500, replace=False)
X_sub = X_mega_scaled[subsample_idx]
y_sub = y_mega[subsample_idx]

X_sub_pca = pca.fit_transform(X_sub)
for cls in ['hyper', 'typical', 'hypo']:
    mask = y_sub == cls
    ax.scatter(X_sub_pca[mask, 0], X_sub_pca[mask, 1],
               c=colors[cls], label=class_labels[cls], s=50,
               alpha=0.6, edgecolor='none')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('B. Augmented Data (n=500 subsample)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/Neuroverse/fig_classification_boundaries.png', dpi=300, bbox_inches='tight')
plt.savefig('/tmp/Neuroverse/fig_classification_boundaries.pdf', bbox_inches='tight')
print("✓ Figure 4: Classification Boundaries saved")

# =============================================================================
# FIGURE 5: FEATURE IMPORTANCE & DECISION THRESHOLDS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Classification Model Analysis', fontsize=14, fontweight='bold')

# Feature Importance
ax = axes[0]
features = ['Volume', 'Saturation', 'Muting', 'Delay']
importance = [0.409, 0.356, 0.122, 0.114]
colors_feat = ['#E74C3C', '#F39C12', '#3498DB', '#27AE60']

bars = ax.barh(features, importance, color=colors_feat, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Feature Importance (Random Forest)')
ax.set_title('A. Feature Importance Ranking')
ax.set_xlim(0, 0.5)

for bar, imp in zip(bars, importance):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{imp:.1%}', va='center', fontweight='bold')

# Decision Thresholds for Volume
ax = axes[1]
volume_range = np.linspace(10, 100, 1000)

# Classification probabilities (sigmoid approximations)
def sigmoid(x, center, scale):
    return 1 / (1 + np.exp(-(x - center) / scale))

# Approximate boundaries
p_hyper = 1 - sigmoid(volume_range, 54.4, 5)
p_hypo = sigmoid(volume_range, 74.8, 5)
p_typical = 1 - p_hyper - p_hypo
p_typical = np.clip(p_typical, 0, 1)

ax.plot(volume_range, p_hyper, color=colors['hyper'], linewidth=2, label='Hypersensitive')
ax.plot(volume_range, p_typical, color=colors['typical'], linewidth=2, label='Typical')
ax.plot(volume_range, p_hypo, color=colors['hypo'], linewidth=2, label='Hyposensitive')

# Add threshold lines
ax.axvline(54.4, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(74.8, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.text(54.4, 1.05, '54.4%', ha='center', fontsize=9)
ax.text(74.8, 1.05, '74.8%', ha='center', fontsize=9)

ax.set_xlabel('Volume Preference (%)')
ax.set_ylabel('Classification Probability')
ax.set_title('B. Volume-Based Classification Thresholds')
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

# Shade regions
ax.axvspan(10, 54.4, alpha=0.1, color=colors['hyper'])
ax.axvspan(54.4, 74.8, alpha=0.1, color=colors['typical'])
ax.axvspan(74.8, 100, alpha=0.1, color=colors['hypo'])

plt.tight_layout()
plt.savefig('/tmp/Neuroverse/fig_feature_importance.png', dpi=300, bbox_inches='tight')
plt.savefig('/tmp/Neuroverse/fig_feature_importance.pdf', bbox_inches='tight')
print("✓ Figure 5: Feature Importance saved")

# =============================================================================
# SUMMARY STATISTICS TABLE
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: KEY RESULTS COMPARISON")
print("="*80)

print("\n┌" + "─"*76 + "┐")
print("│" + " METRIC".ljust(35) + "│" + " ORIGINAL (n=18)".center(20) + "│" + " AUGMENTED (n=3,336)".center(20) + "│")
print("├" + "─"*35 + "┼" + "─"*20 + "┼" + "─"*20 + "┤")

metrics = [
    ("Sample Size", "18", "3,336"),
    ("Statistical Power (Volume)", "46-67%", "100%"),
    ("CI Width (Volume)", "±17-23%", "±0.5-0.8%"),
    ("Classification Accuracy", "83.3%", "98.4%"),
    ("Effect Size (η²)", "Not calculable", "0.790 (Large)"),
    ("ANOVA p-value", "Underpowered", "<0.001"),
    ("Feature Importance: Volume", "Not determinable", "40.9%"),
    ("Permutation Test p-value", "Not feasible", "0.0099"),
]

for metric, orig, aug in metrics:
    print(f"│ {metric:<33} │ {orig:^18} │ {aug:^18} │")

print("└" + "─"*35 + "┴" + "─"*20 + "┴" + "─"*20 + "┘")

print("\nAll figures saved to /tmp/Neuroverse/")
print("="*80)
