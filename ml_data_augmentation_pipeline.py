#!/usr/bin/env python3
"""
ML Data Augmentation Pipeline for Neuroverse Dataset
Expands n=18 to n=1000+ using multiple ML techniques with validation.
"""

import json
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ks_2samp, wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

# Load original data
with open('/tmp/Neuroverse/questionnaire_behavior_alignment.json', 'r') as f:
    data = json.load(f)

results = data['individual_results']

print("="*80)
print("ML DATA AUGMENTATION PIPELINE")
print("Expanding Neuroverse Dataset from n=18 to n=1000+")
print("="*80)

# Prepare original feature matrix
feature_names = ['volume', 'muting', 'delay', 'saturation']
X_orig = np.array([
    [r['settled_volume'], r['mutes_per_minute'], r['final_delay'], r['final_saturation']]
    for r in results
])
y_orig = np.array([r['combined_class'] for r in results])

print(f"\nOriginal Dataset: n={len(X_orig)}")
print(f"Class Distribution: {dict(Counter(y_orig))}")
print(f"Features: {feature_names}")

# =============================================================================
# TECHNIQUE 1: SMOTE (Synthetic Minority Over-sampling Technique)
# =============================================================================
print("\n" + "="*80)
print("TECHNIQUE 1: SMOTE-BASED AUGMENTATION")
print("="*80)

def smote_augment(X, y, target_samples_per_class=200, k_neighbors=3):
    """
    SMOTE: Creates synthetic samples by interpolating between existing samples.
    Good for: Balanced class generation, preserves local structure.
    """
    X_aug = []
    y_aug = []

    for cls in np.unique(y):
        X_cls = X[y == cls]
        n_original = len(X_cls)
        n_to_generate = target_samples_per_class - n_original

        if n_to_generate <= 0:
            X_aug.extend(X_cls)
            y_aug.extend([cls] * n_original)
            continue

        # Fit k-NN on class samples
        k = min(k_neighbors, n_original - 1)
        if k < 1:
            k = 1
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(X_cls)

        # Generate synthetic samples
        for _ in range(n_to_generate):
            # Pick random sample
            idx = np.random.randint(0, n_original)
            sample = X_cls[idx]

            # Find k nearest neighbors
            distances, indices = nn.kneighbors([sample])

            # Pick random neighbor (exclude self)
            neighbor_idx = indices[0][np.random.randint(1, len(indices[0]))]
            neighbor = X_cls[neighbor_idx]

            # Interpolate
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)

            # Clip to realistic ranges
            synthetic[0] = np.clip(synthetic[0], 10, 100)  # Volume
            synthetic[1] = np.clip(synthetic[1], 0, 2)      # Muting
            synthetic[2] = np.clip(synthetic[2], 0, 100)    # Delay
            synthetic[3] = np.clip(synthetic[3], 0, 100)    # Saturation

            X_aug.append(synthetic)
            y_aug.append(cls)

        # Add original samples
        X_aug.extend(X_cls)
        y_aug.extend([cls] * n_original)

    return np.array(X_aug), np.array(y_aug)

X_smote, y_smote = smote_augment(X_orig, y_orig, target_samples_per_class=200)
print(f"SMOTE Dataset: n={len(X_smote)}")
print(f"Class Distribution: {dict(Counter(y_smote))}")

# =============================================================================
# TECHNIQUE 2: Gaussian Mixture Model (GMM) Sampling
# =============================================================================
print("\n" + "="*80)
print("TECHNIQUE 2: GAUSSIAN MIXTURE MODEL SAMPLING")
print("="*80)

def gmm_augment(X, y, target_samples_per_class=200, n_components=2):
    """
    GMM: Learns multimodal distributions within each class.
    Good for: Capturing subgroups within classes (e.g., ADHD subtypes).
    """
    X_aug = []
    y_aug = []

    for cls in np.unique(y):
        X_cls = X[y == cls]
        n_original = len(X_cls)

        # Determine number of components (can't exceed samples)
        n_comp = min(n_components, n_original - 1)
        if n_comp < 1:
            n_comp = 1

        # Fit GMM
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
        gmm.fit(X_cls)

        # Generate samples
        n_to_generate = target_samples_per_class
        synthetic, _ = gmm.sample(n_to_generate)

        # Clip to realistic ranges
        synthetic[:, 0] = np.clip(synthetic[:, 0], 10, 100)  # Volume
        synthetic[:, 1] = np.clip(synthetic[:, 1], 0, 2)      # Muting
        synthetic[:, 2] = np.clip(synthetic[:, 2], 0, 100)    # Delay
        synthetic[:, 3] = np.clip(synthetic[:, 3], 0, 100)    # Saturation

        X_aug.extend(synthetic)
        y_aug.extend([cls] * n_to_generate)

    return np.array(X_aug), np.array(y_aug)

X_gmm, y_gmm = gmm_augment(X_orig, y_orig, target_samples_per_class=200)
print(f"GMM Dataset: n={len(X_gmm)}")
print(f"Class Distribution: {dict(Counter(y_gmm))}")

# =============================================================================
# TECHNIQUE 3: Copula-Based Augmentation (Preserves Feature Correlations)
# =============================================================================
print("\n" + "="*80)
print("TECHNIQUE 3: COPULA-BASED AUGMENTATION")
print("="*80)

def copula_augment(X, y, target_samples_per_class=200):
    """
    Copula: Preserves correlation structure between features.
    Good for: Maintaining realistic feature relationships.
    """
    from scipy.stats import norm, rankdata

    X_aug = []
    y_aug = []

    for cls in np.unique(y):
        X_cls = X[y == cls]
        n_original = len(X_cls)

        if n_original < 3:
            # Not enough samples for copula, use simple Gaussian
            means = np.mean(X_cls, axis=0)
            stds = np.std(X_cls, axis=0) + 1e-6
            synthetic = np.random.normal(means, stds * 1.2, (target_samples_per_class, 4))
        else:
            # Transform to uniform marginals (empirical CDF)
            U = np.zeros_like(X_cls, dtype=float)
            for i in range(X_cls.shape[1]):
                U[:, i] = rankdata(X_cls[:, i]) / (n_original + 1)

            # Transform to normal space
            Z = norm.ppf(U)

            # Estimate correlation matrix
            corr_matrix = np.corrcoef(Z.T)

            # Handle potential numerical issues
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

            # Generate correlated normal samples
            try:
                L = np.linalg.cholesky(corr_matrix)
                Z_new = np.random.randn(target_samples_per_class, 4) @ L.T
            except np.linalg.LinAlgError:
                # Fallback to uncorrelated
                Z_new = np.random.randn(target_samples_per_class, 4)

            # Transform back through inverse marginal CDFs
            synthetic = np.zeros((target_samples_per_class, 4))
            for i in range(4):
                # Use empirical quantiles
                u_new = norm.cdf(Z_new[:, i])
                sorted_vals = np.sort(X_cls[:, i])
                indices = (u_new * (n_original - 1)).astype(int)
                indices = np.clip(indices, 0, n_original - 1)

                # Add small noise to avoid exact duplicates
                synthetic[:, i] = sorted_vals[indices] + np.random.normal(0, np.std(X_cls[:, i]) * 0.1, target_samples_per_class)

        # Clip to realistic ranges
        synthetic[:, 0] = np.clip(synthetic[:, 0], 10, 100)
        synthetic[:, 1] = np.clip(synthetic[:, 1], 0, 2)
        synthetic[:, 2] = np.clip(synthetic[:, 2], 0, 100)
        synthetic[:, 3] = np.clip(synthetic[:, 3], 0, 100)

        X_aug.extend(synthetic)
        y_aug.extend([cls] * target_samples_per_class)

    return np.array(X_aug), np.array(y_aug)

X_copula, y_copula = copula_augment(X_orig, y_orig, target_samples_per_class=200)
print(f"Copula Dataset: n={len(X_copula)}")
print(f"Class Distribution: {dict(Counter(y_copula))}")

# =============================================================================
# TECHNIQUE 4: Noise Injection with Constraints
# =============================================================================
print("\n" + "="*80)
print("TECHNIQUE 4: CONSTRAINED NOISE INJECTION")
print("="*80)

def noise_injection_augment(X, y, target_samples_per_class=200, noise_scale=0.15):
    """
    Noise Injection: Adds controlled perturbations to existing samples.
    Good for: Simple augmentation that stays close to observed data.
    """
    X_aug = []
    y_aug = []

    for cls in np.unique(y):
        X_cls = X[y == cls]
        n_original = len(X_cls)

        # Generate multiple noisy versions of each sample
        samples_per_original = target_samples_per_class // n_original + 1

        for sample in X_cls:
            for _ in range(samples_per_original):
                # Add scaled noise
                noise = np.random.randn(4) * np.abs(sample) * noise_scale
                synthetic = sample + noise

                # Clip
                synthetic[0] = np.clip(synthetic[0], 10, 100)
                synthetic[1] = np.clip(synthetic[1], 0, 2)
                synthetic[2] = np.clip(synthetic[2], 0, 100)
                synthetic[3] = np.clip(synthetic[3], 0, 100)

                X_aug.append(synthetic)
                y_aug.append(cls)

                if len([y for y in y_aug if y == cls]) >= target_samples_per_class:
                    break

            if len([y for y in y_aug if y == cls]) >= target_samples_per_class:
                break

    return np.array(X_aug), np.array(y_aug)

X_noise, y_noise = noise_injection_augment(X_orig, y_orig, target_samples_per_class=200)
print(f"Noise Injection Dataset: n={len(X_noise)}")
print(f"Class Distribution: {dict(Counter(y_noise))}")

# =============================================================================
# VALIDATION: Distribution Fidelity Metrics
# =============================================================================
print("\n" + "="*80)
print("VALIDATION: DISTRIBUTION FIDELITY")
print("="*80)

def evaluate_augmentation_quality(X_orig, X_aug, y_orig, y_aug, name):
    """Evaluate how well augmented data matches original distribution."""
    print(f"\n{name}:")
    print("-" * 60)

    metrics = {}

    for cls in np.unique(y_orig):
        X_orig_cls = X_orig[y_orig == cls]
        X_aug_cls = X_aug[y_aug == cls]

        if len(X_orig_cls) < 2 or len(X_aug_cls) < 2:
            continue

        # Kolmogorov-Smirnov test for each feature
        ks_stats = []
        for i, feat in enumerate(feature_names):
            ks_stat, p_val = ks_2samp(X_orig_cls[:, i], X_aug_cls[:, i])
            ks_stats.append(ks_stat)

        # Wasserstein distance (Earth Mover's Distance)
        wd_stats = []
        for i in range(4):
            wd = wasserstein_distance(X_orig_cls[:, i], X_aug_cls[:, i])
            wd_stats.append(wd)

        avg_ks = np.mean(ks_stats)
        avg_wd = np.mean(wd_stats)

        print(f"  {cls.upper():<10} KS-stat: {avg_ks:.3f}  Wasserstein: {avg_wd:.3f}")
        metrics[cls] = {'ks': avg_ks, 'wd': avg_wd}

    overall_ks = np.mean([m['ks'] for m in metrics.values()])
    overall_wd = np.mean([m['wd'] for m in metrics.values()])
    print(f"  {'OVERALL':<10} KS-stat: {overall_ks:.3f}  Wasserstein: {overall_wd:.3f}")

    return metrics

smote_metrics = evaluate_augmentation_quality(X_orig, X_smote, y_orig, y_smote, "SMOTE")
gmm_metrics = evaluate_augmentation_quality(X_orig, X_gmm, y_orig, y_gmm, "GMM")
copula_metrics = evaluate_augmentation_quality(X_orig, X_copula, y_orig, y_copula, "Copula")
noise_metrics = evaluate_augmentation_quality(X_orig, X_noise, y_orig, y_noise, "Noise Injection")

# =============================================================================
# ENSEMBLE: Combine Best Augmentation Methods
# =============================================================================
print("\n" + "="*80)
print("ENSEMBLE AUGMENTED DATASET")
print("="*80)

# Combine all augmentation methods
X_ensemble = np.vstack([X_orig, X_smote, X_gmm, X_copula])
y_ensemble = np.concatenate([y_orig, y_smote, y_gmm, y_copula])

print(f"Ensemble Dataset: n={len(X_ensemble)}")
print(f"Class Distribution: {dict(Counter(y_ensemble))}")
print(f"Expansion Factor: {len(X_ensemble) / len(X_orig):.1f}x")

# =============================================================================
# CLASSIFIER TRAINING ON AUGMENTED DATA
# =============================================================================
print("\n" + "="*80)
print("CLASSIFIER TRAINING ON AUGMENTED DATA")
print("="*80)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ensemble)

# Train multiple classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

cv_results = {}

for name, clf in classifiers.items():
    print(f"\n{name}:")

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y_ensemble, cv=skf, scoring='accuracy')

    print(f"  10-Fold CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"  Min: {scores.min():.3f}, Max: {scores.max():.3f}")

    cv_results[name] = {
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'min': float(scores.min()),
        'max': float(scores.max())
    }

    # Train on full augmented data
    clf.fit(X_scaled, y_ensemble)

    # Test on original data only
    X_orig_scaled = scaler.transform(X_orig)
    y_pred_orig = clf.predict(X_orig_scaled)
    orig_accuracy = np.mean(y_pred_orig == y_orig)
    print(f"  Accuracy on Original Data: {orig_accuracy:.3f}")
    cv_results[name]['orig_accuracy'] = float(orig_accuracy)

# Best model feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*80)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_scaled, y_ensemble)

for name, importance in zip(feature_names, rf.feature_importances_):
    bar = "#" * int(importance * 50)
    print(f"  {name:<12}: {importance:.3f} {bar}")

# =============================================================================
# PREDICTION CONFIDENCE & UNCERTAINTY QUANTIFICATION
# =============================================================================
print("\n" + "="*80)
print("PREDICTION CONFIDENCE & UNCERTAINTY")
print("="*80)

# Get prediction probabilities for original samples
X_orig_scaled = scaler.transform(X_orig)
proba = rf.predict_proba(X_orig_scaled)

print("\nOriginal Participant Predictions with Confidence:")
print("-" * 80)

uncertainty_results = []

for i, r in enumerate(results):
    pred_class = rf.predict([X_orig_scaled[i]])[0]
    actual_class = y_orig[i]
    confidence = np.max(proba[i]) * 100

    # Entropy as uncertainty measure
    entropy = -np.sum(proba[i] * np.log(proba[i] + 1e-10))

    match = "✓" if pred_class == actual_class else "✗"

    uncertainty_results.append({
        'id': r['id'],
        'actual': actual_class,
        'predicted': pred_class,
        'confidence': confidence,
        'entropy': entropy,
        'probabilities': {
            'hyper': float(proba[i][list(rf.classes_).index('hyper')]),
            'typical': float(proba[i][list(rf.classes_).index('typical')]),
            'hypo': float(proba[i][list(rf.classes_).index('hypo')])
        }
    })

    print(f"{r['id']:<35} Actual: {actual_class:<8} Pred: {pred_class:<8} Conf: {confidence:.1f}% {match}")

avg_confidence = np.mean([u['confidence'] for u in uncertainty_results])
avg_entropy = np.mean([u['entropy'] for u in uncertainty_results])
print(f"\nAverage Confidence: {avg_confidence:.1f}%")
print(f"Average Entropy (lower=more certain): {avg_entropy:.3f}")

# =============================================================================
# GENERATE LARGE SYNTHETIC POPULATION
# =============================================================================
print("\n" + "="*80)
print("LARGE SYNTHETIC POPULATION GENERATION")
print("="*80)

def generate_large_population(X, y, n_total=1000):
    """Generate large population maintaining learned distributions."""
    # Use GMM as it captures multimodal patterns best
    X_large = []
    y_large = []

    # Proportional to original class distribution (but balanced minimum)
    class_counts = Counter(y)
    min_per_class = n_total // len(class_counts)

    for cls in np.unique(y):
        X_cls = X[y == cls]
        n_samples = max(min_per_class, int(n_total * class_counts[cls] / len(y)))

        # Fit GMM
        n_comp = min(3, len(X_cls) - 1)
        if n_comp < 1:
            n_comp = 1

        gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
        gmm.fit(X_cls)

        synthetic, _ = gmm.sample(n_samples)

        # Clip
        synthetic[:, 0] = np.clip(synthetic[:, 0], 10, 100)
        synthetic[:, 1] = np.clip(synthetic[:, 1], 0, 2)
        synthetic[:, 2] = np.clip(synthetic[:, 2], 0, 100)
        synthetic[:, 3] = np.clip(synthetic[:, 3], 0, 100)

        X_large.extend(synthetic)
        y_large.extend([cls] * n_samples)

    return np.array(X_large), np.array(y_large)

X_large, y_large = generate_large_population(X_orig, y_orig, n_total=1000)
print(f"Large Population: n={len(X_large)}")
print(f"Class Distribution: {dict(Counter(y_large))}")

# Statistics of large population
print("\nLarge Population Statistics:")
print("-" * 60)
for cls in ['hyper', 'typical', 'hypo']:
    X_cls = X_large[y_large == cls]
    print(f"\n{cls.upper()}:")
    for i, feat in enumerate(feature_names):
        mean = np.mean(X_cls[:, i])
        std = np.std(X_cls[:, i])
        q25 = np.percentile(X_cls[:, i], 25)
        q75 = np.percentile(X_cls[:, i], 75)
        print(f"  {feat:<12}: {mean:.2f} (SD={std:.2f}, IQR={q25:.2f}-{q75:.2f})")

# =============================================================================
# SAVE ALL RESULTS
# =============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output = {
    'original_dataset': {
        'n': len(X_orig),
        'class_distribution': dict(Counter(y_orig))
    },
    'augmentation_methods': {
        'smote': {
            'n': len(X_smote),
            'distribution': dict(Counter(y_smote)),
            'quality_metrics': smote_metrics
        },
        'gmm': {
            'n': len(X_gmm),
            'distribution': dict(Counter(y_gmm)),
            'quality_metrics': gmm_metrics
        },
        'copula': {
            'n': len(X_copula),
            'distribution': dict(Counter(y_copula)),
            'quality_metrics': copula_metrics
        },
        'noise_injection': {
            'n': len(X_noise),
            'distribution': dict(Counter(y_noise)),
            'quality_metrics': noise_metrics
        }
    },
    'ensemble_dataset': {
        'n': len(X_ensemble),
        'expansion_factor': len(X_ensemble) / len(X_orig),
        'class_distribution': dict(Counter(y_ensemble))
    },
    'classifier_performance': cv_results,
    'feature_importance': dict(zip(feature_names, rf.feature_importances_.tolist())),
    'prediction_uncertainty': uncertainty_results,
    'large_population': {
        'n': len(X_large),
        'class_distribution': dict(Counter(y_large)),
        'statistics': {
            cls: {
                feat: {
                    'mean': float(np.mean(X_large[y_large == cls][:, i])),
                    'std': float(np.std(X_large[y_large == cls][:, i])),
                    'q25': float(np.percentile(X_large[y_large == cls][:, i], 25)),
                    'q75': float(np.percentile(X_large[y_large == cls][:, i], 75))
                }
                for i, feat in enumerate(feature_names)
            }
            for cls in ['hyper', 'typical', 'hypo']
        }
    }
}

with open('/tmp/Neuroverse/ml_augmented_dataset.json', 'w') as f:
    json.dump(output, f, indent=2)

# Save the actual synthetic data for further analysis
np.savez('/tmp/Neuroverse/augmented_data.npz',
         X_orig=X_orig, y_orig=y_orig,
         X_ensemble=X_ensemble, y_ensemble=y_ensemble,
         X_large=X_large, y_large=y_large,
         feature_names=feature_names)

print(f"Results saved to:")
print(f"  - /tmp/Neuroverse/ml_augmented_dataset.json")
print(f"  - /tmp/Neuroverse/augmented_data.npz")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Original Dataset: n={len(X_orig)}")
print(f"Ensemble Augmented: n={len(X_ensemble)} ({len(X_ensemble)/len(X_orig):.1f}x expansion)")
print(f"Large Population: n={len(X_large)} ({len(X_large)/len(X_orig):.1f}x expansion)")
print(f"\nBest Classifier Performance:")
print(f"  Random Forest CV Accuracy: {cv_results['Random Forest']['mean']:.1%}")
print(f"  Gradient Boosting CV Accuracy: {cv_results['Gradient Boosting']['mean']:.1%}")
print(f"\nAverage Prediction Confidence: {avg_confidence:.1f}%")
print("="*80)
