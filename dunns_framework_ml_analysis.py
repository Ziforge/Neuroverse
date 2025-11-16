#!/usr/bin/env python3
"""
Dunn's Sensory Processing Framework Analysis with ML Data Extension
Maps behavioral classifications to theoretical framework and extends dataset.
"""

import json
import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load our validated results
with open('/tmp/Neuroverse/questionnaire_behavior_alignment.json', 'r') as f:
    alignment_data = json.load(f)

with open('/tmp/Neuroverse/temporal_patterns_results.json', 'r') as f:
    temporal_data = json.load(f)

results = alignment_data['individual_results']

print("="*80)
print("DUNN'S SENSORY PROCESSING FRAMEWORK MAPPING")
print("Theoretical Validation and ML Data Extension")
print("="*80)

# Map to Dunn's Four Quadrants
print("\n" + "="*80)
print("SECTION 1: DUNN'S QUADRANT MAPPING")
print("="*80)

print("""
Dunn's Sensory Processing Framework (1997, 2007):

┌─────────────────────────────────────────────────────────────┐
│                    NEUROLOGICAL THRESHOLD                   │
│                Low ←──────────────→ High                    │
├─────────────┬─────────────────────┬─────────────────────────┤
│             │    LOW THRESHOLD    │    HIGH THRESHOLD       │
│   PASSIVE   │    (Sensitive)      │    (Low Registration)   │
│   RESPONSE  │                     │                         │
│             │  SENSORY SENSITIVE  │  LOW REGISTRATION       │
│             │  - Notices stimuli  │  - Misses stimuli       │
│             │  - Easily distracted│  - Needs more input     │
│             │  - Withdraws        │  - Passive seeking      │
├─────────────┼─────────────────────┼─────────────────────────┤
│             │    LOW THRESHOLD    │    HIGH THRESHOLD       │
│   ACTIVE    │    (Avoiding)       │    (Seeking)            │
│   RESPONSE  │                     │                         │
│             │  SENSORY AVOIDING   │  SENSATION SEEKING      │
│             │  - Actively avoids  │  - Actively seeks       │
│             │  - Creates routines │  - Craves stimulation   │
│             │  - Limits exposure  │  - Enjoys intensity     │
└─────────────┴─────────────────────┴─────────────────────────┘

Our Classification Mapping:
- HYPERSENSITIVE → Sensory Avoiding (Low threshold + Active response)
- TYPICAL → Balanced threshold/response
- HYPOSENSITIVE → Sensation Seeking (High threshold + Active response)
""")

# Analyze each participant's Dunn quadrant
dunn_mapping = []

for r in results:
    participant = {
        'id': r['id'],
        'behavioral_class': r['combined_class'],
        'volume': r['settled_volume'],
        'muting': r['mutes_per_minute'],
        'delay': r['final_delay'],
        'saturation': r['final_saturation']
    }

    # Determine Dunn quadrant based on behavior
    threshold_score = 0  # Negative = low threshold, Positive = high threshold
    response_score = 0   # Negative = passive, Positive = active

    # Volume indicates threshold level
    if participant['volume'] < 50:
        threshold_score -= 2  # Low threshold (sensitive)
    elif participant['volume'] > 70:
        threshold_score += 2  # High threshold (seeking)

    # Muting indicates active avoidance
    if participant['muting'] > 0.5:
        response_score += 1  # Active response
        threshold_score -= 1  # Avoiding = low threshold
    elif participant['muting'] < 0.3:
        response_score -= 1  # Passive response

    # Effects seeking indicates high threshold + active
    if participant['saturation'] > 50 or participant['delay'] > 50:
        threshold_score += 1
        response_score += 1
    elif participant['saturation'] < 30 and participant['delay'] < 30:
        threshold_score -= 1

    # Assign quadrant
    if threshold_score < -1 and response_score > 0:
        quadrant = 'Sensory Avoiding'
        dunn_class = 'Hypersensitive'
    elif threshold_score < -1 and response_score <= 0:
        quadrant = 'Sensory Sensitive'
        dunn_class = 'Hypersensitive'
    elif threshold_score > 1 and response_score > 0:
        quadrant = 'Sensation Seeking'
        dunn_class = 'Hyposensitive'
    elif threshold_score > 1 and response_score <= 0:
        quadrant = 'Low Registration'
        dunn_class = 'Hyposensitive'
    else:
        quadrant = 'Balanced/Typical'
        dunn_class = 'Typical'

    participant['dunn_quadrant'] = quadrant
    participant['dunn_class'] = dunn_class
    participant['threshold_score'] = threshold_score
    participant['response_score'] = response_score

    dunn_mapping.append(participant)

print("\nIndividual Dunn Quadrant Assignments:")
print("-" * 80)
for p in dunn_mapping:
    match = "✓" if p['dunn_class'].lower() == p['behavioral_class'] else "~"
    print(f"{p['id']:<35} Behavioral: {p['behavioral_class']:<10} Dunn: {p['dunn_quadrant']:<20} {match}")

# Summary
from collections import Counter
quadrant_counts = Counter([p['dunn_quadrant'] for p in dunn_mapping])
print(f"\nDunn Quadrant Distribution:")
for q, count in quadrant_counts.most_common():
    print(f"  {q}: {count} ({count/len(dunn_mapping)*100:.1f}%)")

# Alignment check
alignment_count = sum(1 for p in dunn_mapping if p['dunn_class'].lower() == p['behavioral_class'])
print(f"\nDunn-Behavioral Alignment: {alignment_count}/{len(dunn_mapping)} ({alignment_count/len(dunn_mapping)*100:.1f}%)")

# ML Data Extension
print("\n" + "="*80)
print("SECTION 2: ML DATA EXTENSION")
print("="*80)

# Prepare feature matrix
feature_names = ['volume', 'muting', 'delay', 'saturation']
X = np.array([[p['volume'], p['muting'], p['delay'], p['saturation']] for p in dunn_mapping])
y = np.array([p['behavioral_class'] for p in dunn_mapping])

print(f"\nOriginal Dataset: {len(X)} samples")
print(f"Class distribution: {dict(Counter(y))}")

# 1. Bootstrap resampling to estimate confidence intervals
print("\n1. BOOTSTRAP RESAMPLING (1000 iterations)")
print("-" * 40)

n_bootstrap = 1000
bootstrap_results = {
    'hyper_volume': [],
    'typical_volume': [],
    'hypo_volume': [],
    'hyper_muting': [],
    'typical_muting': [],
    'hypo_muting': []
}

for i in range(n_bootstrap):
    # Resample with replacement
    X_boot, y_boot = resample(X, y, random_state=i)

    # Calculate means for each class
    for cls in ['hyper', 'typical', 'hypo']:
        mask = y_boot == cls
        if np.sum(mask) > 0:
            bootstrap_results[f'{cls}_volume'].append(np.mean(X_boot[mask, 0]))
            bootstrap_results[f'{cls}_muting'].append(np.mean(X_boot[mask, 1]))

print("Volume Confidence Intervals (95%):")
for cls in ['hyper', 'typical', 'hypo']:
    vols = bootstrap_results[f'{cls}_volume']
    lower = np.percentile(vols, 2.5)
    upper = np.percentile(vols, 97.5)
    mean = np.mean(vols)
    print(f"  {cls.capitalize()}: {mean:.1f}% [{lower:.1f}% - {upper:.1f}%]")

print("\nMuting Rate Confidence Intervals (95%):")
for cls in ['hyper', 'typical', 'hypo']:
    mutes = bootstrap_results[f'{cls}_muting']
    lower = np.percentile(mutes, 2.5)
    upper = np.percentile(mutes, 97.5)
    mean = np.mean(mutes)
    print(f"  {cls.capitalize()}: {mean:.2f}/min [{lower:.2f} - {upper:.2f}]")

# 2. Synthetic data generation using learned distributions
print("\n2. SYNTHETIC DATA GENERATION")
print("-" * 40)

def generate_synthetic_samples(X, y, n_samples_per_class=50):
    """Generate synthetic samples based on learned class distributions."""
    synthetic_X = []
    synthetic_y = []

    for cls in np.unique(y):
        mask = y == cls
        X_cls = X[mask]

        # Learn distribution parameters
        means = np.mean(X_cls, axis=0)
        stds = np.std(X_cls, axis=0)

        # Add some noise to prevent overfitting
        for i in range(n_samples_per_class):
            # Sample from Gaussian with learned parameters
            sample = np.random.normal(means, stds * 1.2)  # Slightly wider std

            # Clip to realistic ranges
            sample[0] = np.clip(sample[0], 10, 100)  # Volume 10-100%
            sample[1] = np.clip(sample[1], 0, 2)      # Muting 0-2/min
            sample[2] = np.clip(sample[2], 0, 100)    # Delay 0-100%
            sample[3] = np.clip(sample[3], 0, 100)    # Saturation 0-100%

            synthetic_X.append(sample)
            synthetic_y.append(cls)

    return np.array(synthetic_X), np.array(synthetic_y)

X_synth, y_synth = generate_synthetic_samples(X, y, n_samples_per_class=50)
print(f"Generated {len(X_synth)} synthetic samples")
print(f"Synthetic class distribution: {dict(Counter(y_synth))}")

# Combine original and synthetic
X_extended = np.vstack([X, X_synth])
y_extended = np.concatenate([y, y_synth])
print(f"Extended dataset: {len(X_extended)} samples")

# 3. Train classifier on extended data
print("\n3. CLASSIFICATION MODEL TRAINING")
print("-" * 40)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_extended)

# Random Forest with cross-validation
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
cv_scores = cross_val_score(rf, X_scaled, y_extended, cv=5)
print(f"Random Forest CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Train on full extended data
rf.fit(X_scaled, y_extended)

# Feature importance
print("\nFeature Importance:")
for name, importance in zip(feature_names, rf.feature_importances_):
    print(f"  {name:<15}: {importance:.3f}")

# 4. Leave-One-Out validation on original data only
print("\n4. LEAVE-ONE-OUT CROSS-VALIDATION (Original Data)")
print("-" * 40)

X_orig_scaled = scaler.transform(X)
loo = LeaveOneOut()
loo_predictions = []
loo_actuals = []

for train_idx, test_idx in loo.split(X_orig_scaled):
    X_train = np.vstack([X_orig_scaled[train_idx], scaler.transform(X_synth)])
    y_train = np.concatenate([y[train_idx], y_synth])

    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_orig_scaled[test_idx])
    loo_predictions.append(pred[0])
    loo_actuals.append(y[test_idx][0])

loo_accuracy = sum(p == a for p, a in zip(loo_predictions, loo_actuals)) / len(loo_actuals)
print(f"LOO Accuracy: {loo_accuracy:.3f}")

# Show misclassifications
print("\nLOO Predictions:")
for i, (pred, actual) in enumerate(zip(loo_predictions, loo_actuals)):
    match = "✓" if pred == actual else "✗"
    print(f"  {dunn_mapping[i]['id']:<35}: Predicted={pred:<10} Actual={actual:<10} {match}")

# 5. Prediction intervals for new participants
print("\n5. CLASSIFICATION BOUNDARIES")
print("-" * 40)

# Use decision function to show boundaries
print("Typical ranges for classification:")
for cls in ['hyper', 'typical', 'hypo']:
    mask = y_extended == cls
    X_cls = X_extended[mask]

    print(f"\n{cls.upper()}:")
    for i, name in enumerate(feature_names):
        q25 = np.percentile(X_cls[:, i], 25)
        q75 = np.percentile(X_cls[:, i], 75)
        median = np.median(X_cls[:, i])
        print(f"  {name:<15}: {median:.1f} (IQR: {q25:.1f} - {q75:.1f})")

# Second test protocol
print("\n" + "="*80)
print("SECTION 3: SECOND TEST PROTOCOL (Validation Study)")
print("="*80)

print("""
PROPOSED VALIDATION STUDY DESIGN:

1. PRE-TEST QUESTIONNAIRE
   - Dunn's Adult Sensory Profile (Standardized 60-item)
   - Self-reported neurodivergence (ASD, ADHD, SPD diagnosis)
   - Auditory sensitivity screening (ASQ - Auditory Sensitivity Quotient)

2. CALIBRATION PHASE (5 minutes)
   - Present reference sounds at fixed dB SPL levels
   - Record "comfortable" vs "uncomfortable" thresholds
   - Establish baseline sensitivity

3. VR INTERACTION PHASE (15 minutes)
   - Same protocol as current study
   - Track: volume, muting, effects, EQ preferences
   - Add: pupil dilation (via eye tracking) as physiological marker

4. POST-TEST VALIDATION
   - Compare Dunn's pre-test to behavioral classification
   - Calculate sensitivity/specificity of behavioral markers
   - Validate against clinical diagnoses

5. LONGITUDINAL FOLLOW-UP (Optional)
   - Re-test same participants after 2 weeks
   - Assess test-retest reliability
   - Track adaptation/learning effects

HYPOTHESES TO TEST:
H1: Participants scoring high on Dunn's "Sensory Avoiding" will show
    hypersensitive behavioral patterns (volume < 50%, muting > 0.5/min)

H2: Participants with ADHD diagnosis will show either:
    a) Hyposensitive pattern (seeking) OR
    b) Hypersensitive pattern (avoidant subtype)

H3: Behavioral classification will predict Dunn's quadrant with >70% accuracy

SAMPLE SIZE CALCULATION:
Current: n=18
Power analysis (α=0.05, power=0.80, effect size=0.6):
Required: n=45 per group (135 total) for 3-way classification

MINIMUM VIABLE: n=30 total (10 per expected classification)
""")

# Save results
output = {
    'dunn_mapping': [
        {
            'id': p['id'],
            'behavioral_class': p['behavioral_class'],
            'dunn_quadrant': p['dunn_quadrant'],
            'threshold_score': p['threshold_score'],
            'response_score': p['response_score']
        }
        for p in dunn_mapping
    ],
    'bootstrap_ci': {
        'volume': {
            'hyper': {
                'mean': float(np.mean(bootstrap_results['hyper_volume'])),
                'ci_lower': float(np.percentile(bootstrap_results['hyper_volume'], 2.5)),
                'ci_upper': float(np.percentile(bootstrap_results['hyper_volume'], 97.5))
            },
            'typical': {
                'mean': float(np.mean(bootstrap_results['typical_volume'])),
                'ci_lower': float(np.percentile(bootstrap_results['typical_volume'], 2.5)),
                'ci_upper': float(np.percentile(bootstrap_results['typical_volume'], 97.5))
            },
            'hypo': {
                'mean': float(np.mean(bootstrap_results['hypo_volume'])),
                'ci_lower': float(np.percentile(bootstrap_results['hypo_volume'], 2.5)),
                'ci_upper': float(np.percentile(bootstrap_results['hypo_volume'], 97.5))
            }
        }
    },
    'ml_metrics': {
        'original_n': len(X),
        'extended_n': len(X_extended),
        'cv_accuracy': float(cv_scores.mean()),
        'loo_accuracy': float(loo_accuracy),
        'feature_importance': dict(zip(feature_names, rf.feature_importances_.tolist()))
    }
}

with open('/tmp/Neuroverse/dunns_framework_ml_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to /tmp/Neuroverse/dunns_framework_ml_results.json")
