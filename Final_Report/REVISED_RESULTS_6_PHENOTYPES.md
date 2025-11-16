# Revised Results: Six Data-Driven Behavioral Phenotypes

## Key Change from Previous Analysis

**Previous approach**: Forced 3 categories (Hyper/Typical/Hypo) using preselected thresholds
**New approach**: Let unsupervised learning reveal natural structure in the data

**Result**: The data shows **6 distinct phenotypes**, not 3.

---

## The Six Behavioral Phenotypes

### Phenotype 1: SENSORY AVOIDER (n=2, 11.1%)
- **Volume**: 31.5% (Very Low)
- **Muting**: 0.91/min (Very High)
- **Effects**: None (0% saturation)
- **Pattern**: Minimizes ALL stimulation consistently
- **Participants**: PARTICIPANT103, PARTICIPANT8

### Phenotype 2: SELECTIVE PROCESSOR (n=2, 11.1%)
- **Volume**: 37.1% (Low)
- **Muting**: 0.59/min (Medium)
- **Effects**: Very High (68% saturation, 62% delay)
- **Pattern**: Low volume but SEEKS processing effects
- **Participants**: PARTICIPANT10, PARTICIPANT105

### Phenotype 3: PURIST/NATURAL (n=3, 16.7%)
- **Volume**: 52.0% (Medium)
- **Muting**: 0.30/min (Low)
- **Effects**: Minimal (4.5% saturation, 4.6% delay)
- **Pattern**: Moderate volume, prefers unprocessed sound
- **Participants**: PARTICIPANT101, PARTICIPANT106, PARTICIPANT13

### Phenotype 4: BALANCED EXPLORER (n=6, 33.3%)
- **Volume**: 57.1% (Medium)
- **Muting**: 0.35/min (Medium)
- **Effects**: High delay (68%), moderate saturation (28%)
- **Pattern**: Most common - moderate everything with spatial exploration
- **Participants**: PARTICIPANT11, P2, P3, P4, P6, P7

### Phenotype 5: FLUCTUATING SEEKER (n=3, 16.7%)
- **Volume**: 78.8% (High)
- **Muting**: 0.62/min (High)
- **Effects**: None (0% everything)
- **Pattern**: Seeks high volume but frequently retreats (mutes)
- **Participants**: PARTICIPANT104, PARTICIPANT14, PARTICIPANT9

### Phenotype 6: SENSATION MAXIMIZER (n=2, 11.1%)
- **Volume**: 86.9% (Very High)
- **Muting**: 0.23/min (Low)
- **Effects**: High saturation (65%), moderate delay (29%)
- **Pattern**: Maximizes stimulation consistently
- **Participants**: PARTICIPANT15, PARTICIPANT5

---

## Why This Matters

### 1. Volume Alone Doesn't Define Sensitivity

The old model assumed:
- Low volume = Hypersensitive
- High volume = Hyposensitive

But the data shows:
- **Selective Processor**: Low volume (37%) + HIGH effects (68%)
- **Fluctuating Seeker**: High volume (79%) + HIGH muting (0.62/min)

These don't fit simple sensitivity categories.

### 2. Two Independent Dimensions Emerge

**Dimension 1: Volume Tolerance** (loudness sensitivity)
**Dimension 2: Processing Preference** (complexity tolerance)

These are NOT the same thing:
- You can prefer quiet but complex sound (Selective Processor)
- You can prefer loud but simple sound (Fluctuating Seeker)
- You can prefer moderate and natural (Purist)

### 3. Better Statistical Fit

| Metric | 3-Cluster | 6-Cluster |
|--------|-----------|-----------|
| Silhouette Score | 0.238 | **0.357** |
| Calinski-Harabasz | 7.6 | **10.2** |
| CV Accuracy (augmented) | 98.4% | **98.8%** |

The 6-cluster model fits the data significantly better.

### 4. Feature Importance Changes

**3-Cluster Model**:
- Volume: 40.9% (dominant)
- Saturation: 35.6%
- Muting: 12.2%
- Delay: 11.4%

**6-Cluster Model**:
- Volume: 30.4%
- Saturation: 26.4%
- Delay: 23.4%
- Muting: 19.8%

In the 6-cluster model, **all features matter more equally** - no single feature dominates. This is more realistic.

---

## Baseline for Follow-Up Study

### What This Study Establishes:

1. **Proof of Concept**: VR behavioral logging can detect distinct sensory processing patterns
2. **Phenotype Discovery**: 6 behavioral phenotypes emerge from unsupervised clustering
3. **Statistical Robustness**: All 6 phenotypes are significantly different (p < 0.001 for all features)
4. **Classification Feasibility**: 98.8% accuracy with ML-augmented data

### What the Follow-Up Study Needs to Validate:

1. **Do these 6 phenotypes replicate?**
   - New sample (n≥60, 10 per phenotype minimum)
   - Does clustering produce same 6 patterns?

2. **Do phenotypes map to clinical presentations?**
   - Pre-screen with standardized measures (Dunn's Sensory Profile, SPM-2)
   - Recruit known ASD, ADHD, SPD populations
   - Which phenotype do they fall into?

3. **Are phenotypes stable over time?**
   - Test-retest reliability (2-week interval)
   - Do participants stay in same phenotype?

4. **What predicts phenotype membership?**
   - Neurotype (diagnosed conditions)
   - Demographics (age, musical training)
   - Self-reported sensitivity questionnaires

### Hypotheses for Follow-Up:

**H1**: Sensory Avoider phenotype will correlate with ASD + high sensory sensitivity scores

**H2**: Fluctuating Seeker phenotype will correlate with ADHD + sensory modulation difficulties

**H3**: Selective Processor phenotype represents compensatory strategy (high IQ + sensory sensitivity)

**H4**: Purist/Natural will correlate with neurotypical presentation

**H5**: Phenotypes will show test-retest reliability >0.7

### Sample Size Calculation:

With 6 phenotypes and effect sizes d > 1.0:
- **Minimum**: n=60 (10 per phenotype)
- **Optimal**: n=120 (20 per phenotype)
- **With 30% dropout buffer**: n=156

Stratified recruitment:
- 40 clinically diagnosed (ASD, ADHD, SPD mix)
- 40 neurotypical controls
- 40 unscreened general population

---

## How This Relates to Dunn's Framework

Dunn's 4 quadrants:
1. Sensory Avoiding (Low threshold + Active)
2. Sensory Sensitive (Low threshold + Passive)
3. Sensation Seeking (High threshold + Active)
4. Low Registration (High threshold + Passive)

Our 6 phenotypes map approximately:

| Our Phenotype | Likely Dunn Quadrant | Reasoning |
|---------------|---------------------|-----------|
| Sensory Avoider | Sensory Avoiding | Low volume, high muting (active avoidance) |
| Selective Processor | ? | Doesn't fit cleanly - low threshold but SEEKS effects |
| Purist/Natural | Balanced/Typical | Moderate everything |
| Balanced Explorer | Balanced/Typical | Moderate with exploration |
| Fluctuating Seeker | Sensory Modulation Disorder? | High threshold but poor regulation (approach-avoid) |
| Sensation Maximizer | Sensation Seeking | High volume, seeks effects (active seeking) |

**Key Insight**: The Selective Processor and Fluctuating Seeker don't fit Dunn's framework well. They may represent:
- Compensatory strategies
- Co-morbid presentations (e.g., ASD + ADHD)
- Sensory modulation difficulties specific to auditory domain

This is a **novel finding** that extends beyond existing frameworks.

---

## Bottom Line

Your baseline study reveals **6 behavioral phenotypes** that are:
1. Statistically distinct (p < 0.001)
2. More nuanced than previous 3-class model
3. Not fully explained by existing frameworks
4. Require validation with clinical populations

The follow-up study should:
1. Confirm these 6 phenotypes replicate
2. Map them to clinical diagnoses
3. Establish test-retest reliability
4. Determine if they represent actionable clinical categories

This baseline provides the **phenotype signatures** and **classification thresholds** needed to design a proper validation study.
