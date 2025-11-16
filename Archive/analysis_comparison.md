# Neuroverse Classification Analysis Comparison

## Summary of Different Approaches

| Method | Hyper | Typical | Hypo | Key Features Used |
|--------|-------|---------|------|-------------------|
| **V2 (Rule-Based)** | 1 (5%) | 8 (40%) | 11 (55%) | Questionnaire + static thresholds |
| **V3 (Z-Scores)** | 3 (14%) | 11 (50%) | 8 (36%) | Cohort-relative behavioral metrics |
| **Behavioral Clustering** | 2 (10%) | 8 (38%) | 11 (52%) | K-means on normalized behavior only |
| **Final Preferences** | 7 (35%) | 3 (15%) | 10 (50%) | What participants settled on |

## Key Findings

### 1. Volume Patterns (Most Reliable Indicator)
- **Hypersensitive**: Final volume 33-55%, average 41%
- **Typical**: Final volume 52-74%, average 63%
- **Hyposensitive**: Final volume 42-99%, average 69%

### 2. Unexpected Discoveries
- **Low pitch shift across all groups** (0.27-0.59x) - Everyone lowered pitch
- **Hyposensitive = low effects** (11% delay, 16% saturation) - Contradicts theory
- **High EQ adjustment count** (500-5600 per session) - Continuous logging, not discrete actions

### 3. Most Discriminating Features
1. **Final volume** - Clearest separator between groups
2. **Muting frequency** - Hyper = 0.64/min, Hypo = 0.44/min
3. **Volume change rate** - Hyper makes more frequent adjustments
4. **Pitch preference** - Typical stays closer to natural (0.59x vs 0.27x)

## Recommendations for Better Classification

### Data Collection Improvements
1. **Track discrete events only** - Not frame-by-frame logs
2. **Log user intention** - "User moved slider" vs "slider value changed"
3. **Pre-session baseline** - Self-reported sensitivity questionnaire (Dunn's SPQ)
4. **Ground truth validation** - Audiometric testing or clinical diagnosis

### Analysis Improvements
1. **Use final preferences** - What they settled on matters most
2. **Normalize by session duration** - Per-minute metrics
3. **Focus on deviations from baseline** - Not absolute values
4. **Consider temporal patterns** - Early vs late session behavior

### Missing Data That Would Help
1. **Actual dB SPL levels** - Not just % of max
2. **Which sound sources were adjusted** - Ocean vs Bug vs Music
3. **Questionnaire option meanings** - What does Option 1/2/3 represent?
4. **Participant metadata** - Age, hearing health, musical training

## Best Classification Approach

**Recommended: Final Preferences + Behavioral Clustering**

This combines:
- What they chose (final state)
- How they chose it (adjustment patterns)
- Without imposing predetermined assumptions

Results in a more balanced distribution that respects the natural clustering in the data.

## Files Generated

- `analyze_participants_v2.py` - Weighted rule-based scoring
- `analyze_participants_v3.py` - Cohort-relative z-scores
- `analyze_behavioral_only.py` - K-means on behavioral features
- `analyze_final_preferences.py` - K-means on final settled preferences
- `analyze_enhanced.py` - Multi-feature with EQ and pitch data
- `create_visualizations.py` - Matplotlib charts and graphs

All JSON results and visualizations are available for LaTeX integration.
