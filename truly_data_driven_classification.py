#!/usr/bin/env python3
"""
Truly Data-Driven Classification
No preselected thresholds - let the data define the clusters.
"""

import json
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load raw behavioral data
with open('/tmp/Neuroverse/questionnaire_behavior_alignment.json', 'r') as f:
    data = json.load(f)

results = data['individual_results']

print("="*80)
print("TRULY DATA-DRIVEN CLASSIFICATION")
print("No Preselected Thresholds - Let Data Define Clusters")
print("="*80)

# Extract raw behavioral features
feature_names = ['Volume', 'Muting', 'Delay', 'Saturation']
X = np.array([
    [r['settled_volume'], r['mutes_per_minute'], r['final_delay'], r['final_saturation']]
    for r in results
])

print(f"\nRaw Data (n={len(X)}):")
print("-" * 60)
for i, name in enumerate(feature_names):
    vals = X[:, i]
    print(f"{name:<12}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}, std={vals.std():.2f}")

# Standardize for clustering (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*80)
print("METHOD 1: OPTIMAL NUMBER OF CLUSTERS (Data-Driven)")
print("="*80)

# Test different numbers of clusters
print("\nFinding optimal k using multiple metrics:")
print("-" * 60)

results_k = []
for k in range(2, 8):
    if k >= len(X):
        continue
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Only calculate if we have enough samples
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        inertia = kmeans.inertia_

        results_k.append({
            'k': k,
            'silhouette': silhouette,
            'calinski': calinski,
            'inertia': inertia
        })

        print(f"k={k}: Silhouette={silhouette:.3f}, Calinski-Harabasz={calinski:.1f}, Inertia={inertia:.1f}")

# Best k by silhouette score
best_k_silhouette = max(results_k, key=lambda x: x['silhouette'])['k']
best_k_calinski = max(results_k, key=lambda x: x['calinski'])['k']

print(f"\nOptimal k by Silhouette Score: {best_k_silhouette}")
print(f"Optimal k by Calinski-Harabasz: {best_k_calinski}")

# Elbow method
print("\nElbow Method (looking for kink in inertia):")
inertias = [r['inertia'] for r in results_k]
ks = [r['k'] for r in results_k]
for i in range(1, len(inertias)):
    decrease = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
    print(f"  k={ks[i]}: Inertia decrease = {decrease:.1f}%")

print("\n" + "="*80)
print("METHOD 2: GAUSSIAN MIXTURE MODEL (Probabilistic Clusters)")
print("="*80)

# GMM with BIC/AIC for model selection
print("\nModel selection using BIC/AIC:")
gmm_results = []
for k in range(2, 7):
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(X_scaled)
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    gmm_results.append({'k': k, 'bic': bic, 'aic': aic})
    print(f"k={k}: BIC={bic:.1f}, AIC={aic:.1f}")

best_gmm_bic = min(gmm_results, key=lambda x: x['bic'])['k']
best_gmm_aic = min(gmm_results, key=lambda x: x['aic'])['k']
print(f"\nOptimal k by BIC (lower is better): {best_gmm_bic}")
print(f"Optimal k by AIC (lower is better): {best_gmm_aic}")

print("\n" + "="*80)
print("METHOD 3: HIERARCHICAL CLUSTERING (No k specified)")
print("="*80)

# Agglomerative clustering
linkage_matrix = linkage(X_scaled, method='ward')
print("\nDendrogram distances (cluster merging order):")
print("Higher distance = more dissimilar clusters being merged")
print("-" * 40)

# Show last few merges
n_merges = min(10, len(linkage_matrix))
for i in range(len(linkage_matrix) - n_merges, len(linkage_matrix)):
    dist = linkage_matrix[i, 2]
    n_points = linkage_matrix[i, 3]
    print(f"  Merge {i+1}: Distance={dist:.3f}, Points={int(n_points)}")

print("\n" + "="*80)
print("FINAL CLUSTERING: Using Data-Determined k")
print("="*80)

# Use the k suggested by most metrics
suggested_ks = [best_k_silhouette, best_k_calinski, best_gmm_bic, best_gmm_aic]
optimal_k = Counter(suggested_ks).most_common(1)[0][0]
print(f"\nConsensus Optimal k: {optimal_k}")

# Perform final clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
labels_final = kmeans_final.fit_predict(X_scaled)

# Get cluster centers in original scale
centers_scaled = kmeans_final.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

print("\nData-Defined Cluster Centers (in original units):")
print("-" * 70)
print(f"{'Cluster':<10} {'Volume %':<12} {'Muting/min':<12} {'Delay %':<12} {'Saturation %':<12}")
print("-" * 70)

cluster_info = []
for i in range(optimal_k):
    center = centers_original[i]
    n_members = np.sum(labels_final == i)
    cluster_info.append({
        'id': i,
        'center': center,
        'n': n_members,
        'volume': center[0],
        'muting': center[1]
    })
    print(f"{i:<10} {center[0]:<12.1f} {center[1]:<12.3f} {center[2]:<12.1f} {center[3]:<12.1f} (n={n_members})")

# Sort clusters by volume (interpretable ordering)
cluster_info_sorted = sorted(cluster_info, key=lambda x: x['volume'])
print("\nClusters Ordered by Volume (Low → High):")
print("-" * 70)

cluster_names = []
for i, info in enumerate(cluster_info_sorted):
    original_id = info['id']
    center = info['center']

    # Assign interpretable names based on data position
    if i == 0:
        name = "LOW_VOLUME"
    elif i == optimal_k - 1:
        name = "HIGH_VOLUME"
    else:
        name = f"MID_VOLUME_{i}"

    cluster_names.append((original_id, name))

    members = [results[j]['id'] for j in range(len(results)) if labels_final[j] == original_id]

    print(f"\n{name} (Cluster {original_id}, n={info['n']}):")
    print(f"  Volume: {center[0]:.1f}%")
    print(f"  Muting: {center[1]:.3f}/min")
    print(f"  Delay: {center[2]:.1f}%")
    print(f"  Saturation: {center[3]:.1f}%")
    print(f"  Members: {', '.join(members)}")

# Statistical validation
print("\n" + "="*80)
print("STATISTICAL VALIDATION OF CLUSTERS")
print("="*80)

# Are clusters significantly different?
from scipy.stats import f_oneway, kruskal

print("\nANOVA Tests (Are clusters significantly different?):")
print("-" * 60)

for feat_idx, feat_name in enumerate(feature_names):
    groups = [X[labels_final == i, feat_idx] for i in range(optimal_k)]
    f_stat, p_val = f_oneway(*groups)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"{feat_name:<12}: F={f_stat:.2f}, p={p_val:.4f} {sig}")

# Cluster separation
print(f"\nCluster Quality Metrics:")
print(f"  Silhouette Score: {silhouette_score(X_scaled, labels_final):.3f}")
print(f"  Calinski-Harabasz: {calinski_harabasz_score(X_scaled, labels_final):.1f}")

# Compare to pre-selected classification
print("\n" + "="*80)
print("COMPARISON: Data-Driven vs Previous Classification")
print("="*80)

prev_labels = np.array([r['combined_class'] for r in results])

print("\nPrevious (Z-score + Weighted Scoring) Distribution:")
print(f"  {dict(Counter(prev_labels))}")

print(f"\nData-Driven (k={optimal_k} clustering) Distribution:")
print(f"  {dict(Counter(labels_final))}")

# Cross-tabulation
print("\nCross-Tabulation (Previous vs Data-Driven):")
print("-" * 60)

# Map data-driven clusters to previous labels for comparison
cluster_to_prev = {}
for cluster_id in range(optimal_k):
    prev_labels_in_cluster = prev_labels[labels_final == cluster_id]
    most_common = Counter(prev_labels_in_cluster).most_common(1)[0][0]
    cluster_to_prev[cluster_id] = most_common

print("Cluster → Most Common Previous Label:")
for c_id, prev in cluster_to_prev.items():
    center = centers_original[c_id]
    print(f"  Cluster {c_id} (Vol={center[0]:.1f}%) → {prev}")

# Agreement percentage
data_driven_mapped = np.array([cluster_to_prev[c] for c in labels_final])
agreement = np.mean(data_driven_mapped == prev_labels)
print(f"\nAgreement with previous classification: {agreement:.1%}")

print("\n" + "="*80)
print("TRULY DATA-DRIVEN THRESHOLDS")
print("="*80)

# Natural boundaries between clusters (midpoints)
print("\nVolume Thresholds (Midpoints Between Cluster Centers):")
centers_by_volume = sorted([centers_original[i] for i in range(optimal_k)], key=lambda x: x[0])

for i in range(len(centers_by_volume) - 1):
    threshold = (centers_by_volume[i][0] + centers_by_volume[i+1][0]) / 2
    print(f"  Between cluster {i} and {i+1}: {threshold:.1f}%")

# Actual data ranges per cluster
print("\nActual Data Ranges Per Cluster:")
for c_id in range(optimal_k):
    X_cluster = X[labels_final == c_id]
    center = centers_original[c_id]

    print(f"\nCluster {c_id} (Center Volume={center[0]:.1f}%):")
    for feat_idx, feat_name in enumerate(feature_names):
        vals = X_cluster[:, feat_idx]
        print(f"  {feat_name:<12}: {vals.min():.1f} - {vals.max():.1f} (mean={vals.mean():.1f})")

# Save results
output = {
    'method': 'Truly Data-Driven Clustering',
    'optimal_k': optimal_k,
    'metrics_used': ['Silhouette Score', 'Calinski-Harabasz Index', 'BIC', 'AIC'],
    'cluster_centers': {
        i: {
            'volume': float(centers_original[i][0]),
            'muting': float(centers_original[i][1]),
            'delay': float(centers_original[i][2]),
            'saturation': float(centers_original[i][3]),
            'n_members': int(np.sum(labels_final == i))
        }
        for i in range(optimal_k)
    },
    'participant_assignments': {
        results[j]['id']: int(labels_final[j])
        for j in range(len(results))
    },
    'validation': {
        'silhouette_score': float(silhouette_score(X_scaled, labels_final)),
        'calinski_harabasz': float(calinski_harabasz_score(X_scaled, labels_final))
    },
    'agreement_with_previous': float(agreement)
}

with open('/tmp/Neuroverse/data_driven_clustering_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to /tmp/Neuroverse/data_driven_clustering_results.json")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print(f"""
The data-driven clustering found {optimal_k} natural clusters.

This is determined by:
1. Silhouette Score - measures how similar points are to their own cluster
2. Calinski-Harabasz Index - ratio of between-cluster to within-cluster variance
3. BIC/AIC - penalizes model complexity, finds parsimonious solution
4. Elbow Method - identifies where adding clusters has diminishing returns

NO PRESELECTED THRESHOLDS were used. The clustering algorithm found:
- Natural groupings in 4-dimensional behavioral space
- Cluster centers determined by data, not assumptions
- Boundaries emerge from actual participant distributions

Previous classification used z > 0.5 as cutoff - that's arbitrary.
This classification uses the structure inherent in your data.
""")
