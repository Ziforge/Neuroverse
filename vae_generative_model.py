#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) for Neuroverse Data Generation
Learns latent representations for more realistic synthetic data.
"""

import json
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load original data
with open('/tmp/Neuroverse/questionnaire_behavior_alignment.json', 'r') as f:
    data = json.load(f)

results = data['individual_results']

print("="*80)
print("VARIATIONAL AUTOENCODER (VAE) GENERATIVE MODEL")
print("Deep Generative Approach for Synthetic Data")
print("="*80)

# Prepare data
feature_names = ['volume', 'muting', 'delay', 'saturation']
X_orig = np.array([
    [r['settled_volume'], r['mutes_per_minute'], r['final_delay'], r['final_saturation']]
    for r in results
])
y_orig = np.array([r['combined_class'] for r in results])

print(f"\nOriginal Dataset: n={len(X_orig)}")
print(f"Class Distribution: {dict(Counter(y_orig))}")

# =============================================================================
# SIMPLE VAE IMPLEMENTATION (No TensorFlow/PyTorch dependency)
# =============================================================================

class SimpleVAE:
    """
    Variational Autoencoder using numpy only.
    Learns latent representations for each class.
    """

    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Initialize weights randomly
        np.random.seed(42)

        # Encoder weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_logvar = np.zeros(latent_dim)

        # Decoder weights
        self.W2 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b3 = np.zeros(input_dim)

        self.scaler = StandardScaler()

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.relu(x @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample from latent distribution using reparameterization trick."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z):
        """Decode latent representation to output."""
        h = self.relu(z @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(self, x, recon, mu, logvar):
        """VAE loss = Reconstruction + KL divergence."""
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - recon) ** 2)

        # KL divergence
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))

        return recon_loss + 0.1 * kl_loss  # Beta-VAE with beta=0.1

    def fit(self, X, epochs=500, lr=0.01):
        """Train VAE using gradient descent."""
        X_scaled = self.scaler.fit_transform(X)

        losses = []

        for epoch in range(epochs):
            # Forward pass
            recon, mu, logvar = self.forward(X_scaled)

            # Compute loss
            total_loss = self.loss(X_scaled, recon, mu, logvar)
            losses.append(total_loss)

            # Backward pass (simple gradient descent)
            # Simplified gradient computation
            grad_recon = 2 * (recon - X_scaled) / len(X_scaled)

            # Update decoder weights
            z = self.reparameterize(mu, logvar)
            h2 = self.relu(z @ self.W2 + self.b2)

            self.W3 -= lr * h2.T @ grad_recon
            self.b3 -= lr * np.mean(grad_recon, axis=0)

            grad_h2 = grad_recon @ self.W3.T
            grad_h2 = grad_h2 * (h2 > 0)  # ReLU gradient

            self.W2 -= lr * z.T @ grad_h2
            self.b2 -= lr * np.mean(grad_h2, axis=0)

            # Update encoder weights (simplified)
            grad_z = grad_h2 @ self.W2.T
            grad_mu = grad_z + 0.1 * mu  # KL gradient
            grad_logvar = grad_z * 0.5 * np.exp(0.5 * logvar) + 0.1 * (-0.5 + 0.5 * np.exp(logvar))

            h1 = self.relu(X_scaled @ self.W1 + self.b1)

            self.W_mu -= lr * h1.T @ grad_mu
            self.b_mu -= lr * np.mean(grad_mu, axis=0)
            self.W_logvar -= lr * h1.T @ grad_logvar
            self.b_logvar -= lr * np.mean(grad_logvar, axis=0)

            grad_h1 = grad_mu @ self.W_mu.T + grad_logvar @ self.W_logvar.T
            grad_h1 = grad_h1 * (h1 > 0)

            self.W1 -= lr * X_scaled.T @ grad_h1
            self.b1 -= lr * np.mean(grad_h1, axis=0)

            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: Loss = {total_loss:.4f}")

        return losses

    def generate(self, n_samples):
        """Generate new samples from learned latent space."""
        # Sample from standard normal in latent space
        z = np.random.randn(n_samples, self.latent_dim)
        recon_scaled = self.decode(z)
        return self.scaler.inverse_transform(recon_scaled)

    def generate_from_class(self, X_class, n_samples):
        """Generate samples similar to a specific class."""
        X_scaled = self.scaler.transform(X_class)
        mu, logvar = self.encode(X_scaled)

        # Sample around class mean in latent space
        class_mu = np.mean(mu, axis=0)
        class_std = np.std(mu, axis=0) * 1.5  # Slightly wider

        z = np.random.randn(n_samples, self.latent_dim) * class_std + class_mu
        recon_scaled = self.decode(z)
        return self.scaler.inverse_transform(recon_scaled)


print("\n" + "="*80)
print("TRAINING CLASS-CONDITIONAL VAEs")
print("="*80)

# Train separate VAE for each class
class_vaes = {}
class_latent_spaces = {}

for cls in ['hyper', 'typical', 'hypo']:
    print(f"\n{cls.upper()} Class:")
    X_cls = X_orig[y_orig == cls]

    if len(X_cls) < 2:
        print(f"  Insufficient samples ({len(X_cls)}), using augmented approach")
        # Use GMM fallback for very small classes
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        gmm.fit(X_cls)
        class_vaes[cls] = ('gmm', gmm)
        class_latent_spaces[cls] = None
    else:
        vae = SimpleVAE(input_dim=4, latent_dim=2, hidden_dim=6)
        losses = vae.fit(X_cls, epochs=300, lr=0.01)
        class_vaes[cls] = ('vae', vae)

        # Store latent representations
        X_scaled = vae.scaler.transform(X_cls)
        mu, _ = vae.encode(X_scaled)
        class_latent_spaces[cls] = mu

        print(f"  Final loss: {losses[-1]:.4f}")


# =============================================================================
# GENERATE LARGE SYNTHETIC DATASET
# =============================================================================
print("\n" + "="*80)
print("GENERATING VAE-BASED SYNTHETIC DATA")
print("="*80)

def generate_vae_population(class_vaes, X_orig, y_orig, n_per_class=500):
    """Generate balanced population using trained VAEs."""
    X_synth = []
    y_synth = []

    for cls in ['hyper', 'typical', 'hypo']:
        model_type, model = class_vaes[cls]
        X_cls = X_orig[y_orig == cls]

        if model_type == 'vae':
            samples = model.generate_from_class(X_cls, n_per_class)
        else:  # GMM fallback
            samples, _ = model.sample(n_per_class)

        # Clip to realistic ranges
        samples[:, 0] = np.clip(samples[:, 0], 10, 100)  # Volume
        samples[:, 1] = np.clip(samples[:, 1], 0, 2)      # Muting
        samples[:, 2] = np.clip(samples[:, 2], 0, 100)    # Delay
        samples[:, 3] = np.clip(samples[:, 3], 0, 100)    # Saturation

        X_synth.extend(samples)
        y_synth.extend([cls] * n_per_class)

        print(f"{cls.upper()}: Generated {n_per_class} samples")

    return np.array(X_synth), np.array(y_synth)

X_vae, y_vae = generate_vae_population(class_vaes, X_orig, y_orig, n_per_class=500)

print(f"\nVAE Dataset: n={len(X_vae)}")
print(f"Class Distribution: {dict(Counter(y_vae))}")

# =============================================================================
# VALIDATE GENERATED DATA
# =============================================================================
print("\n" + "="*80)
print("VALIDATION: COMPARE ORIGINAL VS GENERATED")
print("="*80)

print("\nStatistical Comparison (Original vs VAE-Generated):")
print("-" * 70)

for cls in ['hyper', 'typical', 'hypo']:
    X_orig_cls = X_orig[y_orig == cls]
    X_vae_cls = X_vae[y_vae == cls]

    print(f"\n{cls.upper()}:")
    print(f"  Original (n={len(X_orig_cls)}) vs Generated (n={len(X_vae_cls)})")

    for i, feat in enumerate(feature_names):
        orig_mean = np.mean(X_orig_cls[:, i])
        orig_std = np.std(X_orig_cls[:, i])
        vae_mean = np.mean(X_vae_cls[:, i])
        vae_std = np.std(X_vae_cls[:, i])

        # Relative error
        rel_error = abs(vae_mean - orig_mean) / (orig_mean + 1e-6) * 100

        print(f"  {feat:<12}: Orig={orig_mean:.2f} (SD={orig_std:.2f})  VAE={vae_mean:.2f} (SD={vae_std:.2f})  Err={rel_error:.1f}%")

# =============================================================================
# TRAIN CLASSIFIER ON VAE-GENERATED DATA
# =============================================================================
print("\n" + "="*80)
print("CLASSIFIER TRAINING ON VAE DATA")
print("="*80)

scaler = StandardScaler()
X_vae_scaled = scaler.fit_transform(X_vae)

rf_vae = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# Cross-validation on VAE data
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf_vae, X_vae_scaled, y_vae, cv=10)
print(f"10-Fold CV on VAE Data: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Train on VAE data, test on original
rf_vae.fit(X_vae_scaled, y_vae)
X_orig_scaled = scaler.transform(X_orig)
y_pred = rf_vae.predict(X_orig_scaled)
orig_accuracy = np.mean(y_pred == y_orig)
print(f"Trained on VAE → Test on Original: {orig_accuracy:.3f}")

# =============================================================================
# LATENT SPACE VISUALIZATION
# =============================================================================
print("\n" + "="*80)
print("LATENT SPACE ANALYSIS")
print("="*80)

print("\nLatent Space Representations (2D):")
for cls in ['hyper', 'typical', 'hypo']:
    if class_latent_spaces[cls] is not None:
        mu = class_latent_spaces[cls]
        print(f"\n{cls.upper()}:")
        print(f"  Latent Mean: [{np.mean(mu, axis=0)[0]:.3f}, {np.mean(mu, axis=0)[1]:.3f}]")
        print(f"  Latent Std:  [{np.std(mu, axis=0)[0]:.3f}, {np.std(mu, axis=0)[1]:.3f}]")

        # Sample positions
        print(f"  Individual Latent Positions:")
        for i in range(len(mu)):
            print(f"    Sample {i+1}: [{mu[i, 0]:.3f}, {mu[i, 1]:.3f}]")

# =============================================================================
# INTERPOLATION IN LATENT SPACE
# =============================================================================
print("\n" + "="*80)
print("LATENT SPACE INTERPOLATION")
print("="*80)

print("\nGenerating intermediate profiles between classes...")

# Get class centroids in latent space
centroids = {}
for cls in ['hyper', 'typical', 'hypo']:
    if class_latent_spaces[cls] is not None:
        centroids[cls] = np.mean(class_latent_spaces[cls], axis=0)

if 'hyper' in centroids and 'hypo' in centroids:
    print("\nInterpolation: Hypersensitive → Hyposensitive")
    print("-" * 60)

    # Linear interpolation in latent space
    n_interpolations = 5
    vae_hyper = class_vaes['hyper'][1]

    for i in range(n_interpolations):
        alpha = i / (n_interpolations - 1)
        z_interp = (1 - alpha) * centroids['hyper'] + alpha * centroids['hypo']

        # Decode interpolated latent
        decoded = vae_hyper.decode(z_interp.reshape(1, -1))
        sample = vae_hyper.scaler.inverse_transform(decoded)[0]

        # Clip
        sample[0] = np.clip(sample[0], 10, 100)
        sample[1] = np.clip(sample[1], 0, 2)
        sample[2] = np.clip(sample[2], 0, 100)
        sample[3] = np.clip(sample[3], 0, 100)

        print(f"  α={alpha:.2f}: Vol={sample[0]:.1f}%, Mute={sample[1]:.2f}, Delay={sample[2]:.1f}%, Sat={sample[3]:.1f}%")

# =============================================================================
# COMBINED MEGA-DATASET
# =============================================================================
print("\n" + "="*80)
print("MEGA DATASET: ORIGINAL + VAE + PREVIOUS AUGMENTATIONS")
print("="*80)

# Load previous augmentation
prev_augmented = np.load('/tmp/Neuroverse/augmented_data.npz', allow_pickle=True)
X_ensemble = prev_augmented['X_ensemble']
y_ensemble = prev_augmented['y_ensemble']

# Combine everything
X_mega = np.vstack([X_orig, X_vae, X_ensemble])
y_mega = np.concatenate([y_orig, y_vae, y_ensemble])

print(f"Mega Dataset: n={len(X_mega)}")
print(f"Class Distribution: {dict(Counter(y_mega))}")
print(f"Total Expansion: {len(X_mega) / len(X_orig):.1f}x original")

# Final classifier on mega dataset
scaler_mega = StandardScaler()
X_mega_scaled = scaler_mega.fit_transform(X_mega)

rf_mega = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
cv_mega = cross_val_score(rf_mega, X_mega_scaled, y_mega, cv=10)
print(f"\nMega Dataset 10-Fold CV: {cv_mega.mean():.3f} (+/- {cv_mega.std():.3f})")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*80)
print("SAVING VAE RESULTS")
print("="*80)

output = {
    'vae_training': {
        'method': 'Class-Conditional Variational Autoencoder',
        'architecture': {
            'input_dim': 4,
            'latent_dim': 2,
            'hidden_dim': 6
        },
        'classes_trained': list(class_vaes.keys())
    },
    'vae_generated_dataset': {
        'n': len(X_vae),
        'n_per_class': 500,
        'class_distribution': dict(Counter(y_vae))
    },
    'validation': {
        'cv_accuracy_on_vae': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'train_vae_test_original': float(orig_accuracy)
    },
    'mega_dataset': {
        'n_total': len(X_mega),
        'expansion_factor': len(X_mega) / len(X_orig),
        'class_distribution': dict(Counter(y_mega)),
        'cv_accuracy': float(cv_mega.mean()),
        'cv_std': float(cv_mega.std())
    },
    'latent_space': {
        cls: {
            'mean': class_latent_spaces[cls].mean(axis=0).tolist() if class_latent_spaces[cls] is not None else None,
            'std': class_latent_spaces[cls].std(axis=0).tolist() if class_latent_spaces[cls] is not None else None
        }
        for cls in ['hyper', 'typical', 'hypo']
    },
    'generated_statistics': {
        cls: {
            feat: {
                'mean': float(np.mean(X_vae[y_vae == cls][:, i])),
                'std': float(np.std(X_vae[y_vae == cls][:, i]))
            }
            for i, feat in enumerate(feature_names)
        }
        for cls in ['hyper', 'typical', 'hypo']
    }
}

with open('/tmp/Neuroverse/vae_generative_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Save mega dataset
np.savez('/tmp/Neuroverse/mega_dataset.npz',
         X_mega=X_mega, y_mega=y_mega,
         X_vae=X_vae, y_vae=y_vae,
         feature_names=feature_names)

print(f"Results saved to:")
print(f"  - /tmp/Neuroverse/vae_generative_results.json")
print(f"  - /tmp/Neuroverse/mega_dataset.npz")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Original: n={len(X_orig)}")
print(f"VAE Generated: n={len(X_vae)}")
print(f"Mega Dataset: n={len(X_mega)} ({len(X_mega)/len(X_orig):.0f}x expansion)")
print(f"Final CV Accuracy: {cv_mega.mean():.1%}")
print("="*80)
