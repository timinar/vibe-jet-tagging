# Baseline Classifier Usage Guide

Quick reference for using the baseline ML classifiers for quark-gluon jet tagging.

## 📋 Dataset Overview

**Task:** Binary classification of particle jets (quark vs. gluon) — a fundamental benchmark in collider physics.

### Loading the Data

```python
import numpy as np
from pathlib import Path

# Load dataset
data = np.load("data/qg_jets.npz")
X, y = data['X'], data['y']

# X: (100000, 139, 4) - jets with up to 139 particles, 4 features each
# y: (100000,) - binary labels (0=gluon, 1=quark)
```

### Dataset Summary

| Property | Value |
|----------|-------|
| **Total samples** | 100,000 (50k quark, 50k gluon) |
| **Class balance** | Perfectly balanced (50/50) |
| **Shape** | X: (100000, 139, 4), y: (100000,) |
| **Features per particle** | 4: [pt, rapidity, φ, pdgid] |
| **Zero-padded** | Yes (padding where pt=0) |

### Key Physics Insight

**Multiplicity is the primary discriminator:**
- Quark jets: **33.4 ± 13.3** particles/jet
- Gluon jets: **53.2 ± 15.8** particles/jet
- **Gluon/Quark ratio: 1.59** — Gluons radiate more due to stronger QCD coupling

> **Why this matters:** Gluons have ~59% more particles than quarks on average. This is the fundamental physics signal that makes classification possible.

### Feature Descriptions

| Feature | Description | Range | Use in Classifiers |
|---------|-------------|-------|-------------------|
| `pt` | Transverse momentum | [0.0001, 548.49] GeV | ✅ Realistic |
| `rapidity` | Pseudorapidity η | [-2.13, 2.20] | ✅ Realistic |
| `azimuthal_angle` | Azimuthal angle φ | [-0.42, 6.72] rad | ✅ Realistic |
| `pdgid` | Particle ID (MC truth) | 14 unique types | ⚠️ Not available in real detectors |

> **Note on `pdgid`:** Our baseline classifiers **do not use pdgid** to remain realistic—real detectors cannot perfectly identify particle species.

---

## 🚀 Using Baseline Classifiers

### 1. Multiplicity Logistic Regression (Simplest Baseline)

Uses only one feature: particle count per jet. Expected AUC: ~0.67-0.70

```python
from vibe_jet_tagging import MultiplicityLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
data = np.load("data/qg_jets.npz")
X, y = data['X'], data['y']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
clf = MultiplicityLogisticRegression()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# Evaluate
auc = roc_auc_score(y_test, y_prob)
print(f"Multiplicity LR AUC: {auc:.4f}")
```

### 2. XGBoost on Raw Particles (Strong Baseline)

Uses flattened kinematic features (pt, rapidity, φ). Expected AUC: ~0.73-0.76

```python
from vibe_jet_tagging import XGBoostRawParticles

# Train classifier
clf = XGBoostRawParticles(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# Evaluate
auc = roc_auc_score(y_test, y_prob)
print(f"XGBoost AUC: {auc:.4f}")
```

### 3. Complete Example with Both Baselines

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from vibe_jet_tagging import MultiplicityLogisticRegression, XGBoostRawParticles

# Load dataset
data = np.load("data/qg_jets.npz")
X, y = data['X'], data['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset: {len(y):,} jets ({len(y_train):,} train, {len(y_test):,} test)")

# Train Multiplicity LR
print("\n[1/2] Training Multiplicity Logistic Regression...")
clf_mult = MultiplicityLogisticRegression()
clf_mult.fit(X_train, y_train)
y_prob_mult = clf_mult.predict_proba(X_test)

auc_mult = roc_auc_score(y_test, y_prob_mult)
acc_mult = accuracy_score(y_test, clf_mult.predict(X_test))
print(f"  ✓ Multiplicity LR: AUC = {auc_mult:.4f}, Acc = {acc_mult:.4f}")

# Train XGBoost
print("\n[2/2] Training XGBoost...")
clf_xgb = XGBoostRawParticles(n_estimators=100, max_depth=5)
clf_xgb.fit(X_train, y_train)
y_prob_xgb = clf_xgb.predict_proba(X_test)

auc_xgb = roc_auc_score(y_test, y_prob_xgb)
acc_xgb = accuracy_score(y_test, clf_xgb.predict(X_test))
print(f"  ✓ XGBoost: AUC = {auc_xgb:.4f}, Acc = {acc_xgb:.4f}")

print("\n" + "="*60)
print("BASELINE RESULTS")
print("="*60)
print(f"Multiplicity LR: AUC = {auc_mult:.4f} (simple physics baseline)")
print(f"XGBoost:         AUC = {auc_xgb:.4f} (strong ML baseline)")
print("="*60)
```

---

## 🎯 Baseline Performance Expectations

| Baseline | Typical AUC | Typical Accuracy | Description |
|----------|-------------|------------------|-------------|
| **Random** | 0.500 | 50% | Coin flip |
| **Multiplicity LR** | 0.67-0.70 | 63-65% | Simple physics feature |
| **XGBoost Raw** | 0.73-0.76 | 67-70% | Strong ML baseline |
| **Literature SOTA** | 0.923 | 85.2% | Particle Transformer (pretrained) |

---

## 📝 Next Steps

1. **Compare with your LLM approach**: Use these AUCs as reference points
2. **Feature importance**: Analyze what XGBoost learns beyond multiplicity
3. **Error analysis**: Examine misclassified jets to understand failure modes
4. **Advanced baselines**: Consider ParticleNet or Energy Flow Networks

---

*Dataset source: [EnergyFlow QG-Jets](https://energyflow.network/docs/datasets/#quark-and-gluon-jets)*
