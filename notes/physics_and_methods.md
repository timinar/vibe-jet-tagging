# Quark-Gluon Jet Tagging: Physics and Methods

A concise overview of the fundamental physics driving quark-gluon discrimination and the evolution of machine learning approaches.

## 🎯 The Problem

**Goal:** Classify jets (collimated sprays of particles) by their origin—quark vs. gluon.

**Why it matters:** Critical for LHC physics searches and precision measurements by enabling better background suppression.

---

## 📊 The Benchmark Dataset

**Pythia8 Quark and Gluon Jets** (Komiske, Metodiev, Thaler) - [Zenodo](https://zenodo.org/records/3164691)

- **2M jets** (1M quarks, 1M gluons) at √s = 14 TeV
- **Tight kinematic cuts:** pt ∈ [500, 550] GeV, |y| < 1.7
- **Point cloud format:** Variable-length list of particles per jet
- **Features per particle:** pt, rapidity (y), azimuthal angle (φ), PDG ID

The tight kinematic selection forces algorithms to learn from jet **substructure**, not trivial energy differences.

---

## ⚛️ The Physics: Why They're Different

### QCD Color Charge

The fundamental distinction stems from **Quantum Chromodynamics (QCD)**:

- Quarks: color factor **CF = 4/3** (fundamental representation)
- Gluons: color factor **CA = 3** (adjoint representation)

### The Magic Number: CA/CF = 9/4 = 2.25

**Gluons are 2.25× more likely to radiate** than quarks. This single ratio drives all observable differences.

### Three Observable Consequences

1. **Higher Multiplicity** (most powerful signal)
   - Quark jets: ~33 particles
   - Gluon jets: ~53 particles
   - Gluon/Quark ratio: **1.59×**

2. **Broader Angular Distribution**
   - Gluon jets are wider and more diffuse
   - Quark jets are narrow and collimated

3. **Softer Fragmentation**
   - Quark jets: energy concentrated in 1-2 leading particles
   - Gluon jets: energy democratically distributed

---

## 🏆 Evolution of ML Methods

### Classical Approaches (AUC ~0.84-0.86)
- **Our implementations:** Logistic Regression (0.843), XGBoost (0.857)
- Hand-crafted features or flattened particle data
- Fast, interpretable, strong baselines

### Point Clouds: PFNs/Deep Sets (AUC ~0.90-0.905)
- **Breakthrough:** Jets as permutation-invariant particle sets
- Energy Flow Networks, Particle Flow Networks
- Theoretically motivated (IRC-safe)
- Reference: [arXiv:1810.05165](https://arxiv.org/abs/1810.05165)

### Graph Neural Networks (AUC ~0.912)
- **ParticleNet:** Dynamic graph construction
- Captures local and hierarchical structure
- Former SOTA
- Reference: [arXiv:1902.08570](https://arxiv.org/abs/1902.08570)

### Transformers (AUC ~0.923) ⭐ Current SOTA
- **Particle Transformer (ParT)**
- Self-attention + physics-motivated pairwise features
- Pre-trained on 100M jets (JetClass)
- **Practical impact:** 61× background rejection at 50% signal efficiency
- Reference: [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)

---

## 📊 Quick Performance Reference

| Method | AUC | Notes |
|--------|-----|-------|
| Random | 0.500 | Coin flip baseline |
| **Our Multiplicity LR** | **0.843** | **Simple baseline (single feature)** |
| **Our XGBoost** | **0.857** | **Strong baseline (kinematic features)** |
| PFNs/Deep Sets | 0.90-0.905 | Point cloud revolution |
| ParticleNet (GNN) | 0.912 | Former SOTA |
| Particle Transformer | **0.923** | **Current SOTA (pretrained)** |

---

## 🔬 Feature Usage Notes

### Raw Features (Per Particle)
1. **pt**: Transverse momentum - ✅ Use (realistic)
2. **y, φ**: Angular coordinates - ✅ Use (realistic)
3. **PDG ID**: Particle type - ⚠️ **MC truth only, not realistic**

### High-Level Observables
- **Multiplicity**: Most powerful single feature (AUC 0.84 with LR)
- **Jet width**: Angular energy spread
- **pt dispersion**: Momentum sharing democratization

**Best practice:** Use only kinematic features (pt, y, φ) for realistic classifiers.

---

## 💡 Key Takeaways

1. **Physics drives ML:** The 2.25× radiation ratio creates learnable patterns
2. **Multiplicity is king:** Single feature achieves 84.3% AUC
3. **Our baselines are strong:** XGBoost at 85.7% AUC beats most pre-2019 methods
4. **Deep learning gains:** PFNs/ParticleNet add ~5% AUC over classical methods
5. **SOTA at 92.3%:** Particle Transformer with pretraining still 6.6% ahead

---

*For usage examples, see [baseline_usage.md](baseline_usage.md)*
