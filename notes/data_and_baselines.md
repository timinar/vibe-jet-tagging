## **Introduction: The Data and the Physics**

This document outlines the problem of quark-gluon jet tagging, a cornerstone of data analysis at the Large Hadron Collider (LHC). The goal is to classify jets—collimated sprays of particles—based on whether they originated from a fundamental quark or a gluon. This capability is crucial for enhancing searches for new physics and suppressing backgrounds in precision measurements. The summary below describes the benchmark dataset used for this task and the fundamental physics that makes discrimination possible.

---

## 🔬 The Benchmark Dataset

To provide a common ground for research, the community has widely adopted the **Pythia8 Quark and Gluon Jets dataset** created by Komiske, Metodiev, and Thaler. It provides a clean, standardized environment for developing and comparing jet tagging algorithms.

* **Generation:** The dataset contains 2 million jets (one million each of quarks and gluons) simulated using the Pythia8 event generator. The jets are produced in proton-proton collisions at a center-of-mass energy of $\sqrt{s}=14$ TeV. The specific process is $Z$+jet production, where the Z boson decays invisibly to neutrinos, ensuring that the only visible particles originate from the jet itself.
* **Kinematic Selection:** A crucial feature is the tight kinematic selection. All jets are restricted to a narrow transverse momentum range of **500-550 GeV** and a rapidity of $|y|<1.7$. This is a deliberate choice to remove trivial kinematic differences, forcing algorithms to learn from the jet's internal structure (its "substructure").
* **Data Representation:** Each jet is presented as a variable-length list of its constituent particles, forming an unordered **"point cloud"**. For each particle, the dataset provides four key features: its transverse momentum ($p_T$), rapidity ($y$), azimuthal angle ($\phi$), and its Particle Data Group ID (PDG ID), which identifies the particle's type.

---

## ⚛️ The Fundamental Physics of Discrimination

The observable differences between quark and gluon jets originate from the fundamental principles of Quantum Chromodynamics (QCD), the theory of the strong nuclear force.



* **QCD Color Charge:** The primary distinction lies in the "color charge" carried by each particle type. Quarks exist in the *fundamental* representation of the SU(3) color group, while gluons, the force carriers, exist in the *adjoint* representation.
* **Color Factors & Radiation:** This difference is quantified by a parameter known as the **QCD color factor**, which determines the probability of a particle radiating a gluon.
    * For quarks, the color factor is $C_F = 4/3$.
    * For gluons, the color factor is $C_A = 3$.
* **The Critical Ratio:** The single most important number in quark-gluon tagging is the ratio of these color factors: **$C_A/C_F = 9/4 = 2.25$**. This means that, all else being equal, a gluon is **2.25 times more likely to radiate another gluon** than a quark is.
* **Observable Consequences:** This enhanced radiation propensity of gluons directly leads to concrete, measurable differences in the final jets:
    1.  **Higher Multiplicity:** The more active radiation cascade results in gluon jets being composed of a larger number of final-state particles. This is the single most powerful discriminating feature.
    2.  **Broader Angular Distribution:** The increased radiation leads to a wider, more diffuse spray of particles. Quark jets, in contrast, are typically narrower and more collimated.
    3.  **Softer Fragmentation:** A quark jet's momentum is often concentrated in one or two leading particles. A gluon jet's energy is more democratically distributed among a larger number of lower-energy particles, giving it a "softer" energy profile.

Ultimately, any successful classification algorithm, from a simple heuristic to a state-of-the-art neural network, must effectively learn to identify these three observable consequences of the underlying difference in QCD color charge.

---
## 📊 Summary of Tagging Methods and Performance

This table summarizes the main architectural approaches applied to the Pythia8 dataset, their typical performance, and key characteristics.

| Method / Architecture | Typical AUC | Key Characteristics | Primary Reference |
| :--- | :--- | :--- | :--- |
| **Boosted Decision Trees (BDT)** | 0.70 - 0.73 | Operates on hand-crafted physics features (e.g., multiplicity, jet width). Fast, interpretable, and good for simple baselines. | - |
| **CNNs (on Jet Images)** | 0.70 - 0.75 | Treats jets as 2D images. Leverages computer vision techniques but suffers from pixelation and information loss. | [arXiv:1612.01551](https://arxiv.org/abs/1612.01551) |
| **PFNs / Deep Sets** | 0.90 - 0.905 | First to treat jets as permutation-invariant, unordered sets of particles ("point clouds"). Can be constructed to be theoretically robust (IRC-safe). | [arXiv:1810.05165](https://arxiv.org/abs/1810.05165) |
| **GNNs (ParticleNet)** | ~0.912 | Represents the jet as a graph of particles. Dynamically computes particle neighborhoods to capture local and hierarchical structure. Former SOTA. | [arXiv:1902.08570](https://arxiv.org/abs/1902.08570) |
| **Equivariant GNNs (LorentzNet)** | ~0.916 | A GNN that enforces Lorentz group equivariance as an inductive bias, improving generalization from limited data. | [arXiv:2201.08187](https://arxiv.org/abs/2201.08187) |
| **Transformers (Particle Transformer)** | **0.923** | **Current SOTA**. Augments the self-attention mechanism with physics-motivated pairwise features between particles. Achieves top performance when pre-trained. | [arXiv:2202.03772](https://arxiv.org/abs/2202.03772) |

---

## 📜 Literature Overview & Architectural Evolution

The approach to solving quark-gluon tagging on this dataset has evolved through several distinct phases, with performance steadily increasing.

* **Classical & Heuristic Baselines:** Early methods used hand-crafted, physics-motivated features (e.g., jet multiplicity, width, N-subjettiness) fed into algorithms like **Boosted Decision Trees (BDTs)**. These are fast and interpretable, establishing a performance baseline of **~70-73% AUC**.
* **Jet Images & CNNs:** A major shift involved representing jets as 2D images on the detector's angular grid and applying **Convolutional Neural Networks (CNNs)**. This approach leveraged advances from computer vision but suffered from information loss due to pixelation and sparsity, achieving **~70-75% AUC**.
* **Point Clouds & GNNs:** The current dominant paradigm treats jets as unordered "point clouds" of particles, a more natural representation. This started with **Energy Flow Networks (EFNs)** and **Particle Flow Networks (PFNs)** based on Deep Sets theory. This was followed by **Graph Neural Networks (GNNs)** like **ParticleNet**, which constructed dynamic graphs to capture particle relationships. ParticleNet was a major breakthrough, achieving **~91.1% AUC** and representing a strong, widely-used baseline.
* **Transformers & State-of-the-Art:** Inspired by successes in NLP, **Transformer architectures** adapted for physics data now define the performance frontier. The key model, **Particle Transformer (ParT)**, augmented the self-attention mechanism with physics-based pairwise particle features, achieving the current SOTA performance of **92.3% AUC** when pre-trained on the large JetClass dataset.

---

## 🏆 State-of-the-Art Benchmarks

The undisputed state-of-the-art performance on the Pythia8 quark-gluon dataset is held by the **Particle Transformer (ParT)** architecture.

* **Top Performance:** **AUC = 0.9230** and **Accuracy = 85.2%**. This was achieved by pre-training the model on the 100-million jet JetClass dataset and then fine-tuning it on the Pythia8 dataset.
* **Practical Impact:** Small AUC gains translate to significant practical improvements. At a 50% signal efficiency (correctly identifying 50% of quarks), the SOTA model achieves a background rejection of **61.0**. This means it correctly rejects 61 gluon jets for every 1 that it misidentifies as a quark—a nearly 80% improvement in rejection power over earlier deep learning models.
* **Strong Baselines:** For comparison, a well-established GNN like **ParticleNet** achieves an AUC of **~0.9116**, while classical BDTs plateau around **0.73 AUC**.

---

## 🔬 Relevant Features and Pre-Processing

The dataset provides low-level information for each particle in a jet, which can be used directly or to engineer higher-level features.

* **Raw Features:** The fundamental input for modern models is the list of constituent particles for each jet, where each particle is described by four features:
    1.  **Transverse Momentum ($p_T$):** The particle's momentum perpendicular to the beamline. Often normalized relative to the jet's total $p_T$ ($z_i = p_{T_i} / p_{T_{jet}}$) to ensure theoretical robustness.
    2.  **Rapidity ($y$) and Azimuthal Angle ($\phi$):** Angular coordinates defining the particle's direction. Often centered relative to the jet's axis to ensure translational invariance.
    3.  **Particle Identification (PDG ID):** A code identifying the particle type (e.g., pion, photon, kaon), which can be encoded or embedded.

* **High-Level Physics Observables:** For simpler models like BDTs, or for providing physical insight, these hand-crafted features are powerful discriminants:
    * **Constituent Multiplicity:** The number of particles in a jet. This is the single most powerful observable, achieving **~65-68% AUC** on its own.
    * **Jet Width/Broadening:** Measures the angular spread of energy in the jet. A wider jet is more likely to be a gluon.
    * **Momentum Dispersion ($p_T^D$):** Captures how democratically momentum is shared among constituents. Achieves **~63-65% AUC**.

---

## ⚛️ Core Physics Concepts for Discrimination

The ability to distinguish quark and gluon jets stems directly from the fundamental theory of the strong force, Quantum Chromodynamics (QCD). The "by-eye" intuition comes from understanding how this difference manifests.



The primordial difference is **QCD Color Charge**. Quarks and gluons carry different types of color charge, quantified by a "color factor" that determines how strongly they radiate other gluons.
* The gluon color factor is $C_A = 3$.
* The quark color factor is $C_F = 4/3$.

The ratio **$C_A/C_F = 9/4 = 2.25$** is the most important number in quark-gluon tagging. It means a gluon is fundamentally **2.25 times more likely to radiate another gluon** than a quark is.

This leads to three key observable consequences:
1.  **Higher Multiplicity:** Because gluons radiate more, their resulting jets contain significantly more particles. Gluon jets have roughly 1.7-2.0 times more constituents than quark jets of the same energy.
2.  **Broader Angular Spread:** The increased radiation causes the energy in gluon jets to be more spread out, making them physically **wider** and more diffuse than the typically narrow and collimated quark jets.
3.  **Softer Fragmentation:** In a quark jet, a large fraction of the energy is often carried by one or two leading particles. In a gluon jet, the energy is distributed more "democratically" among a larger number of lower-energy particles.
