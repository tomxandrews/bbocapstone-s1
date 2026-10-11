# Bayesian Black-Box Optimisation (BBO) Capstone (bbo-capstone)

## 1. Executive Non-Technical Summary (~100 Words)

This project tackles a high-stakes machine learning challenge: optimizing eight complex, unknown black-box systems under severe evaluation constraints. Each system simulates an expensive real-world process—such as contamination detection, biomedical therapeutic formulation, warehouse logistics, and automated neural network hyperparameter tuning—where physical testing is costly, noisy, or resource-intensive. Rather than relying on inefficient trial-and-error, we implemented Bayesian optimization using Gaussian Process surrogates and intelligent acquisition functions (Upper Confidence Bound and Expected Improvement). Across weekly iterations, this data-driven framework methodically balanced exploring uncertain spaces with exploiting high-performing coordinates, discovering all-time record outputs across multiple systems, including nearly tripling the chemical manufacturing yield.

---

## 2. Project Goal, ML Relevance & Professional Connection

### 2.1 The Goal of the Challenge
In modern engineering and scientific research, practitioners routinely encounter systems whose internal mechanisms cannot be expressed through closed-form mathematical equations. These black-box functions can only be evaluated through costly, time-consuming experiments. The objective of this capstone project is to discover the global maximum output for eight synthetic black-box functions of increasing dimensionality (2D to 8D) under a strict evaluation budget of one query per function per weekly cycle.

### 2.2 Relevance to Machine Learning
Black-box optimization (BBO) is fundamental to advanced machine learning and data science:
* Hyperparameter Optimization (HPO): Training large neural networks or gradient-boosted ensembles can take days or weeks of compute. Grid and random search scale exponentially, whereas Bayesian Optimization navigates high-dimensional parameter spaces with maximal sample efficiency.
* Active Learning & Experimental Design: In applications such as biomedical therapeutics, robotics, and industrial process synthesis, gathering ground-truth labels requires expensive laboratory assays or physical prototypes. Bayesian optimization provides a rigorous probabilistic foundation for choosing the single most informative next experiment.
* Quantified Predictive Uncertainty: Unlike standard point-estimate regressors, Bayesian surrogates produce full posterior probability distributions (mean and variance), allowing algorithms to make mathematically sound risk-reward trade-offs.

### 2.3 Connection to Personal & Professional Goals
This capstone bridges theoretical machine learning principles and mission-critical production decision-making. Developing a custom BBO architecture establishes expertise in:
1. Designing resilient active learning pipelines for sample-starved industrial regimes.
2. Managing the bias-variance trade-off in Gaussian Process covariance kernels (Anisotropic Matern 5/2 vs. RBF).
3. Architecting scalable, reproducible ML workflows using clean code practices, automated tracking, and rigorous Model Cards and Datasheets.

---

## 3. Problem Formulation, Input Constraints & Function Architecture

### 3.1 Input Space Format & Operational Constraints
All eight functions operate across continuous, normalized hypercubes subject to strict format specifications:
* Domain Bounding: Coordinates are bounded within [0.000000, 0.999999]^d, where d is the dimensionality of the function.
* String Submission Syntax: Each coordinate is submitted as an exact 6-decimal string formatted with no spaces around hyphens:
  Format: x1-x2-...-xd (e.g., 0.660114-0.674026 for 2D)
* Budget Limitation: Exactly one evaluation per function per weekly iteration.

### 3.2 Output Characteristics & Maximization Framing
Every task in this competition is strictly framed as a maximization problem:
* Sign Transformations: Real-world processes that naturally represent minimizations (e.g., adverse compound side-effects in Function 3 or recipe penalty scores in Function 6) are transformed via mathematical negation so that higher output values reflect superior performance.
* Non-Linearities & Noise: Surface topographies range from flat zero-plateaus with localized proximity spikes to multi-modal ridges and severe penalty drop-offs.

### 3.3 Overview of the Eight Black-Box Functions

| Function | Dim | Real-World Application Domain | Initial N_0 | Initial Best Output | Current Global Best | Key Landscape Characteristics |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Function 1** | 2D | Environmental Contamination Source Detection | 10 | 0.000000 | **0.000000** *(Active: -0.003606)* | Uninformative zero-plateau with localized exponential proximity spike |
| **Function 2** | 2D | Noisy Log-Likelihood Modeling | 10 | 0.611205 | **0.611205** | Noisy, multi-modal response surface with an interior ascending ridge |
| **Function 3** | 3D | Pharmacological Therapeutic Side-Effect Minimization | 15 | -0.034835 | **-0.034835** | Negated side-effects; steep penalty cliff when compound x2 > 0.15 |
| **Function 4** | 4D | Central Warehouse Placement Logistics | 30 | -4.025542 | **-3.857021** (W2 Peak) | Dynamic distance model; optimal configuration in central hyper-volume |
| **Function 5** | 4D | Chemical Process Yield Optimization | 20 | 1088.859618 | **3044.749281** (W2 Peak) | Unimodal, steep ascending ridge toward upper boundary (x2, x3 -> 1.0) |
| **Function 6** | 5D | Cake Recipe Taste Optimization | 20 | -0.714265 | **-0.664287** (W1 Peak) | Expert taster penalties; severe penalty cliff when ingredient x4 > 0.025 |
| **Function 7** | 6D | Neural Network Hyperparameter Tuning | 30 | 1.364968 | **1.364970** (W1 Peak) | High-dimensional performance surface with isolated local optima clusters |
| **Function 8** | 8D | Complex Neural Architecture Tuning | 40 | 9.598482 | **9.598482** | Severe curse of dimensionality; optimal regime requires x0..x3 <= 0.05 |


---

## 4. Machine Learning Strategies, Modeling & Exploration-Exploitation Trade-Offs

### 4.1 Gaussian Process Surrogate Modeling
We employ Gaussian Process (GP) Regression as our primary surrogate architecture. A GP defines a prior distribution over functions:
f(x) ~ GP(m(x), k(x, x'))
* Covariance Kernel: We utilize an anisotropic Matern 5/2 Kernel combined with a White Noise Kernel to handle smooth non-linearities and observational noise.
* Automatic Relevance Determination (ARD): By assigning independent length scales l_i to each feature dimension, ARD automatically identifies which input parameters dominate output variance (e.g., identifying that x4 in Function 6 governed taste penalties, whereas other dimensions were less sensitive).
* Noise Handling: A Gaussian white noise kernel absorbs observational noise, preventing ill-conditioned covariance matrix inversion.

### 4.2 Acquisition Functions & Tactical Decision-Making
To decide the optimal next query point, we evaluate two primary acquisition functions:
1. Upper Confidence Bound (UCB):
   alpha_UCB(x) = mu(x) + beta_t * sigma(x)
   Used during early exploratory mapping. The parameter beta_t controls the exploration bonus:
   * Week 1 (Global Exploration): beta in [1.5, 2.5] to evaluate broad uncertainty regions.
   * Week 2 (Calibrated Transition): Decayed beta in [1.0, 1.5] as promising neighborhoods emerged.
   * Week 3 (Peak Exploitation): Low beta = 0.8 or transitioned to Expected Improvement.
2. Expected Improvement (EI):
   Measures the expectation of improvement over the current best observed value. We transitioned Function 5 (Chemical Yield) to EI in Week 2, immediately driving output from 1088.86 to 3044.75.

### 4.3 High-Dimensional Optimization: Hybrid Quasi-Random Sobol Sampling
In high dimensions (Function 7 in 6D and Function 8 in 8D), acquisition surfaces are riddled with local minima. Standard gradient ascent frequently traps optimizers in sub-optimal basins. We implemented a Hybrid Sobol Quasi-Random Initialisation:
* Generate 4,096 scrambled, low-discrepancy Sobol sequence candidate vectors across [0, 1]^d.
* Evaluate acquisition scores alpha(x) across the Sobol grid.
* Seed multi-start L-BFGS-B local optimizers at the top 10 candidates to converge to exact local peaks.

### 4.4 Critical Evaluation of Alternative ML Techniques

#### Simple Linear & Logistic Regressions
Fitting standard linear models (y = w^T x + b) to these black-box functions severely violates fundamental statistical assumptions:
* Linearity of Response Surface: Function 5 is unimodal and concave; a linear model predicts infinite yield toward domain boundaries. Function 1 features flat 0.0 plateaus with localized spikes; a linear hyperplane fits a meaningless diagonal slope.
* Homoscedasticity: Error variance in Functions 1 and 4 is near zero across flat regions but spikes sharply near active peaks, violating constant error variance.
* Feature Independence & Additivity: In Function 8 (8D), parameters interact non-linearly; achieving high output (> 9.5) requires features x0, x1, x2, x3 to be simultaneously small (<= 0.05). Linear models cannot capture higher-order multiplicative interactions.

#### Support Vector Machines (SVMs)
* Classification & Feasibility Filtering: Support Vector Classifiers (SVC) equipped with Radial Basis Function (RBF) kernels serve as effective non-linear feasibility boundaries. In Function 1, an SVC separates the active contamination zone (y < 0) from background noise (y = 0). In Function 6, a soft-margin SVC with slack variables absorbs subjective taster noise to classify acceptable vs. rejected recipe formulations.
* Limitations of Support Vector Regression (SVR): While SVR models non-linear response surfaces via the kernel trick, it only produces deterministic point predictions (y_hat). SVR lacks a posterior distribution and cannot estimate epistemic uncertainty (sigma(x)), making it incapable of driving Bayesian acquisition functions (UCB/EI) for active learning.


---

## 5. Master Weekly Iteration Log & Performance Milestones

* Week 1 (Baseline Exploration):
  - Function 6 established NEW GLOBAL MAXIMUM: -0.664287 (vs baseline -0.714265) by constraining x4 <= 0.025.
  - Function 7 refined peak to 1.364970.
  - Mean Absolute Error (MAE) across surrogates: 1.7946.

* Week 2 (Targeted Probing & Peak Exploitation):
  - Function 4 achieved NEW GLOBAL MAXIMUM: -3.857021 (vs baseline -4.025542) in central warehouse hyper-volume.
  - Function 5 achieved MASSIVE BREAKTHROUGH: 3044.749281 (nearly 3x prior 1088.86 record) via Expected Improvement.
  - Function 1 boundary probe (-1.43e-124) proved exponential decay away from source.

* Week 3 (Consolidation & Peak Exploitation Submissions):
  - Function 1: 0.660114-0.674026 (Pinpointing centroid between known active detections)
  - Function 2: 0.691327-0.960000 (Exploiting interior log-likelihood ridge)
  - Function 3: 0.640000-0.650000-0.098696 (Constraining x2 < 0.10 to eliminate therapeutic side-effects)
  - Function 4: 0.490000-0.470000-0.390610-0.260000 (Refining central hyper-volume peak)
  - Function 5: 0.200000-0.890000-0.999999-0.999999 (Aggressive EI exploitation along upper yield ridge)
  - Function 6: 0.690000-0.150000-0.710000-0.730000-0.010000 (Locking x4 = 0.01 to maintain record taste score)
  - Function 7: 0.054000-0.488000-0.251000-0.214000-0.425000-0.738000 (Trust-region refinement around 1.365 peak)
  - Function 8: 0.035000-0.045000-0.030000-0.045000-0.425000-0.790000-0.475000-0.885000 (Preserving 4D corner)

---

## 6. Datasheet for the Dataset

### 6.1 Motivation
* Purpose: Provide standardized benchmark datasets simulating physical and computational black-box optimization challenges.
* Creators: Imperial College London / Emeritus Machine Learning & AI Programme Faculty.

### 6.2 Composition
* Instances: 8 distinct datasets corresponding to Functions 1 through 8. Initial observation counts:
  N_0 = 10 (F1, F2), N_0 = 15 (F3), N_0 = 30 (F4), N_0 = 20 (F5, F6), N_0 = 30 (F7), N_0 = 40 (F8).
* Data Fields: Feature vectors x in [0, 1]^d and scalar target response y in R.
* Confidentiality & Noise: Internal analytical functions are undisclosed. Data contains heteroscedastic noise and penalty clipping.

### 6.3 Preprocessing & Storage
* File Formats: Stored as binary NumPy arrays (initial_inputs.npy, initial_outputs.npy).
* Accumulation: In accordance with the BBO protocol, newly returned weekly portal evaluations are appended row-wise.

---

## 7. Model Card: Bayesian Optimization Surrogate Engine

### 7.1 Model Details
* Architecture: Gaussian Process Regressor (scikit-learn).
* Kernel Specification: Anisotropic Matern 5/2 with Automatic Relevance Determination (ARD) plus White Noise Kernel.
* Optimizer: L-BFGS-B with 25 random restarts for marginal log-likelihood maximization.

### 7.2 Intended Use
* Primary Task: Active query point selection for expensive black-box function maximization.
* Out-of-Scope Use: Extrapolation far beyond the normalized unit hypercube [0, 1]^d, or real-time sub-millisecond inference on streaming sensor data.

### 7.3 Performance Metrics & Quantitative Residuals
Surrogate accuracy is benchmarked weekly via prediction residuals e = |y_actual - mu| and Mean Absolute Error (MAE):
* Week 1 Mean Absolute Error: 1.7946 across all models.
* Key Residual Insights: Surrogates demonstrated sub-0.05 residual accuracy on Functions 1, 3, and 7, while highlighting rapid gradient escalation on Function 5 (3044.75 actual vs. 1180.83 predicted), triggering adaptive trust-region recalibration.

### 7.4 Model Limitations & Failure Modes
1. Curse of Dimensionality: In 8D (Function 8), sample density remains sparse relative to volume (2^8 = 256 hyper-corners), risking false correlations across uninformative features.
2. Cubic Computational Complexity: GP inversion scales as O(N^3), which is efficient for N <= 100 but requires sparse variational approximations if scaled to thousands of evaluations.
3. Severe Non-Stationarity: Surfaces with discontinuous penalty drops (Function 1 and Function 6) can violate the stationary covariance assumption, requiring conservative length-scale bounds.

---

## 8. Repository Structure & Reproducibility Guide

### 8.1 Directory Layout

```text
bbo-capstone/
|-- README.md                                <- Complete Project Documentation & Rubric Deliverable
|-- capstone_bayesian_optimization.py        <- Executable, modular Python BO Pipeline
|-- capstone_bayesian_optimization.ipynb     <- Interactive Google Colab / Jupyter Notebook
|-- capstone_bbo_master_tracker.xlsx         <- Excel Master Tracker with Formulas & Logs
|-- requirements.txt                         <- Reproducible Python Dependencies
|-- .gitignore                               <- Git Ignore Rules for Python / Jupyter
+-- LICENSE                                  <- MIT Open-Source License
```

### 8.2 Installation & Setup
1. Clone the repository:
   git clone <repository-url>
   cd bbo-capstone
2. Install dependencies:
   pip install -r requirements.txt

---

## 9. Author & Programme Information
* Programme: Imperial College London - Professional Certificate in Machine Learning & Artificial Intelligence
* Module: Stage 2 Capstone Component 14.2 (Documentation & Portfolio Deliverable)
* Author: Capstone Project Participant
* Date: October 2026
