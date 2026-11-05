# 📐 Statistical Learning Visualizer (DMSL Sandbox)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **An interactive, extensible visual laboratory for statistical learning theory.** Built with Python, Streamlit, and Plotly to provide real-time geometric and mathematical intuition for core machine learning models, optimization landscapes, and regularization mechanics.

---

## 🌟 Highlights

- **Intuition-First Interactivity**: Zero static plots. Every concept is accompanied by real-time reactive parameter sliders (penalties, bandwidths, pruning thresholds, degrees, and iteration steps) that instantly update decision boundaries, surfaces, and spectra.
- **Strict Mathematical Fidelity**: Direct visual implementations of foundational statistical concepts (Mercer's Theorem, Wolfe dual sparsity, Nadaraya-Watson estimators, cost-complexity pruning with the 1-SE rule, and AdaBoost.M1 sequential reweighting).
- **Modern Clean Analytics UX**: Dark slate visual design with high-contrast accessibility, zero title/legend collisions, and responsive full-block canvas sizing.
- **Extensible Architecture**: Built on a dynamic plugin-style module registry (`modules/registry.py`), allowing new lecture decks or algorithms to be plugged into the sidebar with zero changes to existing pages.

---

## 🔬 Core Learning Modules & Interactive Tiles

### 1. Support Vector Machines: Linear & Foundations
*Exploration of maximal margin classifiers, dual formulations, and soft margins.*
- **📐 Hyperplane Geometry**: Interactive normal vector $\beta^*$, signed orthogonal projections, and bias intercept $\beta_0$.
- **⚔️ Perceptron vs OLS vs SVM**: Comparative sensitivity to class balance, margin constraints, and masking.
- **📏 Hard-Margin Optimal Separating Hyperplane (OSH)**: Maximizing margin $M = 1/\|\beta\|$ with highlighted support vector halos ($\alpha_i > 0$).
- **📉 Soft-Margin & Slack ($\xi_i$)**: Visualizing margin slack penalties $\xi_i = \max(0, 1 - y_i f(x_i))$ and margin violations.
- **🎚️ Regularization Parameter $C$**: The trade-off between variance and bias across logarithmic scales ($C \in [10^{-2}, 10^3]$).
- **🚀 Feature Space Lift**: Lifting non-separable 2D concentric rings into 3D quadratic space $[x_1, x_2, x_1^2 + x_2^2]^T$ where a linear separating plane exists.

---

### 2. Support Vector Machines: Kernels, RKHS & Regularization
*Non-linear boundaries, Reproducing Kernel Hilbert Spaces, and surrogate loss functions.*
- **📐 Polynomial Kernel Dynamics**: Degree $d \in [1, 5]$ and intercept $c_0$ in $k(x, x') = (c_0 + \langle x, x'\rangle)^d$, plus an explicit 3D projection onto the 6D monomial basis $[x_1^2, x_2^2, \sqrt{2}x_1x_2]^T$ for $d=2$.
- **🌐 RBF Kernel & Localization**: Continuous log-scale $\gamma = 1/(2\sigma^2)$ demonstrating the smooth transition from global linear boundaries to localized overfitting island bubbles, paired with the radial decay profile $k(r)$.
- **📊 Localized Kernel Regression**: Replication of the Nadaraya-Watson estimator:
  $$\hat{f}(x_0) = \frac{\sum_{i=1}^N k_\lambda(x_0, x_i)y_i}{\sum_{i=1}^N k_\lambda(x_0, x_i)}$$
  comparing Epanechnikov quadratic windows with Gaussian kernels, featuring an interactive query probe $x_0$ and window width $\lambda$.
- **🔥 Gram Matrix & Mercer Spectrum**: $N \times N$ similarity matrix heatmap $K_{ij} = k(x_i, x_j) \succeq 0$ and Mercer eigenvalue decay plot measuring effective feature space dimensionality.
- **🎯 The Loss Function Arena**: Margin probe $yf \in [-3, 3]$ comparing SVM Hinge, Logistic Binomial Deviance, AdaBoost Exponential, Squared Error, and Huberized Hinge losses (recreating ESL Figs 10.4 & 12.4).
- **⚖️ SVM vs Kernel Logistic Regression**: Direct benchmark comparing sparse dual representations (SVM: only SVs have $\alpha_i > 0$) against dense representation (KLR: 100% of points receive non-zero weights).

---

### 3. Decision Trees, Pruning & AdaBoost Ensemble Learning
*Recursive partitioning, complexity regularization, and committee methods.*
- **🗺️ Recursive Binary Splitting**: Axis-aligned rectangular partitioning $\hat{f}(x) = \sum c_j I(x \in R_j)$ and full 3D isometric piecewise constant step surface visualization.
- **📐 Node Impurity Metrics**: Real-time comparison of Gini index $2p(1-p)$, Cross-Entropy, and 0-1 misclassification error, with decision space shaded by leaf purity.
- **✂️ Cost-Complexity Pruning**: 10-fold cross-validation error curves with $\pm 1 \text{SE}$ error bands against tree size $|T_\alpha|$ and penalty $\alpha$, supporting both the **One-Standard-Error ('se')** and **Minimum CV Error ('min')** rules.
- **⚡ Tree Instability & High Variance**: Demonstrating how greedy top-down splits cause minor data perturbations to cascade downward, resulting in drastic decision boundary shifts.
- **🚀 AdaBoost.M1 Dynamics**: Step-by-step scrubber through boosting iterations $m = 1 \dots M$:
  - Observation markers scaled proportionally to their sample weights $w_i^{(m)}$, illustrating how misclassified points swell in importance.
  - Active single-split decision stump $G_m(x)$.
  - Emergence of the cumulative non-linear committee boundary $G(x) = \text{sign}\left(\sum \alpha_m G_m(x)\right)$.
- **🏆 AdaBoost Power Benchmark**: Real-time convergence plot replicating the classic benchmark: Single Decision Stump ($err \approx 0.458$) vs 244-node Complex Tree ($err \approx 0.247$) vs AdaBoost of stumps ($err \to 0.058$).

---

## 🏗️ Software Architecture

```
dmsl_viz/
├── app.py                     # Main Streamlit application & layout entrypoint
├── requirements.txt           # Clean, cross-platform dependencies
├── run.sh                     # Local launcher script
├── .streamlit/
│   └── config.toml            # Custom dark analytics theme configuration
├── modules/
│   ├── registry.py            # Dynamic multi-page module registry
│   ├── svm_linear/            # Module 1: Linear SVM & OSH
│   │   ├── page.py
│   │   └── tiles/             # 6 isolated interactive tiles
│   ├── svm_kernels/           # Module 2: Kernels, RKHS & Loss Functions
│   │   ├── page.py
│   │   └── tiles/             # 6 isolated interactive tiles
│   └── trees_and_boosting/    # Module 3: Decision Trees & AdaBoost
│       ├── page.py
│       └── tiles/             # 6 isolated interactive tiles
└── utils/
    ├── datasets.py            # 2D & 3D synthetic dataset generators
    └── plotting.py            # Standardized Plotly layout engine & themes
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/s-ishaan/dmsl_viz.git
   cd dmsl_viz
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the visualizer:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Cloud (1-Click)

1. Fork or push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**, select your repository `dmsl_viz`, branch `main`, and main file path `app.py`.
4. Click **Deploy!** Your app will be live with a shareable public URL in under 2 minutes.

---

## 📚 Theoretical References

- Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer.
- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees (CART)*. CRC Press.
- Freund, Y., & Schapire, R. E. (1997). *A decision-theoretic generalization of on-line learning and an application to boosting*. Journal of Computer and System Sciences.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning (ESL)*. Springer.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
