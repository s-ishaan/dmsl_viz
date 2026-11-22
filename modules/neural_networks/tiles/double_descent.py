"""
Tile 5: Over-parametrization & The Double Descent Phenomenon
Focus: Slides 322-323 - Belkin, Hsu, Ma, and Mandal (2019) PNAS.
Recreating the double descent curve bridging the classical bias-variance trade-off
and the modern over-parameterized interpolating regime (p >> N).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Double Descent Knobs")

            cap_ratio = st.slider(
                "Model Capacity Ratio (p / N)",
                min_value=0.15,
                max_value=3.5,
                value=1.0,
                step=0.05,
                key="dd_ratio",
                help="Ratio of parameters/features p to training samples N. Ratio = 1.0 is the exact Interpolation Threshold.",
            )

            if cap_ratio < 0.85:
                regime_tag = "Classical Under-Parameterized Regime"
                regime_color = "#38BDF8"
            elif cap_ratio <= 1.25:
                regime_tag = "Interpolation Threshold (Variance Explosion)"
                regime_color = "#EF4444"
            else:
                regime_tag = "Modern Over-Parameterized Interpolating Regime"
                regime_color = "#10B981"

            st.markdown(f"**Current State**: :{regime_color}[{regime_tag}]")

            n_samples = st.slider(
                "Training Observations N",
                min_value=20,
                max_value=60,
                value=35,
                step=5,
                key="dd_n",
            )

            noise_lvl = st.slider(
                "Observation Noise σ_ε",
                min_value=0.05,
                max_value=0.45,
                value=0.20,
                step=0.05,
                key="dd_noise",
            )

            with st.expander("⚙️ Theoretical Simulation Settings", expanded=False):
                rbf_gamma = st.slider("Kernel / Feature Bandwidth γ", 0.5, 5.0, 2.0, 0.5, key="dd_gamma")
                n_test = 200

    # Generate synthetic 1D ground truth: f*(x) = sin(2 pi x)
    rng = np.random.RandomState(42)
    x_dense = np.linspace(0.0, 1.0, 300)
    y_dense_clean = np.sin(2.0 * np.pi * x_dense)

    x_train = np.sort(rng.uniform(0.05, 0.95, n_samples))
    y_train = np.sin(2.0 * np.pi * x_train) + rng.randn(n_samples) * noise_lvl

    x_test = np.linspace(0.02, 0.98, n_test)
    y_test_clean = np.sin(2.0 * np.pi * x_test)
    y_test = y_test_clean + rng.randn(n_test) * noise_lvl

    # Parametric feature expansion: Random Fourier / RBF basis features of dimension p
    # Capacity sweep over ratios 0.1 to 3.5
    ratios_axis = np.linspace(0.15, 3.5, 45)
    train_risks = []
    test_risks = []

    # Fixed random projection frequencies
    max_p = int(3.5 * n_samples)
    W_proj = rng.randn(max_p) * rbf_gamma
    b_proj = rng.uniform(0, 2.0 * np.pi, max_p)

    def get_features(X_in, p_dim):
        # Cosine features: shape (len(X), p_dim)
        X_col = X_in.reshape(-1, 1)
        Phi = np.cos(X_col * W_proj[:p_dim] + b_proj[:p_dim]) * np.sqrt(2.0 / p_dim)
        return Phi

    for r in ratios_axis:
        p_dim = max(2, int(r * n_samples))
        Phi_tr = get_features(x_train, p_dim)
        Phi_te = get_features(x_test, p_dim)

        # Minimum-norm least squares solution via pseudoinverse (pinv)
        # When p < N: standard OLS. When p >= N: unique minimum L2-norm interpolator.
        # Add tiny eps=1e-8 for numerical stability exactly at threshold
        if p_dim <= n_samples:
            w_hat = np.linalg.lstsq(Phi_tr, y_train, rcond=1e-5)[0]
        else:
            w_hat = np.linalg.pinv(Phi_tr, rcond=1e-5) @ y_train

        y_pred_tr = Phi_tr @ w_hat
        y_pred_te = Phi_te @ w_hat

        tr_mse = float(np.mean((y_train - y_pred_tr) ** 2))
        te_mse = float(np.mean((y_test - y_pred_te) ** 2))

        # Clamp extreme visualization spikes at threshold
        train_risks.append(tr_mse)
        test_risks.append(min(12.0, te_mse))

    # Active model based on cap_ratio
    active_p = max(2, int(cap_ratio * n_samples))
    Phi_tr_act = get_features(x_train, active_p)
    Phi_dense_act = get_features(x_dense, active_p)
    Phi_te_act = get_features(x_test, active_p)

    if active_p <= n_samples:
        w_act = np.linalg.lstsq(Phi_tr_act, y_train, rcond=1e-5)[0]
    else:
        w_act = np.linalg.pinv(Phi_tr_act, rcond=1e-5) @ y_train

    y_dense_pred = Phi_dense_act @ w_act
    curr_train_mse = float(np.mean((y_train - Phi_tr_act @ w_act) ** 2))
    curr_test_mse = float(np.mean((y_test - Phi_te_act @ w_act) ** 2))
    norm_w = float(np.linalg.norm(w_act))

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capacity p / N", f"{cap_ratio:.2f} ({active_p} params)")
        m2.metric("Training Risk", f"{curr_train_mse:.4f}", help="Hits exactly 0 at p/N >= 1.0 (Interpolation Threshold)")
        m3.metric("Test Risk", f"{min(99.0, curr_test_mse):.3f}")
        m4.metric("Weight Norm ||w||", f"{norm_w:.2f}", help="Weights explode at p=N, then shrink in over-parameterized regime.")

        tab_double_descent, tab_curves, tab_theory = st.tabs([
            "📈 The Double Descent Curve (Recreating Slide 322)",
            "🌊 Fitted Functions Across 3 Regimes",
            "🏔️ Benign Non-Convexity & Strict Saddles (Slide 323)",
        ])

        with tab_double_descent:
            st.markdown("##### 📈 Risk vs Capacity: Classical Regime → Interpolation Threshold → Modern Regime (Slide 322)")

            fig_dd = go.Figure()

            # Shaded vertical band at threshold p=N
            fig_dd.add_vrect(
                x0=0.92,
                x1=1.08,
                fillcolor="rgba(239, 68, 68, 0.15)",
                line_width=1,
                line_dash="dot",
                line_color="#EF4444",
                annotation_text="Interpolation Threshold (p = N)",
                annotation_font_color="#EF4444",
                annotation_position="top left",
            )

            # Training Risk (Dashed Black/Slate)
            fig_dd.add_trace(
                go.Scatter(
                    x=ratios_axis,
                    y=train_risks,
                    mode="lines",
                    name="Training Risk (Hits 0 at Threshold)",
                    line=dict(color="#94A3B8", width=2.5, dash="dash"),
                    hovertemplate="p/N = %{x:.2f}: Train Risk = %{y:.4f}<extra></extra>",
                )
            )

            # Test Risk (Solid Emerald)
            fig_dd.add_trace(
                go.Scatter(
                    x=ratios_axis,
                    y=test_risks,
                    mode="lines",
                    name="Test Risk (Double Descent Curve)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.5),
                    hovertemplate="p/N = %{x:.2f}: Test Risk = %{y:.3f}<extra></extra>",
                )
            )

            # Current capacity position marker
            fig_dd.add_vline(
                x=cap_ratio,
                line_width=2.0,
                line_color="#FFFFFF",
                annotation_text=f"Active: p/N = {cap_ratio:.2f}",
                annotation_font_color="#FFFFFF",
                annotation_position="bottom right",
            )

            fig_dd.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Model Capacity (Parameters p / Observations N)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Mean Squared Error Risk", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    range=[-0.05, 3.5],
                ),
                margin=dict(l=40, r=20, t=15, b=85),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=9.5),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_dd, width="stretch", key="dd_fig_dd")

        with tab_curves:
            st.markdown(f"##### 🌊 Fitted Prediction Function: {regime_tag}")

            fig_fit = go.Figure()

            # True underlying function
            fig_fit.add_trace(
                go.Scatter(
                    x=x_dense,
                    y=y_dense_clean,
                    mode="lines",
                    name="True Signal f*(x)",
                    line=dict(color="#64748B", width=2, dash="dash"),
                    hovertemplate="True y: %{y:.3f}<extra></extra>",
                )
            )

            # Training points
            fig_fit.add_trace(
                go.Scatter(
                    x=x_train,
                    y=y_train,
                    mode="markers",
                    name=f"Training Points (N = {n_samples})",
                    marker=dict(color="#38BDF8", size=7, line=dict(color="#0F172A", width=1)),
                    hovertemplate="x: %{x:.2f}<br>y: %{y:.3f}<extra></extra>",
                )
            )

            # Fitted model
            fig_fit.add_trace(
                go.Scatter(
                    x=x_dense,
                    y=y_dense_pred,
                    mode="lines",
                    name=f"Fitted Model (p = {active_p})",
                    line=dict(color=regime_color, width=3.2),
                    hovertemplate="Pred y: %{y:.3f}<extra></extra>",
                )
            )

            fig_fit.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Input Coordinate x", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Response y", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    range=[-2.2, 2.2],
                ),
                margin=dict(l=40, r=20, t=15, b=85),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=9.5),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_fit, width="stretch", key="dd_fig_fit")

        with tab_theory:
            st.markdown(
                r"""
                ##### 🏔️ The Three Regimes of Double Descent (Belkin et al., 2019)
                
                1. **Classical "Under-Parameterized" Regime ($p < N$)**:
                   - Model complexity is constrained by sample size.
                   - Exhibits the classical U-shaped curve: increasing capacity reduces bias until variance begins to dominate.
                
                2. **The Interpolation Threshold ($p \approx N$)**:
                   - Exactly enough parameters to memorize every noisy training sample ($\text{Training Error} = 0$).
                   - The system of equations is square: $(\Phi^T \Phi)$ is ill-conditioned with near-zero singular values.
                   - Parameter norm $\|w\|_2$ explodes, causing massive oscillations between data points and catastrophic test error spikes!
                
                3. **Modern "Over-Parameterized" Regime ($p \gg N$)**:
                   - Vastly more parameters than data points (infinite interpolating solutions exist).
                   - First-order optimizers (SGD) implicitly regularize toward the **minimum $\ell_2$ norm interpolant**:
                     $$\min \|w\|_2 \quad \text{subject to } \Phi w = y$$
                   - As $p \to \infty$, the space of zero-training-loss solutions becomes so rich that the minimum-norm solution is ultra-smooth, driving test error back down!
                """
            )

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Benign Non-Convexity (Slide 323)", expanded=False):
        st.markdown(
            r"""
            **Benign Non-Convexity in Deep Learning (Slide 323)**:
            - **Historical dogma**: People favored strictly convex optimization problems (such as SVMs) because global optimality is mathematically guaranteed.
            - **Modern empirical reality**: Deep neural networks optimize highly non-convex loss landscapes, yet simple gradient methods find outstanding solutions almost every time.
            - **The Three Pillars of Benign Non-Convexity**:
              1. **All local minima are global (or nearly global) minima**: In high dimensions, local minima with high loss values vanish; virtually all valleys share similar low risk.
              2. **All saddle points are strict saddle points**: A strict saddle has at least one direction with strictly negative curvature ($\lambda_{\min}(\nabla^2 \mathcal{E}) < 0$). Stochastic perturbations and momentum allow optimizers to escape saddles effortlessly.
              3. **Smart initializations (He, Xavier)**: Place $\theta_0$ in the basin of attraction of a favorable global minimizer.
            """
        )
