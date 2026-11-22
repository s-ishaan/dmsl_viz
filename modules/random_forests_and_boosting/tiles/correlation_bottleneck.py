"""
Tile 4: The Correlation Bottleneck & Bagging Variance Formula
Focus: Slide 303 - Mathematical and empirical exploration of the variance of an average of i.d. trees:
Var((1/B) * ∑ T_b) = ρ * σ² + ((1 - ρ)/B) * σ²
Demonstrates why tree correlation ρ creates an irreducible variance floor that limits bagging,
setting the exact motivation for Random Forests.
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
            st.markdown("#### 🎛️ Variance Decomposition Knobs")

            rho_corr = st.slider(
                "Pairwise Tree Correlation ρ",
                min_value=0.0,
                max_value=0.80,
                value=0.35,
                step=0.05,
                key="cb_rho",
                help="Average correlation between predictions of any pair of constituent trees. Standard bagging typically yields ρ ≈ 0.3 - 0.5.",
            )

            single_var = st.slider(
                "Single Tree Variance σ²",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.25,
                key="cb_var",
                help="Estimation variance of an individual unpruned decision tree.",
            )

            b_max = st.slider(
                "Ensemble Size B",
                min_value=10,
                max_value=250,
                value=100,
                step=10,
                key="cb_b_max",
                help="Number of trees averaged in the ensemble.",
            )

            with st.expander("🎲 Monte Carlo Simulation", expanded=False):
                run_sim = st.checkbox("Run Empirical Simulation (N=1500 trials)", value=True, key="cb_run_sim")
                sim_trials = 1500

    # Theoretical variance calculations across B
    b_axis = np.arange(1, b_max + 1)
    floor_term = rho_corr * single_var
    reducible_curve = ((1.0 - rho_corr) / b_axis) * single_var
    total_var_curve = floor_term + reducible_curve

    curr_floor = floor_term
    curr_reducible = ((1.0 - rho_corr) / b_max) * single_var
    curr_total = curr_floor + curr_reducible
    pct_eliminated = ((single_var - curr_total) / single_var) * 100.0

    # Optional Monte Carlo simulation of equicorrelated normal estimators
    # T_b = sqrt(ρ) * Z_0 + sqrt(1 - ρ) * Z_b, where Z_0, Z_b ~ N(0, σ²)
    sim_variances = []
    if run_sim:
        rng = np.random.RandomState(42)
        z0 = rng.randn(sim_trials, 1) * np.sqrt(single_var)
        b_samples_eval = np.unique(np.linspace(1, b_max, min(30, b_max), dtype=int))
        for b_val in b_samples_eval:
            zb = rng.randn(sim_trials, b_val) * np.sqrt(single_var)
            tb_mat = np.sqrt(rho_corr) * z0 + np.sqrt(max(0.0, 1.0 - rho_corr)) * zb
            ens_means = np.mean(tb_mat, axis=1)
            sim_variances.append(float(np.var(ens_means)))

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Irreducible Floor ρσ²", f"{curr_floor:.3f}", help="Floor that cannot be eliminated by simply adding more trees.")
        m2.metric("Reducible Term", f"{curr_reducible:.3f}", help="Component shrinking to 0 as B -> ∞")
        m3.metric("Ensemble Var at B", f"{curr_total:.3f}", delta=f"-{pct_eliminated:.1f}% vs single", delta_color="inverse")
        m4.metric("Ideal i.i.d. Floor (ρ=0)", "0.000", help="If trees were completely independent (Random Forests goal).")

        tab_decomp, tab_sim, tab_compare = st.tabs([
            "📉 Variance Decomposition Curve (Slide 303)",
            "🎲 Empirical Distribution Comparison",
            "⚔️ Bagging vs Boosting Paradigm",
        ])

        with tab_decomp:
            st.markdown(r"##### 📉 Variance Decomposition: $\rho\sigma^2$ (Irreducible Floor) + $\frac{1-\rho}{B}\sigma^2$ (Reducible)")

            fig_decomp = go.Figure()

            # Shaded Area: Irreducible floor ρσ²
            fig_decomp.add_trace(
                go.Scatter(
                    x=b_axis,
                    y=[floor_term] * len(b_axis),
                    mode="lines",
                    name=f"Irreducible Floor ρσ² ({floor_term:.3f})",
                    line=dict(color="#EF4444", width=2.5, dash="dash"),
                    hovertemplate="Floor: %{y:.3f}<extra></extra>",
                )
            )

            # Total Variance Curve
            fig_decomp.add_trace(
                go.Scatter(
                    x=b_axis,
                    y=total_var_curve,
                    mode="lines",
                    name="Total Ensemble Variance",
                    line=dict(color=COLOR_HYPERPLANE, width=3.5),
                    fill="tonexty",
                    fillcolor="rgba(16, 185, 129, 0.12)",
                    hovertemplate="B=%{x}: Total Var = %{y:.3f}<extra></extra>",
                )
            )

            # Ideal i.i.d. curve (ρ = 0)
            ideal_curve = single_var / b_axis
            fig_decomp.add_trace(
                go.Scatter(
                    x=b_axis,
                    y=ideal_curve,
                    mode="lines",
                    name="Ideal i.i.d. Bound (ρ = 0 → 0)",
                    line=dict(color="#38BDF8", width=1.8, dash="dot"),
                    hovertemplate="B=%{x}: i.i.d. Var = %{y:.3f}<extra></extra>",
                )
            )

            # Empirical simulation overlay
            if run_sim:
                fig_decomp.add_trace(
                    go.Scatter(
                        x=b_samples_eval,
                        y=sim_variances,
                        mode="markers",
                        name="Monte Carlo Simulation",
                        marker=dict(color="#F59E0B", size=6, symbol="circle-open", line=dict(width=1.5)),
                        hovertemplate="Sim B=%{x}: Var = %{y:.3f}<extra></extra>",
                    )
                )

            fig_decomp.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Number of Ensemble Trees B", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Variance of Averaged Prediction", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
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

            st.plotly_chart(fig_decomp, width="stretch", key="cb_fig_decomp")

        with tab_sim:
            st.markdown("##### 🎲 Estimator Spread: Single Tree vs Correlated Bagging vs De-correlated Forest")

            # Simulate 3 regimes at fixed B=50
            rng = np.random.RandomState(42)
            b_fix = min(50, b_max)
            z_shared = rng.randn(2000, 1) * np.sqrt(single_var)

            # 1. Single Tree (B = 1)
            dist_single = rng.randn(2000) * np.sqrt(single_var)

            # 2. Correlated Bagging (ρ = rho_corr)
            zb_bag = rng.randn(2000, b_fix) * np.sqrt(single_var)
            dist_bag = np.mean(np.sqrt(rho_corr) * z_shared + np.sqrt(max(0.0, 1.0 - rho_corr)) * zb_bag, axis=1)

            # 3. De-correlated Random Forest (ρ_rf = 0.05)
            zb_rf = rng.randn(2000, b_fix) * np.sqrt(single_var)
            dist_rf = np.mean(np.sqrt(0.05) * z_shared + np.sqrt(0.95) * zb_rf, axis=1)

            fig_dist = go.Figure()

            fig_dist.add_trace(
                go.Histogram(
                    x=dist_single,
                    name=f"Single Tree (Var = {np.var(dist_single):.2f})",
                    opacity=0.45,
                    marker=dict(color="#EF4444"),
                    nbinsx=40,
                )
            )

            fig_dist.add_trace(
                go.Histogram(
                    x=dist_bag,
                    name=f"Standard Bagging ρ={rho_corr:.2f} (Var = {np.var(dist_bag):.2f})",
                    opacity=0.6,
                    marker=dict(color="#F59E0B"),
                    nbinsx=40,
                )
            )

            fig_dist.add_trace(
                go.Histogram(
                    x=dist_rf,
                    name=f"Random Forest ρ=0.05 (Var = {np.var(dist_rf):.2f})",
                    opacity=0.65,
                    marker=dict(color=COLOR_HYPERPLANE),
                    nbinsx=40,
                )
            )

            fig_dist.update_layout(
                barmode="overlay",
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Ensemble Prediction Deviation", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Frequency Count (2000 Trials)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
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

            st.plotly_chart(fig_dist, width="stretch", key="cb_fig_dist")

        with tab_compare:
            st.markdown("##### ⚔️ Algorithmic Comparison: Bagging vs Boosting (Slide 305)")

            st.markdown(
                r"""
                | Dimension | 🎲 Bootstrap Aggregation (Bagging) | 🚀 Adaptive Boosting (MART / AdaBoost) |
                |---|---|---|
                | **Primary Objective** | **Variance Reduction** (Averages noisy, unbiased deep trees) | **Bias Reduction** (Sequentially fits residuals of shallow trees) |
                | **Tree Generation** | **Parallel & Independent**: $B$ trees grown on separate bootstrap resamples | **Sequential & Dependent**: Tree $m$ actively repairs errors of tree $m-1$ |
                | **Identically Distributed?**| **Yes (i.d.)**: Each bootstrap tree has identical distribution $\mathbb{E}[T_b] = \mathbb{E}[T]$ | **No**: Trees adaptively shift distribution via observation reweighting |
                | **Voting Mechanism** | **One Member, One Vote**: Uniform simple average $(1/B) \sum T_b$ | **Weighted Committee**: Proportional to accuracy $\sum \alpha_m G_m(x)$ or shrinkage $\nu$ |
                | **Data Resampling** | Bootstrap sampling with replacement (random $\{0, 1, 2, \dots\}$ counts) | Uses all data points but assigns continuous sample weights $w_i$ |
                | **Overfitting Risk** | **Low**: Adding more trees $B \to \infty$ never causes overfitting | **Moderate**: Too large $M$ without shrinkage $\nu$ overfits noise |
                | **Limiting Constraint** | **Pairwise Correlation $\rho$**: $\text{Var} \to \rho \sigma^2$ (solved by Random Forests) | Sensitive to severe label noise and outliers |
                """
            )

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The Bagging Variance Equation (Slide 303)", expanded=False):
        st.markdown(
            r"""
            **Variance of Averaged Random Variables (Slide 303)**:
            - Let $T_1, \dots, T_B$ be identically distributed random estimators, each with variance $\text{Var}(T_b) = \sigma^2$ and positive pairwise correlation $\text{Corr}(T_j, T_k) = \rho$ for $j \ne k$.
            - The variance of their simple average $\bar{T} = \frac{1}{B} \sum_{b=1}^B T_b$ is derived as:
              $$\begin{aligned}
              \text{Var}(\bar{T}) &= \frac{1}{B^2} \sum_{b=1}^B \text{Var}(T_b) + \frac{1}{B^2} \sum_{j \ne k} \text{Cov}(T_j, T_k) \\
              &= \frac{1}{B^2} (B \sigma^2) + \frac{1}{B^2} B(B - 1) \rho \sigma^2 \\
              &= \frac{\sigma^2}{B} + \frac{B - 1}{B} \rho \sigma^2 \\
              &= \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
              \end{aligned}$$
            - **Key Theoretical Insight**:
              - As $B \to \infty$, the second term $\frac{1 - \rho}{B} \sigma^2 \to 0$.
              - The first term $\rho \sigma^2$ **never disappears**!
              - Therefore, the pairwise correlation $\rho$ between bagged trees strictly bounds the benefit of averaging.
              - To achieve superior variance reduction, one must **de-correlate** the trees so that $\rho \to 0$. This is the exact breakthrough of Leo Breiman's **Random Forests** (Slide 306).
            """
        )
