"""
Tile 1: MART / Gradient Tree Boosting Dynamics
Focus: Algorithm 10.3 - Stepwise steepest descent in function space.
Fitting regression trees sequentially to pseudo-residuals r_im = y_i - f_{m-1}(x_i),
showing how piecewise constant updates aggregate into complex non-linear approximations.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ MART Boosting Knobs")

            total_m = st.slider(
                "Total Boosting Steps M",
                min_value=5,
                max_value=60,
                value=25,
                step=5,
                key="mart_total_m",
                help="Total number of sequential additive trees to fit.",
            )

            current_m = st.slider(
                "Active Iteration Step m",
                min_value=1,
                max_value=total_m,
                value=min(8, total_m),
                step=1,
                key="mart_current_m",
                help="Step forward/backward to inspect residual fitting at iteration m.",
            )

            max_depth = st.slider(
                "Base Tree Max Depth (J = 2^depth)",
                min_value=1,
                max_value=4,
                value=2,
                step=1,
                key="mart_depth",
                help="Depth of each base CART tree. Depth 1 = Stumps (additive), Depth 2 = 4 terminal regions.",
            )

            shrinkage = st.slider(
                "Shrinkage Rate ν (Learning Rate)",
                min_value=0.05,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="mart_shrinkage",
                help="Scales each tree's contribution: f_m(x) = f_{m-1}(x) + ν * T_m(x).",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Count N", 40, 140, 75, 5, key="mart_n")
                noise_lvl = st.slider("Noise σ", 0.05, 0.40, 0.18, 0.02, key="mart_noise")
                target_curve = st.selectbox(
                    "Target Signal Function",
                    ["Multi-Frequency Wave", "Step Discontinuity", "Non-linear Polynomial"],
                    index=0,
                    key="mart_target_type",
                )

    # Generate synthetic 1D regression data
    rng = np.random.RandomState(42)
    x_dense = np.linspace(0.0, 1.0, 300)

    if target_curve == "Multi-Frequency Wave":
        y_dense_clean = np.sin(4.0 * x_dense) + 0.45 * np.cos(9.0 * x_dense)
    elif target_curve == "Step Discontinuity":
        y_dense_clean = np.where(x_dense < 0.35, -0.8, np.where(x_dense < 0.7, 0.9, -0.3))
    else:
        y_dense_clean = 2.5 * (x_dense - 0.5) ** 3 - 1.2 * (x_dense - 0.5) + 0.2

    x_train = np.sort(rng.uniform(0.02, 0.98, n_samples))
    if target_curve == "Multi-Frequency Wave":
        y_train_clean = np.sin(4.0 * x_train) + 0.45 * np.cos(9.0 * x_train)
    elif target_curve == "Step Discontinuity":
        y_train_clean = np.where(x_train < 0.35, -0.8, np.where(x_train < 0.7, 0.9, -0.3))
    else:
        y_train_clean = 2.5 * (x_train - 0.5) ** 3 - 1.2 * (x_train - 0.5) + 0.2

    y_train = y_train_clean + rng.randn(n_samples) * noise_lvl

    # Algorithm 10.3 implementation:
    # 1. Initialize f_0(x) = y_bar
    f0 = np.mean(y_train)
    f_train = np.full(n_samples, f0)
    f_dense = np.full(len(x_dense), f0)

    trees = []
    residuals_history = []
    train_mse_history = []
    tree_preds_history = []
    f_train_history = [f_train.copy()]
    f_dense_history = [f_dense.copy()]

    initial_mse = float(np.mean((y_train - f_train) ** 2))
    train_mse_history.append(initial_mse)

    for m in range(1, total_m + 1):
        # Line 2(a): r_im = y_i - f_{m-1}(x_i)
        r_m = y_train - f_train
        residuals_history.append(r_m.copy())

        # Line 2(b)-(c): Fit CART to residuals
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=m)
        tree.fit(x_train.reshape(-1, 1), r_m)
        trees.append(tree)

        # Predictions of m-th tree
        tree_pred_train = tree.predict(x_train.reshape(-1, 1))
        tree_pred_dense = tree.predict(x_dense.reshape(-1, 1))
        tree_preds_history.append(tree_pred_dense)

        # Line 2(d): Update f_m(x) = f_{m-1}(x) + ν * T_m(x)
        f_train += shrinkage * tree_pred_train
        f_dense += shrinkage * tree_pred_dense

        f_train_history.append(f_train.copy())
        f_dense_history.append(f_dense.copy())

        mse = float(np.mean((y_train - f_train) ** 2))
        train_mse_history.append(mse)

    # Active step quantities (1-indexed)
    active_residuals = residuals_history[current_m - 1]
    active_tree_pred = tree_preds_history[current_m - 1]
    active_f_dense = f_dense_history[current_m]
    active_f_train = f_train_history[current_m]
    curr_mse = train_mse_history[current_m]
    max_res = float(np.max(np.abs(active_residuals)))

    with col_plot:
        # 4-Column Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Step m", f"{current_m} / {total_m}")
        m2.metric("Train MSE", f"{curr_mse:.4f}", delta=f"{curr_mse - initial_mse:.4f}", delta_color="inverse")
        m3.metric("Max |Residual|", f"{max_res:.3f}")
        m4.metric("Shrinkage ν", f"{shrinkage:.2f}")

        tab_model, tab_residuals, tab_loss = st.tabs([
            "🎯 Cumulative Ensemble Model f_m(x)",
            "📈 Residuals & Current Tree Fit T_m(x)",
            "📉 Stepwise Loss Descent",
        ])

        with tab_model:
            st.markdown(f"##### 🎯 Cumulative Prediction f_{current_m}(x) vs True Signal (M = {current_m})")

            fig_model = go.Figure()

            # True Underlying Curve
            fig_model.add_trace(
                go.Scatter(
                    x=x_dense,
                    y=y_dense_clean,
                    mode="lines",
                    name="True Signal f*(x)",
                    line=dict(color="#64748B", width=2, dash="dash"),
                    hovertemplate="x: %{x:.2f}<br>True y: %{y:.3f}<extra></extra>",
                )
            )

            # Training observations
            fig_model.add_trace(
                go.Scatter(
                    x=x_train,
                    y=y_train,
                    mode="markers",
                    name="Observed Training y_i",
                    marker=dict(color="#38BDF8", size=7, opacity=0.85, line=dict(color="#0F172A", width=1)),
                    hovertemplate="x: %{x:.2f}<br>Obs y: %{y:.3f}<extra></extra>",
                )
            )

            # Initial baseline f_0
            fig_model.add_trace(
                go.Scatter(
                    x=x_dense,
                    y=[f0] * len(x_dense),
                    mode="lines",
                    name=f"Baseline f₀ = ȳ ({f0:.2f})",
                    line=dict(color="#F59E0B", width=1.5, dash="dot"),
                    hovertemplate="Baseline: %{y:.3f}<extra></extra>",
                )
            )

            # Cumulative MART Model f_m(x)
            fig_model.add_trace(
                go.Scatter(
                    x=x_dense,
                    y=active_f_dense,
                    mode="lines",
                    name=f"MART f_{current_m}(x)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.5),
                    hovertemplate="x: %{x:.2f}<br>MART: %{y:.3f}<extra></extra>",
                )
            )

            fig_model.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Input Feature x", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Response y & Prediction f(x)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_model, width="stretch", key="mart_fig_model")

        with tab_residuals:
            st.markdown(f"##### 📈 Pseudo-Residuals r_{{i, {current_m}}} & Fitted CART Tree T_{current_m}(x)")

            fig_res = go.Figure()

            # Zero baseline
            fig_res.add_hline(y=0.0, line_width=1.2, line_dash="dash", line_color="#475569")

            # Residual stems (vertical lines to 0)
            for xi, ri in zip(x_train, active_residuals):
                fig_res.add_trace(
                    go.Scatter(
                        x=[xi, xi],
                        y=[0, ri],
                        mode="lines",
                        line=dict(color="rgba(244, 63, 94, 0.4)", width=1.5),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            # Residual points
            fig_res.add_trace(
                go.Scatter(
                    x=x_train,
                    y=active_residuals,
                    mode="markers",
                    name=f"Residuals r_{{i,{current_m}}}",
                    marker=dict(color="#F43F5E", size=7, symbol="diamond"),
                    hovertemplate="x: %{x:.2f}<br>Residual: %{y:.3f}<extra></extra>",
                )
            )

            # Fitted Tree prediction on residuals
            fig_res.add_trace(
                go.Scatter(
                    x=x_dense,
                    y=active_tree_pred,
                    mode="lines",
                    name=f"Fitted Tree T_{current_m}(x)",
                    line=dict(color="#F59E0B", width=2.8),
                    hovertemplate="x: %{x:.2f}<br>Tree Step: %{y:.3f}<extra></extra>",
                )
            )

            fig_res.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Input Feature x", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Residual r = y - f_{m-1}(x)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_res, width="stretch", key="mart_fig_res")

        with tab_loss:
            st.markdown("##### 📉 Empirical Loss Descent Across Boosting Iterations")

            fig_loss = go.Figure()

            steps_axis = list(range(0, total_m + 1))
            fig_loss.add_trace(
                go.Scatter(
                    x=steps_axis,
                    y=train_mse_history,
                    mode="lines+markers",
                    name="Training MSE",
                    line=dict(color=COLOR_HYPERPLANE, width=2.8),
                    marker=dict(size=5, color=COLOR_HYPERPLANE),
                    hovertemplate="Step %{x}: MSE = %{y:.4f}<extra></extra>",
                )
            )

            # Active step vertical line
            fig_loss.add_vline(
                x=current_m,
                line_width=1.5,
                line_dash="dash",
                line_color="#FFFFFF",
                annotation_text=f"m = {current_m}",
                annotation_font_color="#FFFFFF",
                annotation_position="top left",
            )

            fig_loss.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Boosting Iteration m", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Mean Squared Error L(y, f)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_loss, width="stretch", key="mart_fig_loss")

    # Mathematical formulation footer
    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Algorithm 10.3 (Gradient Tree Boosting)", expanded=False):
        st.markdown(
            r"""
            **Multiple Additive Regression Trees (MART)** formulates gradient descent directly in function space:
            
            1. **Initialization**: Start with a constant model $f_0(x) = \arg\min_\gamma \sum_{i=1}^N L(y_i, \gamma) = \bar{y}$ (for squared error loss).
            2. **For $m = 1, \dots, M$**:
               - Compute negative gradient pseudo-residuals:
                 $$r_{im} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f=f_{m-1}} = y_i - f_{m-1}(x_i)$$
               - Fit a regression tree $T_m$ to targets $r_{im}$, partitioning the space into $J$ terminal regions $\{R_{jm}\}_{j=1}^J$.
               - Compute optimal leaf adjustments:
                 $$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma) = \text{mean}(r_{im} \mid x_i \in R_{jm})$$
               - Update model with shrinkage $\nu$:
                 $$f_m(x) = f_{m-1}(x) + \nu \sum_{j=1}^J \gamma_{jm} I(x \in R_{jm})$$
            3. **Output**: Cumulative ensemble $\hat{f}(x) = f_M(x)$.
            """
        )
