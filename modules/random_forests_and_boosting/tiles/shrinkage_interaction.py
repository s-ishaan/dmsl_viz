"""
Tile 2: Shrinkage Factor ν & Interaction Depth J
Focus: Exploring the regularizing effect of learning rate shrinkage (f_m = f_{m-1} + ν * T_m)
and the interaction capacity controlled by constituent tree size J (interaction level <= J - 1).
Directly recreates the empirical deviance curves of Slides 295 & 297.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Shrinkage & Interaction Knobs")

            tree_size_j = st.selectbox(
                "Base Tree Size J (Leaves)",
                [2, 4, 6, 10],
                index=2,
                key="si_tree_size",
                help="Constituent tree size J. J=2 (stumps) captures only main effects. J=6 captures up to 5-factor interactions (rule of thumb).",
            )

            shrinkage_nu = st.slider(
                "Shrinkage Factor ν",
                min_value=0.02,
                max_value=1.0,
                value=0.1,
                step=0.02,
                key="si_nu",
                help="Learning rate scaling parameter ν. Lower values (ν <= 0.1) prevent late-stage deviance explosion.",
            )

            max_m = st.slider(
                "Max Boosting Iterations M",
                min_value=50,
                max_value=350,
                value=150,
                step=25,
                key="si_max_m",
                help="Number of additive boosting iterations to track.",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                dataset_type = st.selectbox(
                    "Topology",
                    ["Moons", "Concentric Circles", "XOR / Checkerboard"],
                    index=0,
                    key="si_dataset",
                )
                n_samples = st.slider("Sample Count N", 100, 350, 200, 25, key="si_n")
                noise_lvl = st.slider("Noise Level", 0.05, 0.40, 0.18, 0.02, key="si_noise")

    # Generate dataset
    X, y_svm = get_dataset(dataset_type, n_samples=n_samples, noise=noise_lvl, random_state=42)
    # Convert labels {-1, 1} to {0, 1} for binomial deviance
    y = np.where(y_svm == 1, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)

    # Convert tree size J to max_leaf_nodes / max_depth
    # J = 2 => depth 1 (stump); J = 4 => depth 2; J = 6 => max_leaf_nodes 6; J = 10 => max_leaf_nodes 10
    max_leaves = tree_size_j

    # Model 1: With chosen shrinkage ν
    gb_shrunk = GradientBoostingClassifier(
        n_estimators=max_m,
        learning_rate=shrinkage_nu,
        max_leaf_nodes=max_leaves,
        random_state=42,
    )
    gb_shrunk.fit(X_train, y_train)

    # Model 2: Without shrinkage (ν = 1.0)
    gb_noshrink = GradientBoostingClassifier(
        n_estimators=max_m,
        learning_rate=1.0,
        max_leaf_nodes=max_leaves,
        random_state=42,
    )
    gb_noshrink.fit(X_train, y_train)

    # Model 3: Stumps (J = 2) with shrinkage for comparison
    gb_stumps = GradientBoostingClassifier(
        n_estimators=max_m,
        learning_rate=shrinkage_nu,
        max_depth=1,
        random_state=42,
    )
    gb_stumps.fit(X_train, y_train)

    # Compute staged binomial deviance and 0-1 error curves
    def get_staged_metrics(model, X_eval, y_eval):
        deviances = []
        errors = []
        eps = 1e-12
        for proba in model.staged_predict_proba(X_eval):
            p1 = np.clip(proba[:, 1], eps, 1.0 - eps)
            # Binomial deviance = -2 * sum(y*log(p) + (1-y)*log(1-p)) / N
            dev = -2.0 * np.mean(y_eval * np.log(p1) + (1.0 - y_eval) * np.log(1.0 - p1))
            err = np.mean((p1 >= 0.5).astype(int) != y_eval)
            deviances.append(dev)
            errors.append(err)
        return deviances, errors

    test_dev_shrunk, test_err_shrunk = get_staged_metrics(gb_shrunk, X_test, y_test)
    test_dev_noshrink, test_err_noshrink = get_staged_metrics(gb_noshrink, X_test, y_test)
    test_dev_stumps, test_err_stumps = get_staged_metrics(gb_stumps, X_test, y_test)

    final_shrunk_err = test_err_shrunk[-1] * 100.0
    final_noshrink_err = test_err_noshrink[-1] * 100.0
    final_shrunk_dev = test_dev_shrunk[-1]
    final_noshrink_dev = test_dev_noshrink[-1]

    with col_plot:
        # Top KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tree Leaves J", f"{tree_size_j}", help=f"Max interaction order: {tree_size_j - 1}")
        m2.metric("Shrunk Test Error", f"{final_shrunk_err:.1f}%")
        m3.metric("No-Shrink Error", f"{final_noshrink_err:.1f}%", delta=f"{final_noshrink_err - final_shrunk_err:+.1f}%", delta_color="inverse")
        m4.metric("Deviance Ratio", f"{final_shrunk_dev:.2f} vs {final_noshrink_dev:.2f}", help="Shrunk Deviance vs No-Shrink Deviance")

        tab_deviance, tab_interaction, tab_boundary = st.tabs([
            "📈 Deviance & Overfitting (Recreating Slide 297)",
            "🌳 Tree Size J Comparison (Recreating Slide 295)",
            "🗺️ 2D Decision Manifold",
        ])

        m_axis = list(range(1, max_m + 1))

        with tab_deviance:
            st.markdown(f"##### 📉 Test Binomial Deviance: Shrinkage (ν = {shrinkage_nu}) vs No Shrinkage (ν = 1.0)")

            fig_dev = go.Figure()

            # No shrinkage curve (Red)
            fig_dev.add_trace(
                go.Scatter(
                    x=m_axis,
                    y=test_dev_noshrink,
                    mode="lines",
                    name="No Shrinkage (ν = 1.0)",
                    line=dict(color="#EF4444", width=2.8),
                    hovertemplate="Iter %{x}: Deviance = %{y:.3f}<extra></extra>",
                )
            )

            # With shrinkage curve (Blue/Emerald)
            fig_dev.add_trace(
                go.Scatter(
                    x=m_axis,
                    y=test_dev_shrunk,
                    mode="lines",
                    name=f"Shrinkage (ν = {shrinkage_nu})",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    hovertemplate="Iter %{x}: Deviance = %{y:.3f}<extra></extra>",
                )
            )

            fig_dev.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Boosting Iterations M", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Test Binomial Deviance", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_dev, width="stretch", key="si_fig_dev")

        with tab_interaction:
            st.markdown(f"##### 🌳 Test Error vs Terms M: Stumps (J=2, additive) vs J={tree_size_j} Trees")

            fig_tree_comp = go.Figure()

            # Stumps J=2 (Sky Blue)
            fig_tree_comp.add_trace(
                go.Scatter(
                    x=m_axis,
                    y=[e * 100.0 for e in test_err_stumps],
                    mode="lines",
                    name="Stumps (J = 2, Main Effects Only)",
                    line=dict(color="#38BDF8", width=2.5),
                    hovertemplate="Iter %{x}: Stump Err = %{y:.1f}%<extra></extra>",
                )
            )

            # Active J (Emerald)
            fig_tree_comp.add_trace(
                go.Scatter(
                    x=m_axis,
                    y=[e * 100.0 for e in test_err_shrunk],
                    mode="lines",
                    name=f"Selected Trees (J = {tree_size_j}, Order-{tree_size_j - 1} Interactions)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    hovertemplate="Iter %{x}: J=" + str(tree_size_j) + " Err = %{y:.1f}%<extra></extra>",
                )
            )

            fig_tree_comp.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Number of Boosting Terms M", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Test Error Rate (%)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_tree_comp, width="stretch", key="si_fig_tree_comp")

        with tab_boundary:
            st.markdown(f"##### 🎯 Final MART Decision Manifold (J = {tree_size_j}, ν = {shrinkage_nu}, M = {max_m})")

            # Dense grid for contour
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            gx = np.linspace(x_min, x_max, 120)
            gy = np.linspace(y_min, y_max, 120)
            gxx, gyy = np.meshgrid(gx, gy)
            grid_pts = np.c_[gxx.ravel(), gyy.ravel()]

            prob_grid = gb_shrunk.predict_proba(grid_pts)[:, 1].reshape(gxx.shape)

            fig_bound = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=420)

            # Probability shading
            fig_bound.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=prob_grid,
                    colorscale="Tealgrn",
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="P(Y=1)", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    contours=dict(start=0.1, end=0.9, size=0.1),
                    hoverinfo="skip",
                )
            )

            # Decision Boundary (P = 0.5)
            fig_bound.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=prob_grid,
                    contours_coloring="none",
                    contours=dict(start=0.5, end=0.5, size=0.1, coloring="none"),
                    line=dict(color="#FFFFFF", width=2.5),
                    name="Boundary (P = 0.5)",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # Scatter points
            fig_bound = add_scatter_points(fig_bound, X, y_svm, marker_size=7)

            st.plotly_chart(fig_bound, width="stretch", key="si_fig_bound")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Interaction Level & Regularization by Shrinkage", expanded=False):
        st.markdown(
            r"""
            **Interaction Level of MART Trees (Slide 294)**:
            - A decision tree of size $J$ (having $J$ terminal leaves) can model interaction effects among at most $J - 1$ variables.
            - $J = 2$ (Stumps): Purely additive model $f(x) = \sum_{j=1}^p f_j(x_j)$. Cannot capture interaction terms like $x_1 x_2$.
            - $J = 3$: Produces models with main effects and 2-factor interactions.
            - $4 \le J \le 8$: Dominant interactions in real-world data are typically low-order. Setting $J \approx 6$ is the recommended **rule of thumb**.

            **The Role of Shrinkage $\nu$ (Slides 296–297)**:
            - When each constituent tree is added, its step is scaled by learning rate $\nu \in (0, 1]$:
              $$f_m(x) = f_{m-1}(x) + \nu \sum_{j=1}^J \gamma_{jm} I(x \in R_{jm})$$
            - **Without shrinkage ($\nu = 1.0$)**: The model aggressively fits training residuals. As $M$ grows large, test deviance explodes upwards due to over-confidence on noisy instances.
            - **With shrinkage ($\nu \le 0.1$)**: The model takes conservative incremental steps. Test deviance remains stable and monotonically levels off, allowing $M > 500$ without fatal overfitting.
            """
        )
