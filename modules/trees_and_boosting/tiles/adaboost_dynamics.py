"""
Tile 5: AdaBoost.M1 Sequential Reweighting Dynamics
Slides: 284-287
Focus: Step-by-step exploration of Algorithm 10.1 (AdaBoost.M1).
Scrub through iterations m = 1 ... M to observe:
1. Individual observation weights w_i swelling for previously misclassified points.
2. The current weak decision stump G_m(x).
3. The emergence of the cumulative non-linear committee G(x) = sign(∑ α_m G_m(x)).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def run_adaboost_manual(X, y, M, max_depth=1):
    """
    Exact implementation of Algorithm 10.1 (Slide 285).
    Returns history of weights, weak classifiers, errors, and alphas.
    """
    N = len(y)
    w = np.full(N, 1.0 / N)

    classifiers = []
    alphas = []
    errors = []
    weights_history = []

    for m in range(M):
        weights_history.append(w.copy())

        # Fit weak learner on weighted samples
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=m)
        clf.fit(X, y, sample_weight=w)
        pred = clf.predict(X)

        # Weighted error rate
        misclassified = (pred != y)
        err_m = np.sum(w * misclassified) / np.sum(w)
        err_m = np.clip(err_m, 1e-10, 0.4999)  # Avoid div by 0 and negative weights

        # Classifier importance weight alpha_m = log((1 - err_m) / err_m)
        alpha_m = np.log((1.0 - err_m) / err_m)

        # Update weights: w_i <- w_i * exp(alpha_m * I(y_i != G_m(x_i)))
        w = w * np.exp(alpha_m * misclassified.astype(float))
        w = w / np.sum(w)  # Normalize

        classifiers.append(clf)
        alphas.append(alpha_m)
        errors.append(err_m)

    return classifiers, alphas, errors, weights_history


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ AdaBoost Knobs")

            total_m = st.slider(
                "Total Boosting Iterations M",
                min_value=5,
                max_value=35,
                value=15,
                step=5,
                key="ab_total_m",
                help="Total ensemble size M (Slide 285).",
            )

            current_step = st.slider(
                "Inspect Step m",
                min_value=1,
                max_value=total_m,
                value=min(3, total_m),
                step=1,
                key="ab_step",
                help="Scrub through boosting iterations step-by-step.",
            )

            base_depth = st.radio(
                "Weak Learner Complexity",
                ["Decision Stump (Depth 1 - Slide 289)", "Shallow Tree (Depth 2)"],
                index=0,
                key="ab_depth",
                help="Stumps are single-split orthogonal axis cuts.",
            )
            depth_val = 1 if "Stump" in base_depth else 2

            dataset_type = st.selectbox(
                "Dataset Topology",
                ["XOR / Checkerboard", "Moons", "Concentric Circles", "Overlapping Blobs"],
                index=0,
                key="ab_dataset",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Points N", 40, 160, 80, 10, key="ab_samples")
                noise = st.slider("Noise Level", 0.05, 0.35, 0.12, 0.02, key="ab_noise")

    # Generate data and run AdaBoost
    X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)
    clfs, alphas, errors, weights_hist = run_adaboost_manual(X, y, total_m, max_depth=depth_val)

    step_idx = current_step - 1
    active_clf = clfs[step_idx]
    active_alpha = alphas[step_idx]
    active_err = errors[step_idx]
    active_weights = weights_hist[step_idx]

    # Precompute 2D grid for decision boundaries
    x_min, x_max = float(np.min(X[:, 0]) - 0.5), float(np.max(X[:, 0]) + 0.5)
    y_min, y_max = float(np.min(X[:, 1]) - 0.5), float(np.max(X[:, 1]) + 0.5)
    gx = np.linspace(x_min, x_max, 130)
    gy = np.linspace(y_min, y_max, 130)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]

    # 1. Single weak learner prediction at step m
    pred_weak_grid = active_clf.predict(grid_pts).reshape(XX.shape)

    # 2. Cumulative ensemble prediction up to step m: G(x) = sign(∑ α_k G_k(x))
    ensemble_score_grid = np.zeros(len(grid_pts))
    for k in range(current_step):
        ensemble_score_grid += alphas[k] * clfs[k].predict(grid_pts)
    pred_ensemble_grid = np.sign(ensemble_score_grid).reshape(XX.shape)

    # Training accuracy of cumulative ensemble at step m
    train_scores = np.zeros(len(X))
    for k in range(current_step):
        train_scores += alphas[k] * clfs[k].predict(X)
    ensemble_acc = np.mean(np.sign(train_scores) == y) * 100.0

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Iteration", f"m = {current_step} / {total_m}")
            m2.metric("Stump Error err_m", f"{active_err * 100.0:.1f}%")
            m3.metric("Stump Weight α_m", f"{active_alpha:.3f}", help="α_m = log((1 - err_m) / err_m). Higher for lower error.")
            m4.metric("Committee Acc", f"{ensemble_acc:.1f}%")

        tab_weighted, tab_ensemble, tab_alphas = st.tabs([
            f"🎯 Step m={current_step}: Observation Weights wᵢ & Weak Split G_m",
            f"🛡️ Cumulative Committee Boundary G(x) (1 ... {current_step})",
            "📊 Classifier Confidence α_k & Error err_k Profile",
        ])

        with tab_weighted:
            st.markdown(f"##### 🎯 Step m = {current_step}: Observation Weights wᵢ (Bubble Size) & Weak Split G_m(x)")

            fig_w = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Draw weak learner decision boundary (stump cut line)
            fig_w.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=pred_weak_grid,
                    contours_coloring="none",
                    contours=dict(start=0, end=0, size=1, coloring="none"),
                    line=dict(color="#F59E0B", width=3.0, dash="dash"),
                    name=f"Stump G_{current_step}(x) Split",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # Scatter points with bubble size proportional to active_weights
            norm_w = active_weights / np.mean(active_weights)
            marker_sizes = np.clip(6.0 + norm_w * 9.0, 5.0, 30.0)

            mask_neg = (y == -1)
            mask_pos = (y == 1)

            # Class -1
            fig_w.add_trace(
                go.Scatter(
                    x=X[mask_neg, 0],
                    y=X[mask_neg, 1],
                    mode="markers",
                    name="Class -1 (Sized by wᵢ)",
                    marker=dict(
                        color="#F43F5E",
                        size=marker_sizes[mask_neg],
                        line=dict(color="#FFFFFF", width=1.2),
                    ),
                    customdata=active_weights[mask_neg],
                    hovertemplate="Class -1<br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>Weight wᵢ: %{customdata:.4f}<extra></extra>",
                )
            )

            # Class +1
            fig_w.add_trace(
                go.Scatter(
                    x=X[mask_pos, 0],
                    y=X[mask_pos, 1],
                    mode="markers",
                    name="Class +1 (Sized by wᵢ)",
                    marker=dict(
                        color="#38BDF8",
                        size=marker_sizes[mask_pos],
                        line=dict(color="#FFFFFF", width=1.2),
                    ),
                    customdata=active_weights[mask_pos],
                    hovertemplate="Class +1<br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>Weight wᵢ: %{customdata:.4f}<extra></extra>",
                )
            )

            st.plotly_chart(fig_w, width="stretch", key="ab_fig_w")
            st.caption("Larger circles highlight instances previously misclassified, forcing stump G_m to prioritize them (Slide 286).")

        with tab_ensemble:
            st.markdown(f"##### 🛡️ Cumulative Ensemble Boundary: G(x) = sign(∑ₖ₌₁ᵐ αₖ Gₖ(x)) (m = {current_step})")

            fig_ens = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Background committee confidence heatmap
            fig_ens.add_trace(
                go.Heatmap(
                    x=gx,
                    y=gy,
                    z=ensemble_score_grid.reshape(XX.shape),
                    colorscale="RdBu",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Committee Score", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    opacity=0.6,
                    hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>Vote Score: %{z:.2f}<extra></extra>",
                )
            )

            # Ensemble 0-level decision boundary contour
            fig_ens.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=pred_ensemble_grid,
                    contours_coloring="none",
                    contours=dict(start=0, end=0, size=1, coloring="none"),
                    line=dict(color="#FFFFFF", width=3.0),
                    name=f"Ensemble Boundary (m={current_step})",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # Scatter points
            fig_ens = add_scatter_points(fig_ens, X, y, marker_size=9)

            st.plotly_chart(fig_ens, width="stretch", key="ab_fig_ens")

        with tab_alphas:
            st.markdown("##### 📶 Weak Classifier Weights αₖ and Error Rates errₖ Across Iterations")

            k_axis = np.arange(1, total_m + 1)
            bar_colors = [COLOR_HYPERPLANE if k <= current_step else "#334155" for k in k_axis]

            fig_bar = go.Figure()

            # Classifier Alpha weights
            fig_bar.add_trace(
                go.Bar(
                    x=k_axis,
                    y=alphas,
                    name="Classifier Weight αₖ",
                    marker=dict(color=bar_colors),
                    hovertemplate="Iteration #%{x}<br>Weight α: %{y:.3f}<extra></extra>",
                )
            )

            # Weighted error line on secondary axis
            fig_bar.add_trace(
                go.Scatter(
                    x=k_axis,
                    y=errors,
                    mode="lines+markers",
                    name="Weighted Error errₖ",
                    yaxis="y2",
                    line=dict(color="#F43F5E", width=2.5),
                    marker=dict(size=6, color="#F43F5E"),
                    hovertemplate="Iteration #%{x}<br>Weighted Error: %{y:.3f}<extra></extra>",
                )
            )

            # Highlight current step
            fig_bar.add_vline(x=current_step, line_width=2, line_dash="dash", line_color="#38BDF8")

            fig_bar.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Boosting Iteration k", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Classifier Weight αₖ", font=dict(color=COLOR_HYPERPLANE, size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis2=dict(
                    title=dict(text="Weighted Error errₖ", font=dict(color="#F43F5E", size=11)),
                    overlaying="y",
                    side="right",
                    range=[0, 0.55],
                    tickfont=dict(color="#94A3B8"),
                    showgrid=False,
                ),
                margin=dict(l=45, r=45, t=15, b=75),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=10),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_bar, width="stretch", key="ab_fig_bar")
