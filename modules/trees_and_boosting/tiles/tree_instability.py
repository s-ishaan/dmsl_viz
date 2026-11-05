"""
Tile 4: Decision Tree Instability & High Variance Propagation
Slide: 280
Focus: Demonstrating how the greedy hierarchical nature of CART induces high variance:
splits conditioned on ancestor splits cause errors and small sample fluctuations
at the top to cascade downward, resulting in wildly different decision boundaries
and tree structures from tiny perturbations of the training data.
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
    COLOR_MARGIN,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Instability Controls")

            perturb_pct = st.slider(
                "Data Perturbation / Drop Rate",
                min_value=0.05,
                max_value=0.30,
                value=0.10,
                step=0.05,
                key="ti_perturb",
                help="Fraction of data randomly perturbed/dropped to simulate sampling variation (Slide 280).",
            )

            resample_seed = st.slider(
                "Perturbation Random Seed",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key="ti_seed",
                help="Change seed to simulate alternative data draws.",
            )

            max_depth = st.slider(
                "Tree Depth",
                min_value=2,
                max_value=6,
                value=4,
                step=1,
                key="ti_depth",
                help="Deeper trees suffer from exponentially greater variance propagation.",
            )

            dataset_type = st.selectbox(
                "Dataset Topology",
                ["XOR / Checkerboard", "Moons", "Concentric Circles", "Overlapping Blobs"],
                index=0,
                key="ti_dataset",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Points N", 50, 160, 90, 10, key="ti_samples")
                noise = st.slider("Noise Level", 0.05, 0.35, 0.15, 0.05, key="ti_noise")

    # Generate base dataset
    X_base, y_base = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)

    # 1. Fit Base Tree (Full dataset)
    tree_base = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree_base.fit(X_base, y_base)

    # 2. Perturbed dataset: drop perturb_pct of instances randomly
    rng_p = np.random.RandomState(resample_seed)
    n_keep = int(len(X_base) * (1.0 - perturb_pct))
    keep_indices = rng_p.choice(len(X_base), size=n_keep, replace=False)
    X_pert = X_base[keep_indices]
    y_pert = y_base[keep_indices]

    tree_pert = DecisionTreeClassifier(max_depth=max_depth, random_state=resample_seed)
    tree_pert.fit(X_pert, y_pert)

    # 3. Grid evaluation for disagreement
    x_min, x_max = float(np.min(X_base[:, 0]) - 0.5), float(np.max(X_base[:, 0]) + 0.5)
    y_min, y_max = float(np.min(X_base[:, 1]) - 0.5), float(np.max(X_base[:, 1]) + 0.5)
    gx = np.linspace(x_min, x_max, 130)
    gy = np.linspace(y_min, y_max, 130)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]

    pred_base_grid = tree_base.predict(grid_pts).reshape(XX.shape)
    pred_pert_grid = tree_pert.predict(grid_pts).reshape(XX.shape)
    disagreement_grid = (pred_base_grid != pred_pert_grid).astype(float)
    disagreement_pct = np.mean(disagreement_grid) * 100.0

    # 4. Bootstrap experiment: root split feature across 10 bootstrap draws
    root_features = []
    root_thresholds = []
    for b_seed in range(1, 13):
        boot_idx = np.random.RandomState(b_seed).choice(len(X_base), size=len(X_base), replace=True)
        t_boot = DecisionTreeClassifier(max_depth=max_depth, random_state=b_seed)
        t_boot.fit(X_base[boot_idx], y_base[boot_idx])
        root_features.append("Feature x₁" if t_boot.tree_.feature[0] == 0 else "Feature x₂")
        root_thresholds.append(t_boot.tree_.threshold[0])

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Spatial Disagreement", f"{disagreement_pct:.1f}%", help="Percentage of 2D space where the two trees make opposite predictions.")
            m2.metric("Dropped Points", f"{len(X_base) - n_keep} / {len(X_base)} ({perturb_pct*100:.0f}%)")
            m3.metric("Base Root Split", f"x_{tree_base.tree_.feature[0]+1} ≤ {tree_base.tree_.threshold[0]:.2f}")
            m4.metric("Perturbed Root Split", f"x_{tree_pert.tree_.feature[0]+1} ≤ {tree_pert.tree_.threshold[0]:.2f}")

        tab_overlay, tab_diff, tab_bootstrap = st.tabs([
            "🔀 Decision Boundary Instability Overlay",
            "⚡ Spatial Prediction Disagreement Map",
            "🎲 Bootstrap Root-Split Fluctuation (12 Draws)",
        ])

        with tab_overlay:
            st.markdown("##### 🔀 Boundary Comparison: Base Tree (Solid) vs Perturbed Tree (Dashed)")

            fig_ov = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Highlight dropped points with red 'X' markers
            dropped_mask = np.ones(len(X_base), dtype=bool)
            dropped_mask[keep_indices] = False
            if np.any(dropped_mask):
                fig_ov.add_trace(
                    go.Scatter(
                        x=X_base[dropped_mask, 0],
                        y=X_base[dropped_mask, 1],
                        mode="markers",
                        name="Removed Points",
                        marker=dict(size=11, color="#F43F5E", symbol="x", line=dict(width=2)),
                        hovertemplate="Removed: (%{x:.2f}, %{y:.2f})<extra></extra>",
                    )
                )

            # Scatter kept points
            fig_ov = add_scatter_points(fig_ov, X_pert, y_pert, marker_size=9)

            # Base Tree boundary (Solid Emerald)
            fig_ov.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=pred_base_grid,
                    contours_coloring="none",
                    contours=dict(start=0, end=0, size=1, coloring="none"),
                    line=dict(color=COLOR_HYPERPLANE, width=3.0),
                    name="Base Tree Boundary (100% Data)",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # Perturbed Tree boundary (Dashed Amber)
            fig_ov.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=pred_pert_grid,
                    contours_coloring="none",
                    contours=dict(start=0, end=0, size=1, coloring="none"),
                    line=dict(color=COLOR_MARGIN, width=3.0, dash="dash"),
                    name="Perturbed Tree Boundary",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            st.plotly_chart(fig_ov, width="stretch", key="ti_fig_ov")

        with tab_diff:
            st.markdown(f"##### ⚡ Disagreement Region (Red Highlight = {disagreement_pct:.1f}% Discordance)")

            fig_diff = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            fig_diff.add_trace(
                go.Heatmap(
                    x=gx,
                    y=gy,
                    z=disagreement_grid,
                    colorscale=[[0, "rgba(15, 23, 42, 0.2)"], [1, "rgba(244, 63, 94, 0.75)"]],
                    showscale=False,
                    hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>Conflict: %{z}<extra></extra>",
                )
            )

            # Scatter points
            fig_diff = add_scatter_points(fig_diff, X_base, y_base, marker_size=8)

            st.plotly_chart(fig_diff, width="stretch", key="ti_fig_diff")

        with tab_bootstrap:
            st.markdown("##### 🎲 Root Split Variable across 12 Independent Bootstrap Resamples")

            fig_boot = go.Figure()
            boot_labels = [f"Resample #{i+1}" for i in range(len(root_features))]
            colors = [COLOR_HYPERPLANE if feat == "Feature x₁" else "#38BDF8" for feat in root_features]

            fig_boot.add_trace(
                go.Bar(
                    x=boot_labels,
                    y=root_thresholds,
                    marker=dict(color=colors),
                    customdata=root_features,
                    hovertemplate="%{x}<br>Split Feature: %{customdata}<br>Split Threshold: %{y:.2f}<extra></extra>",
                )
            )

            fig_boot.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Bootstrap Data Draw", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Root Split Threshold s", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                margin=dict(l=45, r=25, t=15, b=60),
            )

            st.plotly_chart(fig_boot, width="stretch", key="ti_fig_boot")
            st.caption("Green bars = split on Feature x₁; Blue bars = split on Feature x₂. Notice how root split toggles completely across resamples.")
