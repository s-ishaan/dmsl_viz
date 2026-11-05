"""
Tile 1: Vector Algebra & Separating Hyperplane Geometry
Slides: 213-214
Focus: Interactive exploration of normal vector β*, signed distances, intercept β₀, and sign classification.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    add_normal_vector,
    add_linear_boundary_and_margins,
    COLOR_HYPERPLANE,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Hyperplane Knobs")
            
            angle_deg = st.slider(
                "Normal Angle θ (°)",
                min_value=0.0,
                max_value=360.0,
                value=45.0,
                step=5.0,
                key="ha_angle",
                help="Orientation of normal vector β = ||β|| [cos θ, sin θ]ᵀ",
            )
            
            norm_beta = st.slider(
                "Weight Norm ||β||",
                min_value=0.2,
                max_value=4.0,
                value=1.5,
                step=0.1,
                key="ha_norm",
                help="Scaling ||β|| changes functional distances without moving the line!",
            )

            beta_0 = st.slider(
                "Intercept β₀",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.25,
                key="ha_beta0",
                help="Translates hyperplane perpendicularly along its normal",
            )

            with st.expander("⚙️ Dataset & Advanced Overlays", expanded=False):
                n_pts = st.slider("Sample Count N", 20, 100, 40, step=10, key="ha_n_pts")
                show_normal = st.checkbox("Unit Normal Vector β*", value=True, key="ha_show_normal")
                show_projections = st.checkbox("Signed Distance Projections", value=True, key="ha_show_proj")
                color_by_pred = st.checkbox("Color by Predicted Class sign(f(x))", value=False, key="ha_color_pred")
                compare_osh = st.checkbox("Compare with Optimal OSH Line", value=False, key="ha_compare_osh")

    # Compute beta vector
    theta_rad = np.radians(angle_deg)
    beta_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    beta = norm_beta * beta_unit

    # Generate synthetic 2D data
    X, y_true = get_dataset("Linearly Separable Blobs", n_samples=n_pts, noise=0.15, separation=2.5, random_state=42)

    # Compute signed distances: d_i = (beta^T x_i + beta_0) / ||beta||
    f_vals = np.dot(X, beta) + beta_0
    signed_distances = f_vals / norm_beta
    y_pred = np.where(f_vals >= 0, 1, -1)

    with col_plot:
        # Metrics container
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Unit Normal β*", f"[{beta_unit[0]:.2f}, {beta_unit[1]:.2f}]")
            m2.metric("Weight Norm ||β||", f"{norm_beta:.2f}")
            m3.metric("Dist to Origin d(0, L)", f"{abs(beta_0)/norm_beta:.2f}")
            acc = np.mean(y_pred == y_true) * 100
            m4.metric("Separation Accuracy", f"{acc:.1f}%")

        # Chart Title & Plotly Figure
        st.markdown("##### 📐 Separating Hyperplane: f(x) = β₀ + βᵀx = 0")
        x_lim = (-4.5, 4.5)
        y_lim = (-4.5, 4.5)
        fig = get_base_2d_figure(x_lim, y_lim, title="", height=440)

        # Scatter points
        y_display = y_pred if color_by_pred else y_true
        label_prefix = "Pred: " if color_by_pred else "True: "
        fig = add_scatter_points(fig, X, y_display, marker_size=10, name_prefix=label_prefix)

        # Plot user-controlled hyperplane line
        xs = np.linspace(x_lim[0] - 2, x_lim[1] + 2, 200)
        if abs(beta[1]) > 1e-4:
            ys = -(beta[0] * xs + beta_0) / beta[1]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="Manual Line: f(x)=0",
                    line=dict(color=COLOR_HYPERPLANE, width=3),
                    hoverinfo="skip",
                )
            )
        else:
            x_val = -beta_0 / (beta[0] if abs(beta[0]) > 1e-6 else 1e-6)
            fig.add_trace(
                go.Scatter(
                    x=[x_val, x_val],
                    y=[y_lim[0] - 2, y_lim[1] + 2],
                    mode="lines",
                    name="Manual Line: f(x)=0",
                    line=dict(color=COLOR_HYPERPLANE, width=3),
                    hoverinfo="skip",
                )
            )

        # Compare with OSH if toggled
        if compare_osh:
            try:
                clf_osh = SVC(kernel="linear", C=100.0).fit(X, y_true)
                w_osh = clf_osh.coef_[0]
                b_osh = clf_osh.intercept_[0]
                fig = add_linear_boundary_and_margins(
                    fig,
                    w_osh,
                    b_osh,
                    x_lim,
                    y_lim,
                    show_margins=True,
                    show_margin_band=True,
                    boundary_color="#38BDF8",  # Sky Blue
                    boundary_name="Optimal OSH",
                    line_width=2.5,
                )
            except Exception:
                pass

        # Normal vector arrow
        if show_normal:
            fig = add_normal_vector(fig, beta, beta_0, scale=1.8)

        # Orthogonal projections
        if show_projections:
            proj_x = []
            proj_y = []
            for i in range(len(X)):
                xi = X[i]
                d = signed_distances[i]
                x_proj = xi - d * beta_unit
                proj_x.extend([xi[0], x_proj[0], None])
                proj_y.extend([xi[1], x_proj[1], None])

            fig.add_trace(
                go.Scatter(
                    x=proj_x,
                    y=proj_y,
                    mode="lines",
                    name="Signed Dist Projections",
                    line=dict(color="rgba(139, 92, 246, 0.4)", width=1.5, dash="dot"),
                    hoverinfo="skip",
                )
            )

        st.plotly_chart(fig, width="stretch")
