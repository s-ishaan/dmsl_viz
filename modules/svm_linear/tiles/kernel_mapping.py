"""
Tile 6: High-Dimensional Feature Space Transformation (RKHS Foundation)
Slide: 212 (Basic Idea: Non-Linear Mapping to Higher Dimensions)
Focus: Dual-canvas exploration: Transforming non-separable 2D data into 3D space where a linear hyperplane
achieves clean separation, and observing the induced non-linear decision boundary in 2D.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.datasets import get_dataset
from utils.plotting import get_base_2d_figure, add_scatter_points


def render():
    st.markdown("#### 6. Feature Space Transformation (RKHS Foundation)")
    col_ctrl, col_main = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🚀 Mapping Function Φ(x)")
            mapping_type = st.selectbox(
                "Feature Mapping Φ(x₁, x₂)",
                [
                    "Paraboloid: [x₁, x₂, x₁² + x₂²]",
                    "Gaussian Radial: [x₁, x₂, exp(-γ(x₁² + x₂²))]",
                ],
                index=0,
                key="km_mapping",
                help="Lifts 2D points into 3D space where linear separation is achievable",
            )

            dataset_type = st.selectbox(
                "2D Dataset Pattern",
                ["Concentric Circles", "Moons"],
                index=0,
                key="km_dataset",
            )

            slice_height = st.slider(
                "3D Hyperplane Threshold (Z-Cut)",
                min_value=0.2,
                max_value=6.0,
                value=2.0,
                step=0.2,
                key="km_slice",
                help="Height of the planar decision surface slicing through 3D space",
            )

            with st.expander("⚙️ Dataset Parameters", expanded=False):
                n_pts = st.slider("Sample Count N", 30, 150, 70, step=10, key="km_n")
                noise = st.slider("Noise", 0.02, 0.25, 0.08, step=0.02, key="km_noise")

    # Generate 2D dataset
    X, y = get_dataset(dataset_type, n_samples=n_pts, noise=noise, random_state=42)

    # Compute 3D lifted features
    x1, x2 = X[:, 0], X[:, 1]
    if "Paraboloid" in mapping_type:
        z = x1 ** 2 + x2 ** 2
        z_label = "x₁² + x₂²"
        phi_X = np.column_stack([x1, x2, z])
        # Separating plane: z = slice_height
        y_pred_3d = np.where(phi_X[:, 2] <= slice_height, -1, 1)
        # 2D induced boundary: x1^2 + x2^2 = slice_height => circle of radius sqrt(slice_height)
        contour_radius = np.sqrt(max(0.01, slice_height))
    else:  # Gaussian Radial
        gamma = 0.5
        z = np.exp(-gamma * (x1 ** 2 + x2 ** 2))
        z_label = f"exp(-{gamma}(x₁² + x₂²))"
        phi_X = np.column_stack([x1, x2, z])
        y_pred_3d = np.where(phi_X[:, 2] >= slice_height, -1, 1)
        # exp(-gamma * r^2) = slice_height => r^2 = -ln(slice_height)/gamma
        val = -np.log(max(1e-4, min(0.999, slice_height))) / gamma if slice_height < 1.0 else 0.0
        contour_radius = np.sqrt(val) if val > 0 else 0.0

    acc_3d = np.mean(y_pred_3d == y) * 100

    with col_main:
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Original Space (2D)", "Non-Linearly Separable")
            m2.metric("Lifted Space (3D)", "Linearly Separable!")
            m3.metric("3D Planar Accuracy", f"{acc_3d:.1f}%")

        # Build 2D and 3D figures at full block dimensions
        x_lim = (-3.5, 3.5)
        y_lim = (-3.5, 3.5)
        fig_2d = get_base_2d_figure(x_lim, y_lim, title="", height=450)
        fig_2d = add_scatter_points(fig_2d, X, y, marker_size=9, show_legend=True)

        if contour_radius > 0:
            theta = np.linspace(0, 2 * np.pi, 200)
            cx = contour_radius * np.cos(theta)
            cy = contour_radius * np.sin(theta)
            fig_2d.add_trace(go.Scatter(
                x=cx, y=cy, mode="lines",
                line=dict(color="#10B981", width=3),
                name="Induced Boundary", hoverinfo="skip"
            ))

        fig_3d = go.Figure()
        mask_neg = (y == -1)
        mask_pos = (y == 1)

        fig_3d.add_trace(go.Scatter3d(
            x=phi_X[mask_neg, 0], y=phi_X[mask_neg, 1], z=phi_X[mask_neg, 2],
            mode="markers", name="Class -1",
            marker=dict(size=4.5, color="#F43F5E", opacity=0.9),
            hovertemplate="<b>Class -1</b><br>Z: %{z:.2f}<extra></extra>",
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=phi_X[mask_pos, 0], y=phi_X[mask_pos, 1], z=phi_X[mask_pos, 2],
            mode="markers", name="Class +1",
            marker=dict(size=4.5, color="#38BDF8", opacity=0.9),
            hovertemplate="<b>Class +1</b><br>Z: %{z:.2f}<extra></extra>",
        ))

        grid_x = np.linspace(-3.0, 3.0, 15)
        grid_y = np.linspace(-3.0, 3.0, 15)
        GX, GY = np.meshgrid(grid_x, grid_y)
        GZ = np.full_like(GX, slice_height)

        fig_3d.add_trace(go.Surface(
            x=GX, y=GY, z=GZ,
            colorscale=[[0, "rgba(16, 185, 129, 0.45)"], [1, "rgba(16, 185, 129, 0.45)"]],
            showscale=False, name="3D Separating Plane", hoverinfo="skip",
        ))

        fig_3d.update_layout(
            height=480,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#1E293B",
            scene=dict(
                xaxis=dict(title="x₁", backgroundcolor="#0F172A", gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                yaxis=dict(title="x₂", backgroundcolor="#0F172A", gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                zaxis=dict(title=z_label, backgroundcolor="#0F172A", gridcolor="#334155", tickfont=dict(color="#94A3B8")),
            ),
            showlegend=False,
        )

        # Full-Block Views
        tab_3d, tab_2d, tab_both = st.tabs([
            "🔮 3D Lifted Space (Planar Cut)",
            "🎯 2D Input Space (Induced Boundary)",
            "📑 Both Views Stacked (Full Width)",
        ])

        with tab_3d:
            st.markdown("##### 🔮 Planar Hyperplane Slicing in 3D Feature Space")
            st.plotly_chart(fig_3d, width="stretch", key="km_fig_3d")

        with tab_2d:
            st.markdown("##### 🎯 Induced Non-Linear Boundary in 2D Input Space")
            st.plotly_chart(fig_2d, width="stretch", key="km_fig_2d")

        with tab_both:
            st.markdown("##### 🔮 3D Lifted Space")
            st.plotly_chart(fig_3d, width="stretch", key="km_fig_3d_both")
            st.markdown("---")
            st.markdown("##### 🎯 2D Induced Boundary")
            st.plotly_chart(fig_2d, width="stretch", key="km_fig_2d_both")
