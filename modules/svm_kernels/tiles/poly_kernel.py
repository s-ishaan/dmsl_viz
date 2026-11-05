"""
Tile 1: Polynomial Kernel Dynamics & Explicit Monomial Feature Mapping
Slides: 227-231, 240
Focus: Exploring the polynomial kernel k(x, x') = (c₀ + γ⟨x, x'⟩)ᵈ, degree d, intercept c₀,
and the explicit 6-dimensional monomial mapping Φ(x) for d=2.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    add_contour_boundary_and_margins,
    highlight_support_vectors,
    COLOR_CLASS_NEG,
    COLOR_CLASS_POS,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Polynomial Kernel Controls")

            dataset_type = st.selectbox(
                "Dataset Topology",
                ["Concentric Circles", "Moons", "XOR / Checkerboard", "Overlapping Blobs"],
                index=0,
                key="pk_dataset",
                help="Geometric configuration of training instances.",
            )

            degree = st.slider(
                "Polynomial Degree d",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                key="pk_degree",
                help="Power d in k(x, x') = (c₀ + ⟨x, x'⟩)ᵈ (Slide 240).",
            )

            coef0 = st.slider(
                "Independent Intercept c₀",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.25,
                key="pk_coef0",
                help="Constant term c₀ inside the polynomial kernel.",
            )

            c_exp = st.slider(
                "Penalty log₁₀(C)",
                min_value=-2.0,
                max_value=3.0,
                value=0.0,
                step=0.5,
                key="pk_c_exp",
                help="Soft margin trade-off penalty parameter C.",
            )
            C_val = 10.0 ** c_exp

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Count", 30, 150, 70, 10, key="pk_samples")
                noise = st.slider("Noise Level", 0.02, 0.35, 0.10, 0.02, key="pk_noise")
                show_surface = st.checkbox("Show Decision Heatmap", value=True, key="pk_surface")

    # Generate dataset
    X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)

    # Fit Polynomial Kernel SVM
    clf = SVC(kernel="poly", degree=degree, coef0=coef0, gamma="scale", C=C_val)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    acc = np.mean(y_pred == y) * 100.0
    n_sv = len(clf.support_)
    sv_ratio = (n_sv / len(y)) * 100.0

    with col_plot:
        # High-level metrics header
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Kernel Formulation", f"(c₀ + ⟨x,x'⟩)^{degree}")
            m2.metric("Penalty C", f"{C_val:.2f}")
            m3.metric("Support Vectors", f"{n_sv} / {len(y)} ({sv_ratio:.0f}%)")
            m4.metric("Train Accuracy", f"{acc:.1f}%")

        # Tabs for 1-block views
        tab_2d, tab_3d = st.tabs([
            "🗺️ 2D Input Space Decision Boundary",
            "🚀 Explicit Monomial Feature Space Φ(x) (d=2)",
        ])

        x_min, x_max = float(np.min(X[:, 0]) - 1.2), float(np.max(X[:, 0]) + 1.2)
        y_min, y_max = float(np.min(X[:, 1]) - 1.2), float(np.max(X[:, 1]) + 1.2)

        with tab_2d:
            st.markdown(f"##### 📐 Polynomial SVM (Degree d = {degree}, c₀ = {coef0:.2f})")
            fig_2d = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Scatter points
            fig_2d = add_scatter_points(fig_2d, X, y, marker_size=10)

            # Non-linear boundary and margins via decision function
            fig_2d = add_contour_boundary_and_margins(
                fig_2d,
                clf.decision_function,
                (x_min, x_max),
                (y_min, y_max),
                resolution=120,
                show_margins=True,
                show_surface=show_surface,
                boundary_name=f"Poly Boundary d={degree}",
                line_width=3,
            )

            # Support vector rings
            X_sv = clf.support_vectors_
            y_sv = y[clf.support_]
            dual_coefs = np.abs(clf.dual_coef_[0])
            labels = [f"Class {y_sv[i]} | Dual αᵢ={dual_coefs[i]:.3f}" for i in range(len(X_sv))]
            fig_2d = highlight_support_vectors(
                fig_2d,
                X_sv,
                y_sv=y_sv,
                labels=labels,
                color="#EAB308",
                name="Support Vectors (αᵢ > 0)",
            )

            st.plotly_chart(fig_2d, width="stretch", key="pk_fig_2d")

        with tab_3d:
            st.markdown("##### 🌌 3D Projection of 6D Monomial Space: [x₁², x₂², √2·x₁x₂]ᵀ")
            if degree != 2:
                st.info("ℹ️ The explicit 6D monomial basis expansion $\\Phi(x) = [1, \\sqrt{2}x_1, \\sqrt{2}x_2, x_1^2, x_2^2, \\sqrt{2}x_1x_2]^T$ (Slide 240) applies directly when degree $d=2$. The plot below projects the data onto the 3 leading quadratic terms.")

            # Compute quadratic features: z1 = x1^2, z2 = x2^2, z3 = sqrt(2)*x1*x2
            z1 = X[:, 0] ** 2
            z2 = X[:, 1] ** 2
            z3 = np.sqrt(2.0) * X[:, 0] * X[:, 1]

            mask_neg = (y == -1)
            mask_pos = (y == 1)

            fig_3d = go.Figure()

            # Class -1
            fig_3d.add_trace(
                go.Scatter3d(
                    x=z1[mask_neg],
                    y=z2[mask_neg],
                    z=z3[mask_neg],
                    mode="markers",
                    name="Class -1",
                    marker=dict(size=5, color=COLOR_CLASS_NEG, symbol="circle", opacity=0.9),
                    hovertemplate="x₁²: %{x:.2f}<br>x₂²: %{y:.2f}<br>√2 x₁x₂: %{z:.2f}<extra>Class -1</extra>",
                )
            )

            # Class +1
            fig_3d.add_trace(
                go.Scatter3d(
                    x=z1[mask_pos],
                    y=z2[mask_pos],
                    z=z3[mask_pos],
                    mode="markers",
                    name="Class +1",
                    marker=dict(size=5, color=COLOR_CLASS_POS, symbol="circle", opacity=0.9),
                    hovertemplate="x₁²: %{x:.2f}<br>x₂²: %{y:.2f}<br>√2 x₁x₂: %{z:.2f}<extra>Class +1</extra>",
                )
            )

            fig_3d.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                scene=dict(
                    bgcolor=BG_PLOT,
                    xaxis=dict(title="z₁ = x₁²", backgroundcolor=BG_PLOT, gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                    yaxis=dict(title="z₂ = x₂²", backgroundcolor=BG_PLOT, gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                    zaxis=dict(title="z₃ = √2 x₁x₂", backgroundcolor=BG_PLOT, gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
                ),
                margin=dict(l=20, r=20, t=10, b=60),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.12,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1"),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_3d, width="stretch", key="pk_fig_3d")
