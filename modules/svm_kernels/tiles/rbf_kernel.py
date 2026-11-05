"""
Tile 2: Radial Basis Function (RBF) Kernel & Localization Dynamics
Slides: 238, 240, 259
Focus: Exploring the Gaussian / RBF kernel k(x, x') = exp(-γ||x - x'||²), the role of γ = 1/(2σ²),
and the dramatic transition from global smooth boundaries to localized overfitting islands.
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
    BG_PAPER,
    BG_PLOT,
    COLOR_HYPERPLANE,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ RBF Kernel Knobs")

            dataset_type = st.selectbox(
                "Dataset Topology",
                ["Moons", "Concentric Circles", "XOR / Checkerboard", "Overlapping Blobs"],
                index=0,
                key="rbf_dataset",
                help="Geometric distribution of data points.",
            )

            gamma_exp = st.slider(
                "Inverse Bandwidth log₁₀(γ)",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.25,
                key="rbf_gamma_exp",
                help="γ in exp(-γ||x - x'||²). High γ = localized tight islands (Slide 240, 259).",
            )
            gamma_val = 10.0 ** gamma_exp
            sigma_val = np.sqrt(1.0 / (2.0 * gamma_val)) if gamma_val > 0 else np.nan

            c_exp = st.slider(
                "Regularization log₁₀(C)",
                min_value=-2.0,
                max_value=3.0,
                value=1.0,
                step=0.5,
                key="rbf_c_exp",
                help="Trade-off penalty C. Large C penalizes margin violations heavily.",
            )
            C_val = 10.0 ** c_exp

            with st.expander("⚙️ Advanced Settings", expanded=False):
                n_samples = st.slider("Sample Count", 30, 160, 80, 10, key="rbf_samples")
                noise = st.slider("Noise Level", 0.02, 0.40, 0.15, 0.02, key="rbf_noise")
                show_surface = st.checkbox("Show Decision Heatmap", value=True, key="rbf_surface")

    # Generate dataset
    X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)

    # Fit RBF SVM
    clf = SVC(kernel="rbf", gamma=gamma_val, C=C_val)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    acc = np.mean(y_pred == y) * 100.0
    n_sv = len(clf.support_)
    sv_ratio = (n_sv / len(y)) * 100.0

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Kernel Parameter γ", f"{gamma_val:.3f}")
            m2.metric("Effective Width σ", f"{sigma_val:.3f}")
            m3.metric("Support Vectors", f"{n_sv} / {len(y)} ({sv_ratio:.0f}%)")
            m4.metric("Train Accuracy", f"{acc:.1f}%")

        # Tabs to give each graph full 1-block width
        tab_boundary, tab_bell = st.tabs([
            "🌐 2D RBF Decision Boundary & Islands",
            "🔔 Radial Influence Decay Curve k(r)",
        ])

        x_min, x_max = float(np.min(X[:, 0]) - 1.2), float(np.max(X[:, 0]) + 1.2)
        y_min, y_max = float(np.min(X[:, 1]) - 1.2), float(np.max(X[:, 1]) + 1.2)

        with tab_boundary:
            boundary_label = "RBF Boundary: Smooth Global" if gamma_val < 0.5 else ("RBF Boundary: Localized Islands" if gamma_val > 5.0 else "RBF Boundary: Balanced")
            st.markdown(f"##### 🎯 {boundary_label} (γ = {gamma_val:.3f}, C = {C_val:.2f})")

            fig_2d = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Scatter points
            fig_2d = add_scatter_points(fig_2d, X, y, marker_size=10)

            # Decision boundary and margins
            fig_2d = add_contour_boundary_and_margins(
                fig_2d,
                clf.decision_function,
                (x_min, x_max),
                (y_min, y_max),
                resolution=130,
                show_margins=True,
                show_surface=show_surface,
                boundary_name="RBF Boundary f(x)=0",
                line_width=3,
            )

            # Support vectors
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

            st.plotly_chart(fig_2d, width="stretch", key="rbf_fig_2d")

        with tab_bell:
            st.markdown(f"##### 📉 Support Vector Weighting Window: k(r) = exp(-γ · r²) vs Euclidean Distance r")
            r_vals = np.linspace(0.0, 4.0, 300)
            k_vals = np.exp(-gamma_val * (r_vals ** 2))

            fig_bell = go.Figure()
            fig_bell.add_trace(
                go.Scatter(
                    x=r_vals,
                    y=k_vals,
                    mode="lines",
                    name=f"k(r) with γ={gamma_val:.3f}",
                    line=dict(color=COLOR_HYPERPLANE, width=3),
                    fill="tozeroy",
                    fillcolor="rgba(16, 185, 129, 0.15)",
                    hovertemplate="Distance r: %{x:.2f}<br>Similarity k(r): %{y:.3f}<extra></extra>",
                )
            )

            # Half-height marker (FWHM)
            r_half = np.sqrt(np.log(2) / gamma_val) if gamma_val > 0 else 0
            if r_half <= 4.0:
                fig_bell.add_trace(
                    go.Scatter(
                        x=[r_half, r_half],
                        y=[0, 0.5],
                        mode="lines+markers",
                        name="Half-Maximum Point",
                        line=dict(color="#F59E0B", width=2, dash="dot"),
                        hovertemplate=f"Half-max distance: r={r_half:.2f}<extra></extra>",
                    )
                )

            fig_bell.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Euclidean Distance r = ||x - xᵢ||", font=dict(color="#CBD5E1", size=11)),
                    range=[0, 4.0],
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Kernel Similarity k(r)", font=dict(color="#CBD5E1", size=11)),
                    range=[-0.05, 1.05],
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                margin=dict(l=45, r=25, t=15, b=75),
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

            st.plotly_chart(fig_bell, width="stretch", key="rbf_fig_bell")
