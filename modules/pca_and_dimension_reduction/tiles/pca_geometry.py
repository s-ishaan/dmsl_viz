"""
Tile 1: PCA Geometry & Spectral Decomposition
Focus: Slides 328-333 - Geometric and algebraic foundations of Principal Component Analysis.
Interactive rotating projection vector a(θ), covariance error ellipse, and eigenvalue decomposition
S_x = E Λ E^T proving that var(a^T x) is maximized along the first eigenvector e_1 with value λ_1.
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
            st.markdown("#### 🎛️ PCA Geometry Knobs")

            theta_deg = st.slider(
                "Projection Vector Angle θ (°)",
                min_value=0,
                max_value=180,
                value=35,
                step=1,
                key="pcag_theta",
                help="Steer unit vector a(θ) = [cos θ, sin θ]^T to probe the projected variance var(a^T x).",
            )

            rho_corr = st.slider(
                "Feature Correlation ρ(x₁, x₂)",
                min_value=-0.92,
                max_value=0.92,
                value=0.75,
                step=0.05,
                key="pcag_rho",
                help="Controls the tilt and eccentricity of the 2D data cloud.",
            )

            var_ratio = st.slider(
                "Variance Anisotropy (σ₁ / σ₂)",
                min_value=1.0,
                max_value=4.0,
                value=2.0,
                step=0.25,
                key="pcag_ratio",
                help="Ratio of standard deviations between coordinate axes.",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Count N", 60, 300, 150, 10, key="pcag_n")
                seed_val = st.number_input("Random Seed", value=42, step=1, key="pcag_seed")

    # Generate correlated 2D Gaussian data
    rng = np.random.RandomState(seed_val)
    s1 = var_ratio
    s2 = 1.0
    cov_12 = rho_corr * s1 * s2
    Sigma = np.array([[s1 ** 2, cov_12], [cov_12, s2 ** 2]])

    raw_data = rng.multivariate_normal(mean=[0.0, 0.0], cov=Sigma, size=n_samples)
    # Center data matrix X (Slide 333)
    X = raw_data - np.mean(raw_data, axis=0)

    # Sample covariance matrix S_x = (1/N) * X^T X (Slide 333)
    S_x = (X.T @ X) / n_samples
    eigenvals, eigenvecs = np.linalg.eigh(S_x)

    # Sort descending: lambda_1 >= lambda_2
    sort_idx = np.argsort(eigenvals)[::-1]
    lam1, lam2 = eigenvals[sort_idx[0]], eigenvals[sort_idx[1]]
    e1, e2 = eigenvecs[:, sort_idx[0]], eigenvecs[:, sort_idx[1]]

    # Ensure consistent eigenvector orientation for clean visuals
    if e1[0] < 0:
        e1 = -e1
    if e2[1] < 0:
        e2 = -e2

    tot_var = lam1 + lam2
    pct_pc1 = (lam1 / tot_var) * 100.0
    pct_pc2 = (lam2 / tot_var) * 100.0

    # Active unit projection vector a(θ)
    theta_rad = np.radians(theta_deg)
    a_vec = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    proj_z = X @ a_vec
    active_var = float(np.var(proj_z))

    # PC1 angle in degrees
    pc1_angle_deg = float(np.degrees(np.arctan2(e1[1], e1[0]))) % 180.0

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Var(aᵀx)", f"{active_var:.3f}")
        m2.metric("PC1 Max Var λ₁", f"{lam1:.3f}", f"{pct_pc1:.1f}% of total")
        m3.metric("PC2 Min Var λ₂", f"{lam2:.3f}", f"{pct_pc2:.1f}% of total")
        m4.metric("Total Var tr(S_x)", f"{tot_var:.3f}", help="tr(S_z) = tr(S_x) = λ₁ + λ₂")

        tab_2d, tab_curve, tab_proj = st.tabs([
            "🌐 2D Spectral Plane & Covariance Ellipse",
            "🔄 Projected Variance vs Angle θ",
            "📉 Projected 1D Data Cloud",
        ])

        with tab_2d:
            st.markdown(f"##### 🌐 Data Cloud, Principal Axes (λ₁={lam1:.2f}, λ₂={lam2:.2f}) & Rotating Probe a(θ={theta_deg}°)")

            fig_2d = go.Figure()

            # Data points
            fig_2d.add_trace(
                go.Scatter(
                    x=X[:, 0],
                    y=X[:, 1],
                    mode="markers",
                    marker=dict(color="#38BDF8", size=6, opacity=0.75, line=dict(color="#0F172A", width=0.8)),
                    name="Observations (Centered)",
                    hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>",
                )
            )

            # 2-Sigma Covariance Ellipse
            t_ell = np.linspace(0, 2.0 * np.pi, 150)
            circle_pts = np.vstack([np.cos(t_ell), np.sin(t_ell)])
            # Transform circle by sqrt(Lambda) and Eigenvector matrix E
            L_sqrt = np.diag([2.0 * np.sqrt(lam1), 2.0 * np.sqrt(lam2)])
            ellipse_pts = (np.column_stack([e1, e2]) @ L_sqrt @ circle_pts).T

            fig_2d.add_trace(
                go.Scatter(
                    x=ellipse_pts[:, 0],
                    y=ellipse_pts[:, 1],
                    mode="lines",
                    name="2σ Covariance Ellipse",
                    line=dict(color="#475569", width=1.5, dash="dash"),
                    hoverinfo="skip",
                )
            )

            # PC1 Eigenvector Axis (Emerald)
            scale1 = 2.4 * np.sqrt(lam1)
            fig_2d.add_trace(
                go.Scatter(
                    x=[-scale1 * e1[0], scale1 * e1[0]],
                    y=[-scale1 * e1[1], scale1 * e1[1]],
                    mode="lines+text",
                    name=f"PC1 Axis (e₁, {pct_pc1:.0f}% Var)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    text=["", "PC 1"],
                    textposition="top right",
                    textfont=dict(color=COLOR_HYPERPLANE, size=11),
                    hoverinfo="text",
                    hovertext=f"PC1: e₁=[{e1[0]:.2f}, {e1[1]:.2f}], λ₁={lam1:.3f}",
                )
            )

            # PC2 Eigenvector Axis (Purple)
            scale2 = 2.4 * np.sqrt(lam2)
            fig_2d.add_trace(
                go.Scatter(
                    x=[-scale2 * e2[0], scale2 * e2[0]],
                    y=[-scale2 * e2[1], scale2 * e2[1]],
                    mode="lines+text",
                    name=f"PC2 Axis (e₂, {pct_pc2:.0f}% Var)",
                    line=dict(color="#A855F7", width=2.5),
                    text=["", "PC 2"],
                    textposition="top left",
                    textfont=dict(color="#A855F7", size=11),
                    hoverinfo="text",
                    hovertext=f"PC2: e₂=[{e2[0]:.2f}, {e2[1]:.2f}], λ₂={lam2:.3f}",
                )
            )

            # Active Probe Vector a(θ) (Amber)
            scale_a = max(scale1, 3.0)
            fig_2d.add_trace(
                go.Scatter(
                    x=[-scale_a * a_vec[0], scale_a * a_vec[0]],
                    y=[-scale_a * a_vec[1], scale_a * a_vec[1]],
                    mode="lines+text",
                    name=f"Probe Vector a({theta_deg}°)",
                    line=dict(color="#F59E0B", width=2.5, dash="dot"),
                    text=["", f"a({theta_deg}°)<br>Var={active_var:.2f}"],
                    textposition="bottom right",
                    textfont=dict(color="#F59E0B", size=10),
                    hoverinfo="skip",
                )
            )

            fig_2d.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Centered Feature x₁", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    scaleanchor="y",
                    scaleratio=1,
                ),
                yaxis=dict(
                    title=dict(text="Centered Feature x₂", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_2d, width="stretch", key="pcag_fig_2d")

        with tab_curve:
            st.markdown(r"##### 🔄 Projected Variance $\text{Var}(a^T x) = a^T S_x a$ Across All Angles $\theta \in [0^\circ, 180^\circ]$")

            angles_deg = np.linspace(0, 180, 181)
            angles_rad = np.radians(angles_deg)
            var_sweep = [
                float(np.dot([np.cos(rad), np.sin(rad)], S_x @ [np.cos(rad), np.sin(rad)]))
                for rad in angles_rad
            ]

            fig_curve = go.Figure()

            # Theoretical variance curve
            fig_curve.add_trace(
                go.Scatter(
                    x=angles_deg,
                    y=var_sweep,
                    mode="lines",
                    name="Projected Variance Var(aᵀx)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.0),
                    hovertemplate="θ = %{x}°: Var = %{y:.3f}<extra></extra>",
                )
            )

            # Maximum Var Line (lambda 1)
            fig_curve.add_hline(
                y=lam1,
                line_width=1.5,
                line_dash="dash",
                line_color=COLOR_HYPERPLANE,
                annotation_text=f"Max Var = λ₁ ({lam1:.3f}) at θ ≈ {pc1_angle_deg:.1f}°",
                annotation_font_color=COLOR_HYPERPLANE,
                annotation_position="top left",
            )

            # Minimum Var Line (lambda 2)
            fig_curve.add_hline(
                y=lam2,
                line_width=1.5,
                line_dash="dash",
                line_color="#A855F7",
                annotation_text=f"Min Var = λ₂ ({lam2:.3f})",
                annotation_font_color="#A855F7",
                annotation_position="bottom left",
            )

            # Current angle probe marker
            fig_curve.add_trace(
                go.Scatter(
                    x=[theta_deg],
                    y=[active_var],
                    mode="markers",
                    name=f"Current Probe ({theta_deg}°, Var={active_var:.3f})",
                    marker=dict(size=11, color="#F59E0B", symbol="diamond"),
                    hovertemplate="Current θ = %{x}°: Var = %{y:.3f}<extra></extra>",
                )
            )

            fig_curve.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Projection Angle θ (Degrees)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    dtick=30,
                ),
                yaxis=dict(
                    title=dict(text="Projected Variance", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_curve, width="stretch", key="pcag_fig_curve")

        with tab_proj:
            st.markdown(f"##### 📉 1D Transformed Coordinates z = aᵀx Along Probe Angle θ = {theta_deg}°")

            fig_proj = go.Figure()

            fig_proj.add_trace(
                go.Histogram(
                    x=proj_z,
                    nbinsx=30,
                    marker=dict(color="#F59E0B", line=dict(color="#0F172A", width=1)),
                    name=f"Histogram of z = aᵀx (Var = {active_var:.3f})",
                    opacity=0.75,
                )
            )

            fig_proj.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Projected Coordinate z", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Observation Count", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                margin=dict(l=40, r=20, t=15, b=45),
            )

            st.plotly_chart(fig_proj, width="stretch", key="pcag_fig_proj")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Spectral Decomposition of Covariance", expanded=False):
        st.markdown(
            r"""
            **Definition of Principal Components**:
            - A Principal Component (PC) is a linear combination of input features $z = a^T x$ that has maximum variation subject to $\|a\|_2 = 1$.
            - **1st Principal Component**:
              $$z_1 = a_1^T x \quad \text{maximizing } \text{Var}(a_1^T x) = a_1^T \Sigma a_1 \quad \text{s.t. } a_1^T a_1 = 1$$
            - **$i$-th Principal Component**:
              $$z_i = a_i^T x \quad \text{maximizing } \text{Var}(a_i^T x) \quad \text{s.t. } a_i^T a_i = 1 \text{ and } \text{Cov}(a_i^T x, a_k^T x) = 0, \, \forall k < i$$

            **Spectral Theorem Solution**:
            - Let $\Sigma$ be the $p \times p$ sample covariance matrix with sorted eigenvalue/eigenvector pairs $\{(\lambda_i, e_i)\}_{i=1}^p$ ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0$).
            - The optimal weights are the eigenvectors: $a_i = e_i$.
            - The variance of the $i$-th PC is exactly the $i$-th eigenvalue:
              $$\text{Var}(z_i) = e_i^T \Sigma e_i = \lambda_i$$
            - Distinct PCs are completely uncorrelated:
              $$\text{Cov}(z_i, z_k) = e_i^T \Sigma e_k = 0 \quad (i \ne k)$$
            - **Total Variance Preservation**:
              $$\text{tr}(\Sigma_z) = \sum_{i=1}^p \lambda_i = \text{tr}(\Sigma_x) = \sum_{j=1}^p \text{Var}(x_j)$$
            """
        )
