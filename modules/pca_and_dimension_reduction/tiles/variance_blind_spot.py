"""
Tile 3: The Variance Blind Spot (Clustering Loss in PCA)
Focus: Slide 337 - Direct replication of the classic PCA limitation.
Demonstrates that maximizing variance (var(z)) is not equivalent to preserving information/clusters:
when within-cluster noise along x_2 exceeds between-cluster separation along x_1,
PC1 projects along x_2, destroying cluster separability into a unimodal blur (Slide 337b),
while PC2 (smaller variance) retains the clean bimodal separation (Slide 337c).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Cluster & Variance Knobs")

            cluster_sep = st.slider(
                "Between-Cluster Separation Δx₁",
                min_value=1.5,
                max_value=6.0,
                value=3.5,
                step=0.25,
                key="vbs_sep",
                help="Horizontal distance separating the two clusters along x₁.",
            )

            vertical_spread = st.slider(
                "Vertical Within-Cluster Spread σ(x₂)",
                min_value=2.0,
                max_value=9.0,
                value=5.0,
                step=0.5,
                key="vbs_s2",
                help="Vertical variance along x₂ (within-cluster noise). When σ(x₂) >> Δx₁, PCA aligns PC1 vertically with x₂.",
            )

            cluster_tilt_deg = st.slider(
                "Cluster Rotation Angle (°)",
                min_value=-25,
                max_value=25,
                value=0,
                step=1,
                key="vbs_tilt",
                help="Rotate the two clusters to observe how principal component axes follow maximal variance.",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                horizontal_noise = st.slider("Horizontal Noise σ(x₁)", 0.15, 0.80, 0.35, 0.05, key="vbs_s1")
                n_samples = st.slider("Total Sample Count N", 100, 600, 300, 50, key="vbs_n")
                seed_val = st.number_input("Random Seed", value=42, step=1, key="vbs_seed")

    # Generate 2-cluster elongated vertical data matching Slide 337
    rng = np.random.RandomState(seed_val)
    n_half = n_samples // 2

    # Cluster 1 (left)
    x1_c1 = rng.randn(n_half) * horizontal_noise - (cluster_sep / 2.0)
    x2_c1 = rng.randn(n_half) * vertical_spread
    # Cluster 2 (right)
    x1_c2 = rng.randn(n_half) * horizontal_noise + (cluster_sep / 2.0)
    x2_c2 = rng.randn(n_half) * vertical_spread

    X_raw = np.vstack([np.column_stack([x1_c1, x2_c1]), np.column_stack([x1_c2, x2_c2])])
    cluster_labels = np.array([0] * n_half + [1] * n_half)

    # Apply optional tilt
    if cluster_tilt_deg != 0:
        rad = np.radians(cluster_tilt_deg)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        X_raw = X_raw @ R.T

    # Center data matrix (Slide 333)
    X = X_raw - np.mean(X_raw, axis=0)

    # Fit PCA
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    e1, e2 = pca.components_[0], pca.components_[1]
    lam1, lam2 = pca.explained_variance_[0], pca.explained_variance_[1]
    pct1, pct2 = pca.explained_variance_ratio_[0] * 100.0, pca.explained_variance_ratio_[1] * 100.0

    z1 = Z[:, 0]  # First PC coordinates
    z2 = Z[:, 1]  # Second PC coordinates

    # Measure cluster overlap / separability on 1D projections via simple threshold accuracy
    # Check if z1 separates clusters vs z2 separates clusters
    acc_z1 = max(
        float(np.mean((z1 > 0) == cluster_labels)),
        float(np.mean((z1 <= 0) == cluster_labels)),
    ) * 100.0
    acc_z2 = max(
        float(np.mean((z2 > 0) == cluster_labels)),
        float(np.mean((z2 <= 0) == cluster_labels)),
    ) * 100.0

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PC1 Variance λ₁", f"{lam1:.2f}", f"{pct1:.1f}% of total")
        m2.metric("PC2 Variance λ₂", f"{lam2:.2f}", f"{pct2:.1f}% of total")
        m3.metric("PC1 Cluster Accuracy", f"{acc_z1:.1f}%", help="Projection onto PC1 (Largest Variance) destroys clusters!")
        m4.metric("PC2 Cluster Accuracy", f"{acc_z2:.1f}%", help="Projection onto PC2 (Smallest Variance) preserves clusters!")

        tab_2d, tab_hist, tab_comparison = st.tabs([
            "🌐 2D Data & Principal Axes",
            "📉 Histograms of PC1 vs PC2",
            "🛡️ Variance vs Information Diagnostic",
        ])

        with tab_2d:
            st.markdown("##### 🌐 Two Elongated Clusters with Principal Axes e₁ (Emerald) and e₂ (Purple)")

            fig_2d = go.Figure()

            # Cluster 1 (Left, Rose)
            fig_2d.add_trace(
                go.Scatter(
                    x=X[cluster_labels == 0, 0],
                    y=X[cluster_labels == 0, 1],
                    mode="markers",
                    marker=dict(color="#38BDF8", size=6, opacity=0.8),
                    name="Cluster 1 (Left)",
                    hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>",
                )
            )

            # Cluster 2 (Right, Sky Blue)
            fig_2d.add_trace(
                go.Scatter(
                    x=X[cluster_labels == 1, 0],
                    y=X[cluster_labels == 1, 1],
                    mode="markers",
                    marker=dict(color="#F43F5E", size=6, opacity=0.8),
                    name="Cluster 2 (Right)",
                    hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>",
                )
            )

            # PC1 Axis (Vertical direction of largest variance)
            scale1 = 2.2 * np.sqrt(lam1)
            fig_2d.add_trace(
                go.Scatter(
                    x=[-scale1 * e1[0], scale1 * e1[0]],
                    y=[-scale1 * e1[1], scale1 * e1[1]],
                    mode="lines+text",
                    name=f"PC 1 Axis (λ₁={lam1:.1f}, {pct1:.0f}% Var)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    text=["", "PC 1"],
                    textposition="top right",
                    textfont=dict(color=COLOR_HYPERPLANE, size=11),
                    hoverinfo="text",
                    hovertext=f"PC1: e₁=[{e1[0]:.2f}, {e1[1]:.2f}]",
                )
            )

            # PC2 Axis (Horizontal direction of true cluster separation)
            scale2 = 2.5 * np.sqrt(lam2)
            fig_2d.add_trace(
                go.Scatter(
                    x=[-scale2 * e2[0], scale2 * e2[0]],
                    y=[-scale2 * e2[1], scale2 * e2[1]],
                    mode="lines+text",
                    name=f"PC 2 Axis (λ₂={lam2:.1f}, {pct2:.0f}% Var)",
                    line=dict(color="#A855F7", width=2.8),
                    text=["", "PC 2"],
                    textposition="bottom right",
                    textfont=dict(color="#A855F7", size=11),
                    hoverinfo="text",
                    hovertext=f"PC2: e₂=[{e2[0]:.2f}, {e2[1]:.2f}]",
                )
            )

            fig_2d.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Feature x₁ (Cluster Signal)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Feature x₂ (Within-Cluster Noise)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_2d, width="stretch", key="vbs_fig_2d")

        with tab_hist:
            st.markdown("##### 📉 Histograms of PC1 vs PC2 Projections")

            # We create a dual histogram figure comparing PC1 (unimodal) vs PC2 (bimodal)
            fig_hist = go.Figure()

            # Histogram of PC1 (Unimodal blur)
            fig_hist.add_trace(
                go.Histogram(
                    x=z1,
                    nbinsx=35,
                    marker=dict(color="#EF4444", line=dict(color="#0F172A", width=0.8)),
                    name="1st PC Projection: Unimodal Blur (Clusters Lost!)",
                    opacity=0.75,
                )
            )

            # Histogram of PC2 (Bimodal separation)
            fig_hist.add_trace(
                go.Histogram(
                    x=z2,
                    nbinsx=35,
                    marker=dict(color=COLOR_HYPERPLANE, line=dict(color="#0F172A", width=0.8)),
                    name="2nd PC Projection: Bimodal Peaks (Clusters Preserved!)",
                    opacity=0.75,
                )
            )

            fig_hist.update_layout(
                barmode="overlay",
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Projected Principal Coordinate z", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_hist, width="stretch", key="vbs_fig_hist")

        with tab_comparison:
            st.markdown(
                r"""
                ##### 🛡️ Theoretical Lesson: Variance is NOT Equivalent to Information
                
                - **The Core Flaw**: PCA is completely **unsupervised** and defines "information" purely as **variance** ($\text{Var}(z)$).
                - When within-cluster noise or measurement variance along an uninformative axis ($x_2$) is large ($\sigma_{x_2} > \Delta x_1$), PCA automatically aligns the 1st Principal Component along the noise axis ($e_1 \approx [0, 1]^T$).
                - If an analyst chooses $q = 1$ to reduce data dimension to 1D:
                  - **Projecting onto PC 1**: Collapses the two separate clusters directly on top of each other into a unimodal distribution ($\text{Accuracy} \approx 50\%$). The cluster structure is completely annihilated!
                  - **Projecting onto PC 2**: The smaller-variance axis captures the true between-cluster separation, cleanly resolving the two clusters into two isolated bimodal distributions ($\text{Accuracy} \approx 100\%$).
                - **Crucial Rule**: In classification, clustering, or discriminant problems, high-variance directions often contain irrelevant noise, while low-variance directions may carry the critical class boundaries.
                """
            )

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Limitation I: Variance Blind Spot", expanded=False):
        st.markdown(
            r"""
            **Why PCA Fails to Preserve Discrete Cluster Geometry**:
            - Let data be generated from a mixture of two Gaussian clusters with means $\mu_1 = (-\Delta/2, 0)^T$ and $\mu_2 = (\Delta/2, 0)^T$ and within-cluster covariance:
              $$\Sigma_{\text{within}} = \begin{pmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{pmatrix}$$
            - The total covariance matrix is:
              $$\Sigma = \Sigma_{\text{within}} + \frac{1}{4} \begin{pmatrix} \Delta^2 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} \sigma_1^2 + \frac{\Delta^2}{4} & 0 \\ 0 & \sigma_2^2 \end{pmatrix}$$
            - If the within-cluster vertical variance is large such that $\sigma_2^2 > \sigma_1^2 + \frac{\Delta^2}{4}$, the maximum eigenvalue $\lambda_1 = \sigma_2^2$ corresponds to eigenvector $e_1 = (0, 1)^T$.
            - Thus, **1-dimensional PCA ($q=1$) projects entirely onto the vertical coordinate $x_2$**, discarding $x_1$ and destroying the entire 2-cluster structure!
            """
        )
