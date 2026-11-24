"""
Tile 4: Non-linear Manifold Collapse on Swiss Roll
Focus: Slide 338 - Limitation II of PCA: Linear projection fails on non-linear manifolds.
Directly replicates Slide 338: A 3D Swiss roll colored sequentially along its spiral ribbon.
Shows how linear PCA projects through spiral layers, tangling distant points together,
setting the exact motivation for non-linear Autoencoders and Manifold Learning.
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
            st.markdown("#### 🎛️ Manifold Geometry Knobs")

            n_turns = st.slider(
                "Spiral Turns (Curvature)",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.25,
                key="mc_turns",
                help="Controls how tightly wound the 3D Swiss Roll spiral is.",
            )

            pca_components = st.selectbox(
                "Linear PCA Projections",
                ["PC 1 vs PC 2 (Max Variance 2D)", "PC 1 vs PC 3", "PC 2 vs PC 3"],
                index=0,
                key="mc_pca_proj",
            )

            noise_lvl = st.slider(
                "Manifold Noise σ",
                min_value=0.0,
                max_value=0.20,
                value=0.04,
                step=0.02,
                key="mc_noise",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Count N", 150, 600, 350, 50, key="mc_n")
                seed_val = st.number_input("Random Seed", value=42, step=1, key="mc_seed")

    # Generate synthetic 3D Swiss Roll
    rng = np.random.RandomState(seed_val)
    # Intrinsic parameter t (spiral angle)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.rand(n_samples) * (n_turns / 2.0))
    # Intrinsic parameter h (height)
    h = 21.0 * rng.rand(n_samples)

    # 3D Coordinates
    x1 = t * np.cos(t) + rng.randn(n_samples) * noise_lvl * 5.0
    x2 = h + rng.randn(n_samples) * noise_lvl * 5.0
    x3 = t * np.sin(t) + rng.randn(n_samples) * noise_lvl * 5.0

    X = np.column_stack([x1, x2, x3])
    # Normalize coordinates
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Compute PCA
    pca = PCA(n_components=3)
    Z = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100.0

    # Color map: match Slide 338 (red -> green -> blue along t)
    # Normalized t in [0, 1]
    t_norm = (t - t.min()) / (t.max() - t.min())

    # Selected projection axes
    if "PC 1 vs PC 2" in pca_components:
        px, py = Z[:, 0], Z[:, 1]
        x_label, y_label = f"PC 1 ({var_exp[0]:.1f}% Var)", f"PC 2 ({var_exp[1]:.1f}% Var)"
    elif "PC 1 vs PC 3" in pca_components:
        px, py = Z[:, 0], Z[:, 2]
        x_label, y_label = f"PC 1 ({var_exp[0]:.1f}% Var)", f"PC 3 ({var_exp[2]:.1f}% Var)"
    else:
        px, py = Z[:, 1], Z[:, 2]
        x_label, y_label = f"PC 2 ({var_exp[1]:.1f}% Var)", f"PC 3 ({var_exp[2]:.1f}% Var)"

    # Compute neighborhood entanglement index:
    # Check if 10 nearest neighbors in PCA 2D space are close along the true intrinsic coordinate t
    from sklearn.neighbors import NearestNeighbors
    nn_pca = NearestNeighbors(n_neighbors=8).fit(np.column_stack([px, py]))
    _, idxs = nn_pca.kneighbors()
    # Mean variance of true t_norm among 2D PCA neighbors (0 if perfectly preserved, high if tangled)
    entanglement = float(np.mean([np.std(t_norm[idxs[i]]) for i in range(len(idxs))]))

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Intrinsic Dimension", "d = 2", help="The Swiss roll is fundamentally a 2D surface curled in 3D.")
        m2.metric("Ambient Dimension", "p = 3")
        m3.metric("2-PC Variance", f"{var_exp[0] + var_exp[1]:.1f}%", help="Variance captured by top 2 principal components")
        m4.metric("Entanglement Score", f"{entanglement:.3f}", help="Measures cross-spiral jumbling in linear PCA projection")

        tab_3d, tab_pca_2d, tab_manifold_diag = st.tabs([
            "🌐 3D Swiss Roll Manifold",
            "📉 2D Linear PCA Projection",
            "📏 Geodesic vs Euclidean Distortion",
        ])

        with tab_3d:
            st.markdown("##### 🌐 3D Swiss Roll Manifold: Continuous Color Along Spiral Length (Red → Green → Blue)")

            fig_3d = go.Figure()

            fig_3d.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=X[:, 2],
                    mode="markers",
                    marker=dict(
                        size=4.5,
                        color=t_norm,
                        colorscale="Spectral",
                        reversescale=True,
                        opacity=0.9,
                        colorbar=dict(
                            title=dict(text="Manifold Position t", font=dict(color="#CBD5E1", size=10)),
                            tickfont=dict(color="#94A3B8"),
                        ),
                    ),
                    hovertemplate="3D: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>t: %{marker.color:.2f}<extra></extra>",
                )
            )

            fig_3d.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                margin=dict(l=20, r=20, t=15, b=45),
                scene=dict(
                    xaxis=dict(title="x₁", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    yaxis=dict(title="x₂", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    zaxis=dict(title="x₃", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                ),
            )

            st.plotly_chart(fig_3d, width="stretch", key="mc_fig_3d")

        with tab_pca_2d:
            st.markdown("##### 📉 2D Linear PCA Projection: Spiral Layers Collapsed & Entangled")

            fig_2d = go.Figure()

            fig_2d.add_trace(
                go.Scatter(
                    x=px,
                    y=py,
                    mode="markers",
                    marker=dict(
                        size=6.5,
                        color=t_norm,
                        colorscale="Spectral",
                        reversescale=True,
                        opacity=0.85,
                        line=dict(color="#0F172A", width=0.8),
                    ),
                    hovertemplate="PCA: (%{x:.2f}, %{y:.2f})<br>Manifold Position t: %{marker.color:.2f}<extra></extra>",
                )
            )

            fig_2d.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text=x_label, font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                margin=dict(l=40, r=20, t=15, b=45),
            )

            st.plotly_chart(fig_2d, width="stretch", key="mc_fig_2d")

        with tab_manifold_diag:
            st.markdown(
                r"""
                ##### 📏 The Geometry of Manifold Failure: Euclidean vs Geodesic Distance
                
                - **Linear Projection Geometry**: Linear PCA operates by projecting data perpendicularly onto an affine 2D hyper-plane:
                  $$z = E_q^T x$$
                - **The Manifold Assumption**: Many real-world high-dimensional datasets reside on a smooth low-dimensional **curved Riemannian sub-manifold** $\mathcal{M}$ (such as a 2D surface embedded in 3D).
                - **Why Linear PCA Fails**:
                  - The true distance between two points on the Swiss Roll is their **geodesic distance** $d_{\mathcal{M}}(x_i, x_j)$ measured *along the curved surface*.
                  - Points on adjacent layers of the spiral may be close in 3D ambient space $\|x_i - x_j\|_2$, but are actually vastly far apart along the unrolled ribbon!
                  - Because PCA is strictly linear, it projects directly through the layers. Red and blue points that should be on opposite ends of the ribbon end up on top of each other!
                - **The Solution**: Non-linear Autoencoders and Graph-Regularized Manifold Learning unroll the intrinsic geometry into a flat 2D rectangle without overlap.
                """
            )

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Limitation II: Non-linear Manifolds", expanded=False):
        st.markdown(
            r"""
            **Non-linear Manifold Collapse**:
            - **Linearity Constraint**: Any PCA projection is a linear mapping $z_i = A x_i$. Linear projections preserve collinearity and parallel lines, but can never "unbend", "unroll", or "unfold" non-linear manifolds.
            - When data resides on a curved surface $\mathcal{M} \subset \mathbb{R}^p$ with intrinsic dimension $q \ll p$:
              $$\text{Ambient Dimension: } p = 3, \quad \text{Intrinsic Manifold Dimension: } q = 2$$
            - A linear projection inevitably flattens across the curvature, collapsing distinct topological sheets into the same subspace.
            - To preserve the true manifold topology, one must employ **non-linear dimension reduction**, such as **Deep Autoencoders** or Kernel PCA / ISOMAP.
            """
        )
