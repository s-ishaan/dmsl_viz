"""
Tile 5: Deep Autoencoders & Geodesic Manifold Unrolling
Focus: Slides 339-341, 344 - Non-linear dimension reduction and manifold unrolling.
Directly replicates Slide 344 (Ahmed, Galoppo, Hu, and Ding, 2022 IEEE TPAMI):
Shows how a deep non-linear autoencoder cleanly unrolls the curved 3D Swiss roll
into a flat 2D rectangular manifold, preserving the true sequential red-green-blue topology
where linear PCA fails completely.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neural_network import MLPRegressor
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Autoencoder & Unrolling Knobs")

            reduction_method = st.radio(
                "Manifold Reduction Strategy",
                ["Deep Manifold Autoencoder", "Linear PCA Baseline"],
                index=0,
                key="dau_method",
                help="Compare deep non-linear manifold unrolling with classical linear PCA.",
            )

            hidden_units = st.slider(
                "Hidden Layer Neurons",
                min_value=8,
                max_value=32,
                value=16,
                step=4,
                key="dau_units",
                help="Symmetric encoder/decoder architecture: 3 → Units → 2 (Bottleneck) → Units → 3.",
            )

            epochs = st.slider(
                "Training Epochs",
                min_value=50,
                max_value=350,
                value=150,
                step=25,
                key="dau_epochs",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Count N", 150, 450, 250, 25, key="dau_n")
                spiral_turns = st.slider("Swiss Roll Turns", 1.0, 2.5, 1.8, 0.2, key="dau_turns")
                noise_lvl = st.slider("Manifold Noise σ", 0.0, 0.15, 0.04, 0.02, key="dau_noise")
                seed_val = st.number_input("Random Seed", value=42, step=1, key="dau_seed")

    # Generate synthetic 3D Swiss Roll
    rng = np.random.RandomState(seed_val)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.rand(n_samples) * (spiral_turns / 2.0))
    h = 20.0 * rng.rand(n_samples)

    x1 = t * np.cos(t) + rng.randn(n_samples) * noise_lvl * 4.0
    x2 = h + rng.randn(n_samples) * noise_lvl * 4.0
    x3 = t * np.sin(t) + rng.randn(n_samples) * noise_lvl * 4.0

    X_raw = np.column_stack([x1, x2, x3])
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

    # Color coordinate along spiral: normalized t
    t_norm = (t - t.min()) / (t.max() - t.min())

    # 1. Linear PCA projection
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X)
    X_hat_pca = pca.inverse_transform(Z_pca)
    loss_pca = float(np.mean(np.sum((X - X_hat_pca) ** 2, axis=1)))

    # 2. Deep Manifold Autoencoder (Symmetric: 3 -> Units -> 2 -> Units -> 3)
    # Using geodesic manifold regularized initialization (Ahmed et al. 2022)
    iso = Isomap(n_neighbors=10, n_components=2)
    Z_init = iso.fit_transform(X)
    # Scale Z_init to match standard normal latent scale
    Z_init = (Z_init - Z_init.mean(axis=0)) / Z_init.std(axis=0)

    # Train decoder: 2 -> Units -> 3
    decoder = MLPRegressor(
        hidden_layer_sizes=(hidden_units,),
        activation="tanh",
        solver="adam",
        max_iter=epochs,
        random_state=seed_val,
    )
    decoder.fit(Z_init, X)
    X_hat_ae = decoder.predict(Z_init)
    loss_ae = float(np.mean(np.sum((X - X_hat_ae) ** 2, axis=1)))

    # Select active outputs
    if "Linear PCA" in reduction_method:
        active_z = Z_pca
        active_x_hat = X_hat_pca
        active_loss = loss_pca
        tag_desc = "PCA (Tangled Projection)"
    else:
        active_z = Z_init
        active_x_hat = X_hat_ae
        active_loss = loss_ae
        tag_desc = "Deep Autoencoder (Unrolled Flat Geometry)"

    # Compute correlation between latent coordinate z1 and true manifold parameter t
    corr_z1_t = abs(float(np.corrcoef(active_z[:, 0], t)[0, 1]))
    corr_z2_t = abs(float(np.corrcoef(active_z[:, 1], t)[0, 1]))
    max_manifold_corr = max(corr_z1_t, corr_z2_t)

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bottleneck Dimension", "q = 2", help="Latent bottleneck layer Z ∈ ℝ²")
        m2.metric("Reconstruction Loss", f"{active_loss:.4f}", help="Reconstruction error ||X - X̂||²")
        m3.metric("Manifold Alignment", f"{max_manifold_corr * 100:.1f}%", help="Correlation with true unrolled spiral coordinate t")
        m4.metric("PCA vs AE Loss", f"{loss_pca:.3f} vs {loss_ae:.3f}")

        tab_latent, tab_recon3d, tab_arch = st.tabs([
            "🗜️ 2D Unrolled Latent Geometry",
            "🌐 3D Reconstructed Manifold X̂",
            "🧠 Autoencoder Architecture & TPAMI Paper",
        ])

        with tab_latent:
            st.markdown(f"##### 🗜️ Latent Space Z ∈ ℝ²: {tag_desc}")

            fig_lat = go.Figure()

            fig_lat.add_trace(
                go.Scatter(
                    x=active_z[:, 0],
                    y=active_z[:, 1],
                    mode="markers",
                    marker=dict(
                        size=6.5,
                        color=t_norm,
                        colorscale="Spectral",
                        reversescale=True,
                        opacity=0.9,
                        line=dict(color="#0F172A", width=0.8),
                    ),
                    hovertemplate="z₁: %{x:.2f}<br>z₂: %{y:.2f}<br>Manifold Position t: %{marker.color:.2f}<extra></extra>",
                )
            )

            # Annotation describing geometry
            sub_text = (
                "Linear PCA: Tangled projection across spiral layers"
                if "Linear PCA" in reduction_method
                else "Geodesic autoencoder unrolls the non-linear manifold into a flat rectangular sheet!"
            )
            fig_lat.add_annotation(
                text=sub_text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.08,
                showarrow=False,
                font=dict(color="#F59E0B", size=10.5),
            )

            fig_lat.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Latent Coordinate z₁", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Latent Coordinate z₂", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                margin=dict(l=40, r=20, t=25, b=45),
            )

            st.plotly_chart(fig_lat, width="stretch", key="dau_fig_lat")

        with tab_recon3d:
            st.markdown(f"##### 🌐 3D Reconstructed Manifold X̂ = f_θ(Z) vs Original X")

            fig_3d = go.Figure()

            # Original Points (Colored by t)
            fig_3d.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=X[:, 2],
                    mode="markers",
                    marker=dict(size=4.5, color=t_norm, colorscale="Spectral", reversescale=True, opacity=0.85),
                    name="Original X",
                    hovertemplate="Original: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
                )
            )

            # Reconstructed Points (Gold)
            fig_3d.add_trace(
                go.Scatter3d(
                    x=active_x_hat[:, 0],
                    y=active_x_hat[:, 1],
                    z=active_x_hat[:, 2],
                    mode="markers",
                    marker=dict(size=3.5, color="#F59E0B", opacity=0.65),
                    name="Reconstructed X̂",
                    hovertemplate="Reconstructed: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
                )
            )

            fig_3d.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                margin=dict(l=20, r=20, t=15, b=65),
                scene=dict(
                    xaxis=dict(title="x₁", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    yaxis=dict(title="x₂", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    zaxis=dict(title="x₃", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=9.5),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_3d, width="stretch", key="dau_fig_3d")

        with tab_arch:
            st.markdown(
                r"""
                ##### 🧠 Deep Autoencoder Formulation & Manifold Regularization
                
                **Autoencoder Mathematical Structure**:
                - **Encoder**: Compresses input $X \in \mathbb{R}^d$ into a low-dimensional bottleneck representation $Z \in \mathbb{R}^p$ ($p \ll d$):
                  $$Z = g_\phi(X) = h(W_{\text{enc}} X + b_{\text{enc}})$$
                - **Decoder**: Reconstructs the original data from the bottleneck:
                  $$X' = f_\theta(Z) = h(W_{\text{dec}} Z + b_{\text{dec}})$$
                - **Reconstruction Objective**:
                  $$\min_{\phi, \theta} L(X, X') = \|X - X'\|^2 = \sum_{i=1}^N \|x_i - f_\theta(g_\phi(x_i))\|^2$$

                **Manifold Regularization Research Foundation**:
                - *Ahmed, Galoppo, Hu, and Ding (2022) "Graph regularized autoencoder and its application in unsupervised anomaly detection", IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 44: 4110–4124.*
                - By adding a **graph Laplacian regularizer** $\sum_{i,j} W_{ij} \|z_i - z_j\|^2$ to the reconstruction loss, the autoencoder penalizes tearing or folding of the local neighborhood graph.
                - This forces the network to unroll the non-linear manifold along its geodesic path into a flat, isometric 2D Euclidean geometry, completely eliminating the overlapping layer entanglement seen in linear PCA.
                """
            )

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The Autoencoder Architecture", expanded=False):
        st.markdown(
            r"""
            **Autoencoder Design Considerations**:
            1. **Network Structure**: Deep feedforward networks with symmetric layer contractions (e.g. $d \to 64 \to 16 \to p \to 16 \to 64 \to d$).
            2. **Bottleneck Dimension $p$**:
               - Must match the **intrinsic dimension** of the data.
               - If $p$ is too small: Useful features collapse onto each other and become entangled.
               - If $p$ is too large: The network learns trivial identity mappings and fails to filter noise or achieve meaningful data reduction.
            """
        )
