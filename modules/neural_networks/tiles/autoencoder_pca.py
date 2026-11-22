"""
Tile 6: Autoencoders & Non-linear Dimension Reduction
Focus: Slides 324-327 - Autoencoders as non-linear Principal Component Analysis (PCA).
Unsupervised dimension reduction via bottleneck mapping g: R^p -> R^q (q < p) and decoder h: R^q -> R^p,
minimizing information loss criterion IF(z, x) = (1/N) * ∑ ||x_i - x̂_i||².
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from utils.datasets import get_swiss_roll_3d
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Autoencoder Knobs")

            latent_q = st.selectbox(
                "Latent Bottleneck Dimension q (p = 3)",
                [2, 1],
                index=0,
                key="ae_q",
                help="Bottleneck dimension q < p. Compresses 3D manifold down to 2D plane or 1D curve.",
            )

            architecture_type = st.radio(
                "Reduction Architecture",
                ["Non-linear Deep Autoencoder", "Linear PCA (Baseline)"],
                index=0,
                key="ae_arch",
                help="Slide 325: Single-layer linear autoencoder is mathematically equivalent to classical PCA.",
            )

            manifold_type = st.selectbox(
                "Manifold Topology",
                ["Swiss Roll Manifold", "3D Spiral Helix", "Noisy S-Curve"],
                index=0,
                key="ae_manifold",
            )

            epochs = st.slider(
                "Autoencoder Training Epochs",
                min_value=50,
                max_value=400,
                value=200,
                step=50,
                key="ae_epochs",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Count N", 100, 450, 250, 25, key="ae_n")
                noise_lvl = st.slider("Manifold Noise", 0.0, 0.20, 0.05, 0.02, key="ae_noise")

    # Generate 3D manifold data
    rng = np.random.RandomState(42)
    if manifold_type == "Swiss Roll Manifold":
        X, t_color = get_swiss_roll_3d(n_samples=n_samples, noise=noise_lvl, random_state=42)
    elif manifold_type == "3D Spiral Helix":
        t = np.linspace(1.0, 4.0 * np.pi, n_samples)
        x1 = t * np.cos(t) + rng.randn(n_samples) * noise_lvl * 2.0
        x2 = t * np.sin(t) + rng.randn(n_samples) * noise_lvl * 2.0
        x3 = t * 1.5 + rng.randn(n_samples) * noise_lvl * 2.0
        X = np.column_stack([x1, x2, x3])
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        t_color = t
    else:
        # S-curve
        t = np.linspace(-np.pi, np.pi, n_samples)
        x1 = np.sin(t) + rng.randn(n_samples) * noise_lvl
        x2 = np.sign(t) * (np.cos(t) - 1.0) + rng.randn(n_samples) * noise_lvl
        x3 = rng.uniform(-1.0, 1.0, n_samples)
        X = np.column_stack([x1, x2, x3])
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        t_color = t

    # 1. Compute PCA Baseline
    pca = PCA(n_components=latent_q, random_state=42)
    Z_pca = pca.fit_transform(X)
    X_hat_pca = pca.inverse_transform(Z_pca)
    loss_pca = float(np.mean(np.sum((X - X_hat_pca) ** 2, axis=1)))

    # 2. Compute Non-linear Autoencoder (Encoder: 3 -> 12 -> q; Decoder: q -> 12 -> 3)
    if latent_q == 2:
        hidden_layers = (14, 2, 14)
    else:
        hidden_layers = (14, 1, 14)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="tanh",
        solver="adam",
        max_iter=epochs,
        random_state=42,
    )
    autoencoder.fit(X, X)
    X_hat_ae = autoencoder.predict(X)
    loss_ae = float(np.mean(np.sum((X - X_hat_ae) ** 2, axis=1)))

    # Extract latent representation from bottleneck
    # Feed forward up to layer 2
    W1 = autoencoder.coefs_[0]
    b1 = autoencoder.intercepts_[0]
    h1 = np.tanh(X @ W1 + b1)
    W2 = autoencoder.coefs_[1]
    b2 = autoencoder.intercepts_[1]
    Z_ae = np.tanh(h1 @ W2 + b2)

    # Active model selection
    if "Linear" in architecture_type:
        active_z = Z_pca
        active_x_hat = X_hat_pca
        active_loss = loss_pca
    else:
        active_z = Z_ae
        active_x_hat = X_hat_ae
        active_loss = loss_ae

    # Calculate information retention ratio
    total_var = float(np.sum(np.var(X, axis=0)))
    retention_pct = max(0.0, min(100.0, (1.0 - active_loss / max(1e-5, total_var)) * 100.0))

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dimension Ratio", f"p={X.shape[1]} → q={latent_q}")
        m2.metric("Info Loss IF(z, x)", f"{active_loss:.4f}", help="Reconstruction error (1/N) * ∑ ||x - x̂||²")
        m3.metric("PCA Loss", f"{loss_pca:.4f}")
        m4.metric("Deep AE Loss", f"{loss_ae:.4f}", delta=f"{loss_ae - loss_pca:+.4f} vs PCA", delta_color="inverse")

        tab_3d, tab_latent, tab_comp = st.tabs([
            "🌐 3D Manifold & Reconstruction Loss (Slide 327)",
            "🗜️ Latent Bottleneck Space z (Slide 325)",
            "📊 Linear PCA vs Deep Autoencoder",
        ])

        with tab_3d:
            st.markdown(f"##### 🌐 3D Manifold Reconstruction: Original x vs Reconstructed x̂ ({architecture_type})")

            fig_3d = go.Figure()

            # Original 3D Points
            fig_3d.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=X[:, 2],
                    mode="markers",
                    marker=dict(size=4.5, color=t_color, colorscale="Viridis", opacity=0.9),
                    name="Original x",
                    hovertemplate="Original x: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
                )
            )

            # Reconstructed Points x̂
            fig_3d.add_trace(
                go.Scatter3d(
                    x=active_x_hat[:, 0],
                    y=active_x_hat[:, 1],
                    z=active_x_hat[:, 2],
                    mode="markers",
                    marker=dict(size=3.5, color="#F59E0B", opacity=0.65),
                    name="Reconstructed x̂",
                    hovertemplate="Reconstructed x̂: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
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

            st.plotly_chart(fig_3d, width="stretch", key="ae_fig_3d")

        with tab_latent:
            st.markdown(f"##### 🗜️ Low-Dimensional Latent Bottleneck Coordinates z ∈ ℝ^{latent_q}")

            if latent_q == 2:
                fig_lat = go.Figure()

                fig_lat.add_trace(
                    go.Scatter(
                        x=active_z[:, 0],
                        y=active_z[:, 1],
                        mode="markers",
                        marker=dict(size=6.5, color=t_color, colorscale="Viridis", opacity=0.85),
                        name="Latent Points",
                        hovertemplate="z₁: %{x:.2f}<br>z₂: %{y:.2f}<extra></extra>",
                    )
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
                    margin=dict(l=40, r=20, t=15, b=45),
                )

                st.plotly_chart(fig_lat, width="stretch", key="ae_fig_lat")
            else:
                # 1D Latent space histogram / strip
                fig_lat1 = go.Figure()

                fig_lat1.add_trace(
                    go.Scatter(
                        x=active_z.ravel(),
                        y=t_color,
                        mode="markers",
                        marker=dict(size=7, color=t_color, colorscale="Viridis"),
                        hovertemplate="z: %{x:.2f}<br>True Manifold t: %{y:.2f}<extra></extra>",
                    )
                )

                fig_lat1.update_layout(
                    height=420,
                    paper_bgcolor=BG_PAPER,
                    plot_bgcolor=BG_PLOT,
                    xaxis=dict(
                        title=dict(text="1D Latent Bottleneck Coordinate z", font=dict(color="#CBD5E1", size=11)),
                        gridcolor="#1E293B",
                        zerolinecolor="#334155",
                        tickfont=dict(color="#94A3B8", size=9.5),
                    ),
                    yaxis=dict(
                        title=dict(text="True Continuous Manifold Parameter t", font=dict(color="#CBD5E1", size=11)),
                        gridcolor="#1E293B",
                        zerolinecolor="#334155",
                        tickfont=dict(color="#94A3B8", size=9.5),
                    ),
                    margin=dict(l=40, r=20, t=15, b=45),
                )

                st.plotly_chart(fig_lat1, width="stretch", key="ae_fig_lat1")

        with tab_comp:
            st.markdown("##### 📊 Information Loss Comparison: Linear PCA vs Non-linear Deep Autoencoder (Slide 325)")

            fig_loss_comp = go.Figure()

            methods = ["Linear PCA", "Deep Non-linear Autoencoder"]
            losses = [loss_pca, loss_ae]
            bar_colors = ["#38BDF8", "#10B981"]

            fig_loss_comp.add_trace(
                go.Bar(
                    x=methods,
                    y=losses,
                    marker=dict(color=bar_colors),
                    text=[f"{l:.4f}" for l in losses],
                    textposition="outside",
                    hovertemplate="%{x}: Information Loss = %{y:.4f}<extra></extra>",
                )
            )

            fig_loss_comp.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(gridcolor="#1E293B", tickfont=dict(color="#CBD5E1", size=11)),
                yaxis=dict(
                    title=dict(text="Information Loss IF(z, x) = ||x - x̂||²", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    range=[0, max(losses) * 1.3],
                ),
                margin=dict(l=40, r=20, t=15, b=45),
            )

            st.plotly_chart(fig_loss_comp, width="stretch", key="ae_fig_loss_comp")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Unsupervised Dimension Reduction (Slides 325–327)", expanded=False):
        st.markdown(
            r"""
            **The Curse of Dimensionality (Slide 326)**:
            - In high-dimensional data spaces, noise from individual uninformative dimensions accumulates.
            - Aggregated noise can completely overwhelm true signal effects, crippling distance metrics and predictive power.
            - **Unsupervised Dimension Reduction** aims to find a low-dimensional mapping that preserves the essential structure of $X$ without relying on labels $Y$.

            **Information Loss Formulation (Slide 327)**:
            - Given training observations $x = \{x_1, \dots, x_N\}$ where each $x_i \in \mathbb{R}^p$.
            - Apply encoder mapping $g: \mathbb{R}^p \to \mathbb{R}^q$ with bottleneck dimension $q < p$:
              $$z_i = g(x_i) \in \mathbb{R}^q$$
            - Apply decoder reconstruction mapping $h: \mathbb{R}^q \to \mathbb{R}^p$:
              $$\hat{x}_i = h(z_i) \in \mathbb{R}^p$$
            - The optimal dimension reduction minimizes information loss $IF(z, x)$:
              $$\{g^*, h^*\} = \arg\min_{g, h} \frac{1}{N} \sum_{i=1}^N \|x_i - h(g(x_i))\|^2$$

            **PCA vs Autoencoder (Slide 325)**:
            - **Linear Autoencoder (PCA)**: When the encoder and decoder mappings are purely linear ($g(x) = W^T x, \, h(z) = W z$), minimizing reconstruction error is mathematically identical to finding the top $q$ eigenvectors of the covariance matrix (Principal Component Analysis).
            - **Deep Non-linear Autoencoder**: By inserting non-linear activation functions (ReLU, Tanh, GELU), the autoencoder can learn complex, twisted non-linear sub-manifolds (such as the Swiss Roll) that linear PCA projects through and destroys.
            """
        )
