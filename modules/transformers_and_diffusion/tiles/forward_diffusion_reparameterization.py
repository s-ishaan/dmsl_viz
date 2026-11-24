"""
Tile 4: Forward Diffusion Markov Chain & Reparameterization
Focus: Slides 356-357 - Forward diffusion (encoding) Markov chain, variance scheduling,
and the Gaussian reparameterization shortcut to jump directly to any timestep t.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.plotting import (
    COLOR_HYPERPLANE,
    COLOR_CLASS_POS,
    COLOR_CLASS_NEG,
    BG_PAPER,
    BG_PLOT,
)


def get_schedule(schedule_type: str, num_steps: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes beta, alpha, alpha_bar, and sigma for different variance schedules."""
    timesteps = np.arange(1, num_steps + 1)

    if schedule_type == "Linear (Standard DDPM)":
        beta_start = 0.0001
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, num_steps)
    elif schedule_type == "Cosine (Improved DDPM)":
        # Nichol & Dhariwal (2021) Cosine schedule
        s = 0.008
        steps = np.arange(num_steps + 1)
        f_t = np.cos(((steps / num_steps) + s) / (1 + s) * np.pi / 2) ** 2
        alphas_bar_full = f_t / f_t[0]
        betas = []
        for t in range(1, num_steps + 1):
            beta = min(0.999, 1.0 - (alphas_bar_full[t] / max(1e-12, alphas_bar_full[t - 1])))
            betas.append(max(0.0001, beta))
        betas = np.array(betas)
    else:  # Quadratic
        betas = np.linspace(0.0001**0.5, 0.02**0.5, num_steps) ** 2

    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    sigmas = np.sqrt(1.0 - alphas_bar)

    return betas, alphas, alphas_bar, sigmas


def generate_source_distribution(dataset_name: str, n_samples: int = 400, seed: int = 42) -> np.ndarray:
    """Generates clean structured 2D source distribution x_0."""
    np.random.seed(seed)

    if dataset_name == "Two Moons":
        n_half = n_samples // 2
        theta1 = np.linspace(0, np.pi, n_half)
        x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=-1)
        theta2 = np.linspace(0, np.pi, n_samples - n_half)
        x2 = np.stack([1.0 - np.cos(theta2), 1.0 - np.sin(theta2) - 0.5], axis=-1)
        X = np.vstack([x1, x2]) * 2.0
        X += np.random.randn(*X.shape) * 0.08
    elif dataset_name == "Swiss Roll":
        t = 1.5 * np.pi * (1.0 + 2.0 * np.random.rand(n_samples))
        x = t * np.cos(t) / 4.0
        y = t * np.sin(t) / 4.0
        X = np.stack([x, y], axis=-1)
        X += np.random.randn(*X.shape) * 0.05
    elif dataset_name == "Concentric Rings":
        n_inner = n_samples // 2
        theta_in = np.random.uniform(0, 2 * np.pi, n_inner)
        r_in = np.random.normal(1.0, 0.1, n_inner)
        x_in = np.stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)], axis=-1)

        theta_out = np.random.uniform(0, 2 * np.pi, n_samples - n_inner)
        r_out = np.random.normal(2.5, 0.12, n_samples - n_inner)
        x_out = np.stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)], axis=-1)
        X = np.vstack([x_in, x_out])
    else:  # 4 Gaussian Clusters
        centers = np.array([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
        cluster_size = n_samples // 4
        parts = []
        for c in centers:
            parts.append(c + np.random.randn(cluster_size, 2) * 0.35)
        X = np.vstack(parts)

    # Standardize to zero mean, unit variance scale
    X = (X - np.mean(X, axis=0)) / (np.std(X) + 1e-8) * 1.5
    return X


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Forward Diffusion Knobs")

            total_T = 1000

            timestep_t = st.slider(
                "Diffusion Timestep t",
                min_value=0,
                max_value=total_T,
                value=250,
                step=10,
                key="fd_timestep",
                help="Position in the forward Markov chain. t=0 is clean data; t=1000 is pure Gaussian noise.",
            )

            schedule_type = st.selectbox(
                "Variance Schedule β_t",
                [
                    "Linear (Standard DDPM)",
                    "Cosine (Improved DDPM)",
                    "Quadratic Schedule",
                ],
                index=0,
                key="fd_schedule",
                help="Controls the noise injection rate β_t across timesteps.",
            )

            dataset_name = st.selectbox(
                "Source Data Topology (x₀)",
                [
                    "Two Moons",
                    "Swiss Roll",
                    "Concentric Rings",
                    "4 Gaussian Clusters",
                ],
                index=0,
                key="fd_dataset",
            )

            with st.expander("⚙️ Simulation Settings", expanded=False):
                n_samples = st.slider("Sample Count N", 100, 600, 350, 50, key="fd_nsamples")
                seed_val = st.number_input("Random Seed", 0, 9999, 42, 1, key="fd_seed")

    # Generate schedules and points
    betas, alphas, alphas_bar, sigmas = get_schedule(schedule_type, total_T)
    x0 = generate_source_distribution(dataset_name, n_samples=n_samples, seed=int(seed_val))

    # Reparameterization shortcut: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    np.random.seed(int(seed_val) + 100)
    eps = np.random.randn(*x0.shape)

    if timestep_t == 0:
        xt = x0.copy()
        alpha_bar_t = 1.0
        beta_t = betas[0]
        sigma_t = 0.0
    else:
        alpha_bar_t = float(alphas_bar[timestep_t - 1])
        beta_t = float(betas[timestep_t - 1])
        sigma_t = float(sigmas[timestep_t - 1])
        xt = np.sqrt(alpha_bar_t) * x0 + sigma_t * eps

    # Signal-to-Noise Ratio (SNR)
    snr_t = alpha_bar_t / max(1e-12, 1.0 - alpha_bar_t)
    log_snr_t = float(np.log10(max(1e-6, snr_t)))

    signal_pct = float(alpha_bar_t * 100)
    noise_pct = float((1.0 - alpha_bar_t) * 100)

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Timestep", f"t = {timestep_t} / {total_T}")
        m2.metric("Remaining Signal √ᾱ_t", f"{np.sqrt(alpha_bar_t):.3f}", f"{signal_pct:.1f}% power")
        m3.metric("Added Noise σ_t", f"{sigma_t:.3f}", f"{noise_pct:.1f}% power")
        m4.metric("Log10 SNR(t)", f"{log_snr_t:.2f} dB", help="Signal-to-Noise Ratio: ᾱ_t / (1 - ᾱ_t)")

        tab_cloud, tab_curves, tab_image_noising = st.tabs([
            "🌫️ 2D Particle Diffusion Cloud x_t",
            "📈 Variance Schedules & SNR Dynamics",
            "🖼️ Pixel Grid Noising Simulation",
        ])

        with tab_cloud:
            st.markdown(rf"##### 🌫️ Forward State $x_t = \sqrt{{\bar{{\alpha}}_t}} x_0 + \sqrt{{1 - \bar{{\alpha}}_t}} \epsilon$ at $t = {timestep_t}$")

            fig_cloud = go.Figure()

            # Clean original x0 ghost points for reference
            fig_cloud.add_trace(
                go.Scatter(
                    x=x0[:, 0],
                    y=x0[:, 1],
                    mode="markers",
                    marker=dict(size=4.5, color="#475569", opacity=0.35),
                    name="Original Data x₀ (Reference)",
                    hoverinfo="none",
                )
            )

            # Active diffused xt points
            cloud_color = COLOR_CLASS_POS if timestep_t < 300 else ("#F59E0B" if timestep_t < 700 else COLOR_CLASS_NEG)
            fig_cloud.add_trace(
                go.Scatter(
                    x=xt[:, 0],
                    y=xt[:, 1],
                    mode="markers",
                    marker=dict(size=6.5, color=cloud_color, opacity=0.85, line=dict(color="#0F172A", width=0.8)),
                    name=f"Noisy State x_t (t={timestep_t})",
                    hovertemplate="x_t: (%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )

            # Circular standard normal 2-sigma boundary for visual orientation
            theta_circ = np.linspace(0, 2 * np.pi, 100)
            fig_cloud.add_trace(
                go.Scatter(
                    x=2.0 * np.cos(theta_circ),
                    y=2.0 * np.sin(theta_circ),
                    mode="lines",
                    line=dict(color="#64748B", width=1.5, dash="dot"),
                    name="Gaussian 2σ Limit N(0, I)",
                    hoverinfo="none",
                )
            )

            fig_cloud.update_layout(
                height=440,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Coordinate x₁", range=[-4.2, 4.2], gridcolor="#1E293B", tickfont=dict(color="#94A3B8", size=9.5)),
                yaxis=dict(title="Coordinate x₂", range=[-4.2, 4.2], gridcolor="#1E293B", tickfont=dict(color="#94A3B8", size=9.5)),
                margin=dict(l=40, r=20, t=20, b=45),
                legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(color="#CBD5E1", size=9)),
            )

            st.plotly_chart(fig_cloud, width="stretch", key="fd_fig_cloud")

        with tab_curves:
            st.markdown(r"##### 📈 Mathematical Trajectories: $\beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$, and $\text{SNR}(t)$")

            steps_plot = np.arange(1, total_T + 1)

            fig_traj = go.Figure()

            fig_traj.add_trace(
                go.Scatter(
                    x=steps_plot,
                    y=alphas_bar,
                    mode="lines",
                    name="Signal Retention ᾱ_t (Cumulative Product)",
                    line=dict(color=COLOR_HYPERPLANE, width=2.5),
                )
            )

            fig_traj.add_trace(
                go.Scatter(
                    x=steps_plot,
                    y=sigmas,
                    mode="lines",
                    name="Noise Amplitude σ_t = √(1 - ᾱ_t)",
                    line=dict(color=COLOR_CLASS_NEG, width=2.5),
                )
            )

            fig_traj.add_trace(
                go.Scatter(
                    x=steps_plot,
                    y=betas * 50.0,  # Scaled for visibility on same axis
                    mode="lines",
                    name="Step Variance β_t (× 50)",
                    line=dict(color="#F59E0B", width=2.0, dash="dash"),
                )
            )

            # Vertical marker at current timestep t
            fig_traj.add_vline(
                x=timestep_t,
                line=dict(color="#38BDF8", width=2, dash="dash"),
                annotation_text=f"t = {timestep_t}",
                annotation_font=dict(color="#38BDF8", size=10),
            )

            fig_traj.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Diffusion Timestep t", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                yaxis=dict(title="Coefficient Magnitude [0, 1]", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                margin=dict(l=40, r=20, t=20, b=45),
                legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(color="#CBD5E1", size=9)),
            )

            st.plotly_chart(fig_traj, width="stretch", key="fd_fig_traj")

        with tab_image_noising:
            st.markdown("##### 🖼️ Visual Synthetic Pixel Patch Diffusion: $x_0 \to x_t \to x_T$")

            # Generate synthetic target pattern (e.g. sharp geometric cross or smiley)
            grid_dim = 16
            yy, xx = np.mgrid[:grid_dim, :grid_dim]
            # Cross pattern
            base_img = np.zeros((grid_dim, grid_dim))
            base_img[5:11, :] = 1.0
            base_img[:, 5:11] = 1.0

            # Add noise according to timestep t
            np.random.seed(int(seed_val) + 200)
            img_noise = np.random.randn(grid_dim, grid_dim)
            noisy_img = np.sqrt(alpha_bar_t) * base_img + sigma_t * img_noise

            fig_img = go.Figure(
                go.Heatmap(
                    z=noisy_img,
                    colorscale="Viridis",
                    showscale=False,
                )
            )
            fig_img.update_layout(
                height=380,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_img, width="stretch", key="fd_fig_img")

            st.caption(f"Synthetic {grid_dim}×{grid_dim} image patch under noise level t={timestep_t} (Signal: {signal_pct:.1f}%, Noise: {noise_pct:.1f}%).")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Forward Diffusion & The Reparameterization Trick", expanded=False):
        st.markdown(
            r"""
            **Forward Gaussian Markov Chain**:
            - Given a data point sampled from the real data distribution $\mathbf{x}_0 \sim q(\mathbf{x}_0)$, the forward diffusion process adds Gaussian noise in $T$ discrete steps according to a variance schedule $\beta_1, \beta_2, \dots, \beta_T$:
              $$q(\mathbf{x}_1, \dots, \mathbf{x}_T \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$
              $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$$
            - By defining $\alpha_t := 1 - \beta_t$, each step transition is:
              $$\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

            **The Closed-Form Reparameterization Trick**:
            - Recursively expanding the transition from $\mathbf{x}_t$ back to $\mathbf{x}_0$:
              $$\mathbf{x}_t = \sqrt{\alpha_t} \left(\sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2}\right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$$
              $$\mathbf{x}_t = \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$$
            - Because the sum of two independent Gaussians $\mathcal{N}(\mathbf{0}, \sigma_1^2 \mathbf{I})$ and $\mathcal{N}(\mathbf{0}, \sigma_2^2 \mathbf{I})$ is $\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$:
              $$\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}$$
            - Defining $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$, induction yields the **closed-form jump equation**:
              $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
            - **Computational Significance**: We never need to iterate through $t$ individual Markov steps during training! We can sample any arbitrary timestep $t \sim \mathcal{U}(1, T)$ in $O(1)$ constant time and evaluate the network directly on $\mathbf{x}_t$.
            """
        )
