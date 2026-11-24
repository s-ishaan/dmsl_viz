"""
Tile 5: DDPM Reverse Denoising & Sampling Dynamics
Focus: Slides 358-361 - The Denoising Process (1-4), matching the reverse posterior,
the noise prediction objective min_θ ||ϵ - ϵ_θ(x_t, t)||^2, and Algorithm 2 Sampling.
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


def get_target_centers(topology: str) -> np.ndarray:
    if topology == "Two Moons":
        return np.array([[-1.5, 0.8], [1.5, -0.8]])
    elif topology == "4 Modes":
        return np.array([[-1.8, -1.8], [-1.8, 1.8], [1.8, -1.8], [1.8, 1.8]])
    else:  # Concentric Rings
        return np.array([[0.0, 0.0]])


def simulate_reverse_step(
    xt: np.ndarray,
    t: int,
    total_T: int,
    target_centers: np.ndarray,
    sampling_mode: str,
    model_accuracy: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates reverse sampling step from xt to x_{t-1}.
    Recreates Ho et al. (2020) Algorithm 2:
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps_theta) + sigma_t * z
    """
    np.random.seed(seed + t)
    n = len(xt)

    beta_t = 0.0001 + (t / total_T) * (0.02 - 0.0001)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = max(1e-6, np.exp(-0.5 * (t / total_T) ** 2 * 20.0))
    sigma_t = np.sqrt(beta_t)

    # True score points toward nearest target center
    eps_ideal = np.zeros_like(xt)
    for i, p in enumerate(xt):
        dists = np.sum((target_centers - p) ** 2, axis=1)
        nearest = target_centers[np.argmin(dists)]
        # Noise component eps is roughly (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)
        disp = p - nearest
        norm_disp = np.linalg.norm(disp)
        if norm_disp > 1e-6:
            eps_ideal[i] = disp / norm_disp * min(2.5, norm_disp)
        else:
            eps_ideal[i] = np.zeros(2)

    # Neural network prediction with potential model inaccuracy
    eps_noise = np.random.randn(*xt.shape) * 0.4 * (1.0 - model_accuracy)
    eps_theta = model_accuracy * eps_ideal + eps_noise

    # Denoising update
    mean_step = (1.0 / np.sqrt(alpha_t)) * (xt - ((1.0 - alpha_t) / np.sqrt(max(1e-6, 1.0 - alpha_bar_t))) * eps_theta)

    if sampling_mode == "Stochastic DDPM (with noise z)" and t > 1:
        z = np.random.randn(*xt.shape)
        xt_prev = mean_step + sigma_t * z
    else:  # Deterministic DDIM
        xt_prev = mean_step

    return xt_prev, eps_theta


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Reverse Sampling Knobs")

            total_T = 100

            rev_t = st.slider(
                "Reverse Denoising Step (T → t)",
                min_value=0,
                max_value=total_T,
                value=50,
                step=5,
                key="ddpm_rev_t",
                help="Start from random noise at t=100 and step downward to recovered data at t=0.",
            )

            sampling_mode = st.radio(
                "Sampling Algorithm",
                [
                    "Stochastic DDPM (with noise z)",
                    "Deterministic DDIM (zero noise z=0)",
                ],
                index=0,
                key="ddpm_sampling_mode",
                help="DDPM adds Gaussian perturbation σ_t z at each step; DDIM traces a deterministic probability flow ODE.",
            )

            topology = st.selectbox(
                "Target Manifold Topology",
                ["Two Moons", "4 Modes", "Concentric Rings"],
                index=0,
                key="ddpm_topology",
            )

            model_accuracy = st.slider(
                "Noise Predictor Quality (1 - Loss)",
                min_value=0.5,
                max_value=1.0,
                value=0.95,
                step=0.05,
                key="ddpm_accuracy",
                help="Simulates training convergence of the neural network ϵ_θ(x_t, t).",
            )

            with st.expander("⚙️ Generation Settings", expanded=False):
                n_particles = st.slider("Particle Count N", 100, 500, 300, 50, key="ddpm_particles")
                seed_val = st.number_input("Random Seed", 0, 9999, 101, 1, key="ddpm_seed")

    target_centers = get_target_centers(topology)

    # Initialize pure Gaussian noise at t = T
    np.random.seed(int(seed_val))
    x_curr = np.random.randn(n_particles, 2) * 1.8

    # Simulate reverse diffusion steps from T down to rev_t
    history_snapshots = {total_T: x_curr.copy()}
    key_checkpoints = [100, 75, 50, 25, 0]

    last_eps = np.zeros_like(x_curr)
    for step_t in range(total_T, rev_t, -1):
        x_curr, last_eps = simulate_reverse_step(
            x_curr,
            step_t,
            total_T,
            target_centers,
            sampling_mode,
            model_accuracy=model_accuracy,
            seed=int(seed_val),
        )
        if step_t in key_checkpoints:
            history_snapshots[step_t] = x_curr.copy()

    history_snapshots[rev_t] = x_curr.copy()

    # Metrics
    mean_disp = float(np.mean(np.linalg.norm(x_curr, axis=1)))
    pred_mse = float(np.mean(last_eps ** 2) * (1.0 - model_accuracy))

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reverse Step", f"t = {rev_t} of {total_T}")
        m2.metric("Sampling Method", "DDPM (Stochastic)" if "Stochastic" in sampling_mode else "DDIM (ODE)")
        m3.metric("Noise Pred MSE", f"{pred_mse:.4f}", help="||ϵ - ϵ_θ(x_t, t)||² matching the simplified DDPM objective")
        m4.metric("Cohesion Radius", f"{mean_disp:.2f}")

        tab_state, tab_vector_field, tab_timeline = st.tabs([
            "🔄 Denoised Particle State x_t",
            "🧭 Score & Denoising Vector Field",
            "🎞️ Step-by-Step Denoising Timeline",
        ])

        with tab_state:
            st.markdown(rf"##### 🔄 Reverse Trajectory at $t = {rev_t}$: $x_{{t-1}} = \frac{{1}}{{\sqrt{{\alpha_t}}}} \left(x_t - \frac{{1 - \alpha_t}}{{\sqrt{{1 - \bar{{\alpha}}_t}}}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$")

            fig_rev = go.Figure()

            # Target mode markers
            fig_rev.add_trace(
                go.Scatter(
                    x=target_centers[:, 0],
                    y=target_centers[:, 1],
                    mode="markers+text",
                    marker=dict(size=14, color=COLOR_HYPERPLANE, symbol="star", line=dict(color="#FFFFFF", width=1.5)),
                    text=[f"Mode {i+1}" for i in range(len(target_centers))],
                    textposition="top center",
                    textfont=dict(color=COLOR_HYPERPLANE, size=10),
                    name="Target Modes (x₀ Ground Truth)",
                )
            )

            # Current state particles
            state_color = COLOR_CLASS_POS if rev_t < 30 else ("#F59E0B" if rev_t < 70 else COLOR_CLASS_NEG)
            fig_rev.add_trace(
                go.Scatter(
                    x=x_curr[:, 0],
                    y=x_curr[:, 1],
                    mode="markers",
                    marker=dict(size=6.0, color=state_color, opacity=0.85, line=dict(color="#0F172A", width=0.8)),
                    name=f"Particles at t={rev_t}",
                    hovertemplate="Position: (%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )

            # Reference 2-sigma circle
            theta_c = np.linspace(0, 2 * np.pi, 100)
            fig_rev.add_trace(
                go.Scatter(
                    x=2.0 * np.cos(theta_c),
                    y=2.0 * np.sin(theta_c),
                    mode="lines",
                    line=dict(color="#475569", width=1.5, dash="dot"),
                    name="Prior Noise Boundary N(0, I)",
                    hoverinfo="none",
                )
            )

            fig_rev.update_layout(
                height=440,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Feature x₁", range=[-3.8, 3.8], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                yaxis=dict(title="Feature x₂", range=[-3.8, 3.8], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                margin=dict(l=40, r=20, t=20, b=45),
                legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(color="#CBD5E1", size=9)),
            )

            st.plotly_chart(fig_rev, width="stretch", key="ddpm_fig_rev")

        with tab_vector_field:
            st.markdown(r"##### 🧭 Predicted Denoising Field $-\epsilon_\theta(x_t, t) \propto \nabla_{x_t} \log p_t(x_t)$")

            # Create grid of points to evaluate predicted score vector field
            gx = np.linspace(-3.2, 3.2, 14)
            gy = np.linspace(-3.2, 3.2, 14)
            g_xx, g_yy = np.meshgrid(gx, gy)
            grid_pts = np.stack([g_xx.ravel(), g_yy.ravel()], axis=-1)

            # Evaluate predicted denoising direction (pushes toward nearest mode)
            arrows_x, arrows_y = [], []
            for p in grid_pts:
                dists = np.sum((target_centers - p) ** 2, axis=1)
                nearest = target_centers[np.argmin(dists)]
                direction = nearest - p
                norm_d = np.linalg.norm(direction)
                if norm_d > 0.05:
                    u = direction / norm_d * 0.28
                else:
                    u = np.zeros(2)
                arrows_x.append(u[0])
                arrows_y.append(u[1])

            fig_field = go.Figure()

            # Quiver plot as lines with cone markers
            fig_field.add_trace(
                go.Scatter(
                    x=grid_pts[:, 0],
                    y=grid_pts[:, 1],
                    mode="markers",
                    marker=dict(size=3.0, color="#64748B"),
                    showlegend=False,
                    hoverinfo="none",
                )
            )

            for i in range(len(grid_pts)):
                fig_field.add_trace(
                    go.Scatter(
                        x=[grid_pts[i, 0], grid_pts[i, 0] + arrows_x[i]],
                        y=[grid_pts[i, 1], grid_pts[i, 1] + arrows_y[i]],
                        mode="lines",
                        line=dict(color="#38BDF8", width=1.5),
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

            # Target modes
            fig_field.add_trace(
                go.Scatter(
                    x=target_centers[:, 0],
                    y=target_centers[:, 1],
                    mode="markers",
                    marker=dict(size=14, color=COLOR_HYPERPLANE, symbol="star"),
                    name="Target Modes",
                )
            )

            fig_field.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Feature x₁", range=[-3.5, 3.5], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                yaxis=dict(title="Feature x₂", range=[-3.5, 3.5], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                margin=dict(l=40, r=20, t=20, b=45),
            )

            st.plotly_chart(fig_field, width="stretch", key="ddpm_fig_field")

        with tab_timeline:
            st.markdown("##### 🎞️ Denoising Filmstrip: From Random Noise ($t=100$) to Clean Data ($t=0$)")

            film_steps = [100, 75, 50, 25, 0]
            f_cols = st.columns(len(film_steps))

            for col_f, stp in zip(f_cols, film_steps):
                with col_f:
                    st.caption(f"**Step t = {stp}**")
                    if stp in history_snapshots:
                        pts = history_snapshots[stp]
                    else:
                        pts = history_snapshots[min(history_snapshots.keys(), key=lambda k: abs(k - stp))]

                    col_dot = COLOR_CLASS_POS if stp <= 25 else ("#F59E0B" if stp <= 50 else COLOR_CLASS_NEG)
                    fig_snap = go.Figure(
                        go.Scatter(
                            x=pts[:, 0],
                            y=pts[:, 1],
                            mode="markers",
                            marker=dict(size=3.5, color=col_dot, opacity=0.8),
                            hoverinfo="none",
                        )
                    )
                    fig_snap.update_layout(
                        height=180,
                        paper_bgcolor=BG_PAPER,
                        plot_bgcolor=BG_PLOT,
                        margin=dict(l=5, r=5, t=5, b=5),
                        xaxis=dict(range=[-3.5, 3.5], showticklabels=False, showgrid=False),
                        yaxis=dict(range=[-3.5, 3.5], showticklabels=False, showgrid=False),
                    )
                    st.plotly_chart(fig_snap, width="stretch", key=f"ddpm_snap_{stp}")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The Denoising Process & DDPM Objective", expanded=False):
        st.markdown(
            r"""
            **The Reverse Process Markov Chain**:
            - While the forward noising process $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ is fixed and tractable, the true reverse transition $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ requires integrating over the entire data distribution $q(\mathbf{x}_0)$, which is intractable.
            - We approximate the reverse transition using a parameterized neural network $p_\theta$:
              $$p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$
              $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right)$$

            **Posterior Conditioning on $\mathbf{x}_0$**:
            - Conditioning on the known clean data point $\mathbf{x}_0$, the reverse step is tractable and Gaussian via Bayes' rule:
              $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}\right)$$
              $$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$$

            **Reduction to Noise Prediction (Ho et al., 2020)**:
            - Substituting the reparameterization $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}\right)$ into the posterior mean yields:
              $$\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}\right)$$
            - By fixing $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \beta_t \mathbf{I}$, minimizing the variational upper bound reduces simply to training a neural network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ to predict the added Gaussian noise:
              $$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$
            - **Algorithm 2 Reverse Step**:
              $$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \text{ if } t > 1 \text{ else } \mathbf{0}$$
            """
        )
