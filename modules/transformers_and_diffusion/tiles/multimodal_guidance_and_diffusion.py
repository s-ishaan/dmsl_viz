"""
Tile 6: Multimodal Conditioning, Latent Diffusion & Classifier-Free Guidance
Focus: Slides 354-355, 362 - Text conditioning in image generation,
Classifier-Free Guidance (CFG), Latent Diffusion Pipelines, and the Generative Model Trilemma.
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


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Guidance & Multimodal Knobs")

            prompt_target = st.selectbox(
                "Conditioning Text Prompt",
                [
                    "Prompt A: 'Solar Corona' (Target Cluster A)",
                    "Prompt B: 'Deep Ocean Spiral' (Target Cluster B)",
                    "Prompt C: 'Emerald Aurora' (Target Cluster C)",
                ],
                index=0,
                key="mmg_prompt",
                help="Text prompt encoded by a text model (e.g. CLIP) into continuous conditioning vector c.",
            )

            cfg_scale = st.slider(
                "Classifier-Free Guidance Scale (s)",
                min_value=0.0,
                max_value=12.0,
                value=4.0,
                step=0.5,
                key="mmg_cfg",
                help="Guidance weight s: ϵ̃ = ϵ_uncond + s * (ϵ_cond - ϵ_uncond). s=1 is standard; s>1 pushes generation toward prompt mode.",
            )

            diffusion_space = st.radio(
                "Diffusion Operating Domain",
                [
                    "Latent Space (Stable Diffusion: 64×64×4)",
                    "Pixel Space (Standard DDPM: 512×512×3)",
                ],
                index=0,
                key="mmg_space",
                help="Latent diffusion compresses image pixels through a VAE encoder before applying diffusion, reducing compute by 64x.",
            )

            with st.expander("⚙️ Simulation Settings", expanded=False):
                noise_seed = st.number_input("Latent Seed", 0, 9999, 42, 1, key="mmg_seed")
                num_eval_steps = st.slider("Denoising Steps", 10, 50, 25, 5, key="mmg_steps")

    # Define target mode locations in 2D latent space
    targets = {
        "Prompt A": np.array([2.2, 1.8]),
        "Prompt B": np.array([-2.0, 1.5]),
        "Prompt C": np.array([0.0, -2.4]),
    }

    if "Prompt A" in prompt_target:
        active_key = "Prompt A"
    elif "Prompt B" in prompt_target:
        active_key = "Prompt B"
    else:
        active_key = "Prompt C"

    c_target = targets[active_key]

    # Starting noise point
    np.random.seed(int(noise_seed))
    x_init = np.random.randn(2) * 1.5

    # Simulate trajectory under Classifier-Free Guidance:
    # eps_tilde = eps_uncond + s * (eps_cond - eps_uncond)
    # Unconditioned score points toward center of all data (origin [0, 0])
    # Conditioned score points toward prompt target c_target
    uncond_traj = [x_init.copy()]
    cond_traj = [x_init.copy()]
    cfg_traj = [x_init.copy()]

    curr_uncond = x_init.copy()
    curr_cond = x_init.copy()
    curr_cfg = x_init.copy()

    for step in range(num_eval_steps):
        decay = 1.0 - (step / num_eval_steps)
        step_size = 0.12 * decay

        # Unconditioned velocity (drifts to origin/generic mode)
        v_uncond = (-curr_uncond) * 0.6
        curr_uncond += v_uncond * step_size
        uncond_traj.append(curr_uncond.copy())

        # Conditioned velocity (points to c_target)
        v_cond = (c_target - curr_cond) * 1.0
        curr_cond += v_cond * step_size
        cond_traj.append(curr_cond.copy())

        # CFG combined velocity: v_uncond + s * (v_cond - v_uncond)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        # Cap step to prevent numerical explosion at high scale s
        v_cfg_norm = np.linalg.norm(v_cfg)
        if v_cfg_norm > 4.0:
            v_cfg = v_cfg / v_cfg_norm * 4.0
        curr_cfg += v_cfg * step_size
        cfg_traj.append(curr_cfg.copy())

    uncond_traj = np.array(uncond_traj)
    cond_traj = np.array(cond_traj)
    cfg_traj = np.array(cfg_traj)

    final_dist = float(np.linalg.norm(cfg_traj[-1] - c_target))
    prompt_fidelity = max(0.0, min(100.0, (1.0 - final_dist / 3.0) * 100))

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Guidance s", f"s = {cfg_scale:.1f}", "CFG Boost" if cfg_scale > 1.0 else "Normal")
        m2.metric("Alignment", f"{prompt_fidelity:.1f}%")
        m3.metric("Domain", "Latent (64×)" if "Latent" in diffusion_space else "Pixel (1×)")
        m4.metric("Target Mode", active_key.replace("Prompt ", "Mode "))

        tab_cfg_plane, tab_pipeline_diag, tab_trilemma = st.tabs([
            "🧭 CFG Trajectory",
            "🏗️ Latent Pipeline",
            "⚖️ Generative Trilemma",
        ])

        with tab_cfg_plane:
            st.markdown(r"##### 🧭 Steering the Diffusion Path: $\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + s \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset))$")

            fig_cfg = go.Figure()

            # Target mode clusters
            for p_name, p_loc in targets.items():
                is_active = (p_name == active_key)
                col = COLOR_HYPERPLANE if is_active else "#64748B"
                fig_cfg.add_trace(
                    go.Scatter(
                        x=[p_loc[0]],
                        y=[p_loc[1]],
                        mode="markers+text",
                        marker=dict(size=16 if is_active else 10, color=col, symbol="star"),
                        text=[f"<b>{p_name}</b>" if is_active else p_name],
                        textposition="top center",
                        textfont=dict(color=col, size=11 if is_active else 9),
                        name=f"Prompt Target ({p_name})",
                        showlegend=is_active,
                    )
                )

            # Unconditioned Trajectory (s = 0)
            fig_cfg.add_trace(
                go.Scatter(
                    x=uncond_traj[:, 0],
                    y=uncond_traj[:, 1],
                    mode="lines+markers",
                    line=dict(color="#94A3B8", width=2, dash="dot"),
                    marker=dict(size=4),
                    name="Unconditioned Path (s = 0)",
                )
            )

            # Standard Conditioned Trajectory (s = 1)
            fig_cfg.add_trace(
                go.Scatter(
                    x=cond_traj[:, 0],
                    y=cond_traj[:, 1],
                    mode="lines+markers",
                    line=dict(color="#F59E0B", width=2, dash="dash"),
                    marker=dict(size=4),
                    name="Standard Conditioned (s = 1)",
                )
            )

            # Active Guided CFG Trajectory
            fig_cfg.add_trace(
                go.Scatter(
                    x=cfg_traj[:, 0],
                    y=cfg_traj[:, 1],
                    mode="lines+markers",
                    line=dict(color=COLOR_CLASS_POS, width=3.5),
                    marker=dict(size=6, color=COLOR_CLASS_POS),
                    name=f"CFG Guided Path (s = {cfg_scale:.1f})",
                )
            )

            # Starting noise point
            fig_cfg.add_trace(
                go.Scatter(
                    x=[x_init[0]],
                    y=[x_init[1]],
                    mode="markers+text",
                    marker=dict(size=12, color=COLOR_CLASS_NEG, symbol="x"),
                    text=["Initial Noise x_T"],
                    textposition="bottom left",
                    textfont=dict(color=COLOR_CLASS_NEG, size=10),
                    name="Initial Latent Noise",
                )
            )

            fig_cfg.update_layout(
                height=440,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(title="Latent Dimension z₁", range=[-3.5, 3.5], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                yaxis=dict(title="Latent Dimension z₂", range=[-3.5, 3.5], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                margin=dict(l=40, r=20, t=20, b=45),
                legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(color="#CBD5E1", size=9)),
            )

            st.plotly_chart(fig_cfg, width="stretch", key="mmg_fig_cfg")

        with tab_pipeline_diag:
            st.markdown("##### 🏗️ Latent Diffusion Architecture (Stable Diffusion Framework)")

            st.markdown(
                """
                ```mermaid
                flowchart LR
                    subgraph Text_Conditioning ["1. Text Conditioning"]
                        P["Text Prompt: 'paradise cosmic beach'"] --> TE["CLIP Text Encoder"]
                        TE --> C["Context Embeddings c"]
                    end

                    subgraph Latent_Diffusion ["2. Latent Diffusion Engine (64x Compute Savings)"]
                        NZ["Random Latent Noise z_T ∈ ℝ^{64×64×4}"] --> UNet["Time-Conditioned U-Net"]
                        C -->|Cross-Attention| UNet
                        UNet -->|Denoising Steps| Z0["Clean Latent z₀ ∈ ℝ^{64×64×4}"]
                    end

                    subgraph Pixel_Decoder ["3. Pixel Reconstruction"]
                        Z0 --> VAE_Dec["VAE Decoder"]
                        VAE_Dec --> Img["High-Res Image X ∈ ℝ^{512×512×3}"]
                    end
                ```
                """
            )

            st.info(
                "💡 **Why Latent Diffusion (Rombach et al., 2022)?**: High-resolution pixels ($512 \\times 512 \\times 3 = 786,432$ values) "
                "contain immense imperceptible high-frequency redundancy. By training a powerful Variational Autoencoder (VAE), "
                "the diffusion model operates strictly in a low-dimensional perceptual latent space ($64 \\times 64 \\times 4 = 16,384$ values), "
                "yielding an **exact 48× to 64× reduction in computational complexity** while preserving visual fidelity."
            )

        with tab_trilemma:
            st.markdown("##### ⚖️ The Generative Learning Trilemma & Paradigm Comparison")

            categories = ["Sampling Speed", "Sample Quality", "Mode Diversity", "Training Stability", "Scalability"]

            fig_radar = go.Figure()

            # GAN
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[5, 4.5, 2.5, 2.0, 3.5],
                    theta=categories,
                    fill="toself",
                    name="GAN (Goodfellow 2014)",
                    line=dict(color="#F43F5E"),
                    opacity=0.6,
                )
            )

            # VAE
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[4.5, 2.5, 4.0, 4.5, 3.0],
                    theta=categories,
                    fill="toself",
                    name="VAE (Kingma & Welling 2013)",
                    line=dict(color="#F59E0B"),
                    opacity=0.6,
                )
            )

            # DDPM
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[2.0, 5.0, 5.0, 5.0, 4.5],
                    theta=categories,
                    fill="toself",
                    name="DDPM / Diffusion (Ho 2020)",
                    line=dict(color=COLOR_CLASS_POS),
                    opacity=0.6,
                )
            )

            # DiT
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[3.0, 5.0, 5.0, 5.0, 5.0],
                    theta=categories,
                    fill="toself",
                    name="DiT (Peebles & Xie 2023)",
                    line=dict(color=COLOR_HYPERPLANE),
                    opacity=0.6,
                )
            )

            fig_radar.update_layout(
                height=430,
                paper_bgcolor=BG_PAPER,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(color="#94A3B8", size=8)),
                    bgcolor=BG_PLOT,
                ),
                margin=dict(l=40, r=40, t=30, b=45),
                legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(color="#CBD5E1", size=9)),
            )

            st.plotly_chart(fig_radar, width="stretch", key="mmg_fig_radar")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Conditioning & Classifier-Free Guidance", expanded=False):
        st.markdown(
            r"""
            **Text Conditioning via Cross-Attention**:
            - To steer image generation using descriptive text prompts $y$, the prompt is first mapped into sequence embeddings $\mathbf{c} = \tau_\theta(y) \in \mathbb{R}^{M \times d_c}$ using a pretrained text encoder (e.g. CLIP).
            - The U-Net backbone integrates conditioning information using **Cross-Attention**:
              $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
              $$\text{where } Q = W^Q \phi_i(z_t), \quad K = W^K \mathbf{c}, \quad V = W^V \mathbf{c}$$
            - The image feature representations $\phi_i(z_t)$ query the text context tokens $\mathbf{c}$, grounding the spatial feature maps in the linguistic concepts.

            **Classifier-Free Guidance (CFG, Ho & Salimans 2022)**:
            - During training, the conditioning label $\mathbf{c}$ is randomly dropped with probability $p_{\text{uncond}} \approx 10\%$, replaced by a null token $\emptyset$.
            - A single neural network learns both the unconditional score $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)$ and conditional score $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})$.
            - During reverse sampling, the modified score is extrapolated using guidance scale $s \ge 1$:
              $$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + s \cdot \left(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)\right)$$
            - **Effect of Guidance Scale $s$**:
              - $s = 0$: Unconditional generation (ignores text prompt).
              - $s = 1$: Standard conditional sampling (matches conditional likelihood).
              - $s > 1$: Amplifies probability modes that have high conditional probability relative to unconditional density, trading sample diversity for high semantic fidelity and sharpness.
              - $s \gg 10$: Can cause over-saturation and contrast distortion.
            """
        )
