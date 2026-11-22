"""
Tile 4: Optimization Arena (GD, SGD, Momentum, RMSProp, Adam)
Focus: Slides 316, 318-320 - Dynamic learning rate algorithms and first-order optimizers.
Compares standard GD, Momentum (Slide 319), RMSProp (Slide 319), and Adam (Slide 320)
navigating ill-conditioned ravines, steep valleys, and strict saddle points.
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
            st.markdown("#### 🎛️ Optimizer Arena Knobs")

            landscape_choice = st.selectbox(
                "Loss Surface Topography",
                [
                    "Elongated Ravine (Beale Valley)",
                    "Strict Saddle Point (Slide 323)",
                    "Multimodal Local Minima (Rastrigin)",
                ],
                index=0,
                key="opt_land",
                help="Test optimizers on ill-conditioned ravines with high condition numbers, strict saddle points, or multiple local basins.",
            )

            learning_rate = st.slider(
                "Learning Rate η",
                min_value=0.001,
                max_value=0.30,
                value=0.03,
                step=0.005,
                format="%.3f",
                key="opt_lr",
                help="Base step size η shared across all algorithms.",
            )

            momentum_alpha = st.slider(
                "Momentum Decay Factor α",
                min_value=0.50,
                max_value=0.98,
                value=0.90,
                step=0.02,
                key="opt_alpha",
                help="Momentum coefficient in Slide 319: Δθ = αΔθ - η∇E.",
            )

            n_steps = st.slider(
                "Optimization Steps T",
                min_value=20,
                max_value=200,
                value=80,
                step=10,
                key="opt_steps",
            )

            with st.expander("⚙️ Initial Position (w₁, w₂)", expanded=False):
                if "Ravine" in landscape_choice:
                    init_w1 = st.slider("Initial w₁", -2.5, 2.5, -1.8, 0.1, key="opt_w1_r")
                    init_w2 = st.slider("Initial w₂", -2.5, 2.5, 1.8, 0.1, key="opt_w2_r")
                elif "Saddle" in landscape_choice:
                    init_w1 = st.slider("Initial w₁", -2.0, 2.0, 0.05, 0.05, key="opt_w1_s")
                    init_w2 = st.slider("Initial w₂", -2.0, 2.0, 1.5, 0.1, key="opt_w2_s")
                else:
                    init_w1 = st.slider("Initial w₁", -2.5, 2.5, 2.2, 0.1, key="opt_w1_m")
                    init_w2 = st.slider("Initial w₂", -2.5, 2.5, 1.8, 0.1, key="opt_w2_m")

    # Define loss functions and analytical gradients
    if "Ravine" in landscape_choice:
        # Ill-conditioned quadratic ravine: f(w1, w2) = 0.5 * w1^2 + 15.0 * w2^2
        def loss_fn(w1, w2):
            return 0.5 * (w1 ** 2) + 12.0 * (w2 ** 2)

        def grad_fn(w1, w2):
            return np.array([w1, 24.0 * w2])

        x_range = (-2.5, 2.5)
        y_range = (-2.5, 2.5)
    elif "Saddle" in landscape_choice:
        # Strict saddle point: f(w1, w2) = w1^2 - w2^2 + 0.1 * w2^4
        def loss_fn(w1, w2):
            return (w1 ** 2) - 1.5 * (w2 ** 2) + 0.25 * (w2 ** 4)

        def grad_fn(w1, w2):
            return np.array([2.0 * w1, -3.0 * w2 + w2 ** 3])

        x_range = (-2.2, 2.2)
        y_range = (-2.2, 2.2)
    else:
        # Rastrigin multimodal: f(w1, w2) = 10*2 + (w1^2 - 10cos(2pi w1)) + (w2^2 - 10cos(2pi w2))
        def loss_fn(w1, w2):
            return 4.0 + (w1 ** 2 - 3.0 * np.cos(2.0 * np.pi * w1)) + (w2 ** 2 - 3.0 * np.cos(2.0 * np.pi * w2))

        def grad_fn(w1, w2):
            return np.array([
                2.0 * w1 + 6.0 * np.pi * np.sin(2.0 * np.pi * w1),
                2.0 * w2 + 6.0 * np.pi * np.sin(2.0 * np.pi * w2),
            ])

        x_range = (-2.5, 2.5)
        y_range = (-2.5, 2.5)

    w_start = np.array([init_w1, init_w2], dtype=float)

    # 1. Standard GD
    traj_gd = [w_start.copy()]
    loss_gd = [loss_fn(*w_start)]
    w_curr = w_start.copy()
    for _ in range(n_steps):
        g = grad_fn(*w_curr)
        w_curr = w_curr - learning_rate * g
        traj_gd.append(w_curr.copy())
        loss_gd.append(loss_fn(*w_curr))

    # 2. Momentum (Slide 319)
    # Δθ^{(t+1)} = α Δθ^{(t)} - η ∇E
    traj_mom = [w_start.copy()]
    loss_mom = [loss_fn(*w_start)]
    w_curr = w_start.copy()
    delta_w = np.zeros(2)
    for _ in range(n_steps):
        g = grad_fn(*w_curr)
        delta_w = momentum_alpha * delta_w - learning_rate * g
        w_curr = w_curr + delta_w
        traj_mom.append(w_curr.copy())
        loss_mom.append(loss_fn(*w_curr))

    # 3. RMSProp (Slide 319)
    # v^{(t+1)} = γ v^{(t)} + (1-γ)(∇E)^2
    # θ^{(t+1)} = θ^{(t)} - η / sqrt(v + 1e-8) * ∇E
    gamma_rms = 0.90
    traj_rms = [w_start.copy()]
    loss_rms = [loss_fn(*w_start)]
    w_curr = w_start.copy()
    v_rms = np.zeros(2)
    for _ in range(n_steps):
        g = grad_fn(*w_curr)
        v_rms = gamma_rms * v_rms + (1.0 - gamma_rms) * (g ** 2)
        w_curr = w_curr - (learning_rate / (np.sqrt(v_rms) + 1e-8)) * g
        traj_rms.append(w_curr.copy())
        loss_rms.append(loss_fn(*w_curr))

    # 4. Adam (Slide 320)
    # m^{(t+1)} = γ1 m + (1-γ1)g
    # v^{(t+1)} = γ2 v + (1-γ2)g^2
    # m_hat = m / (1 - γ1^t), v_hat = v / (1 - γ2^t)
    gamma1, gamma2 = 0.90, 0.999
    traj_adam = [w_start.copy()]
    loss_adam = [loss_fn(*w_start)]
    w_curr = w_start.copy()
    m_adam = np.zeros(2)
    v_adam = np.zeros(2)
    for t in range(1, n_steps + 1):
        g = grad_fn(*w_curr)
        m_adam = gamma1 * m_adam + (1.0 - gamma1) * g
        v_adam = gamma2 * v_adam + (1.0 - gamma2) * (g ** 2)
        m_hat = m_adam / (1.0 - (gamma1 ** t))
        v_hat = v_adam / (1.0 - (gamma2 ** t))
        w_curr = w_curr - (learning_rate / (np.sqrt(v_hat) + 1e-8)) * m_hat
        traj_adam.append(w_curr.copy())
        loss_adam.append(loss_fn(*w_curr))

    traj_gd = np.array(traj_gd)
    traj_mom = np.array(traj_mom)
    traj_rms = np.array(traj_rms)
    traj_adam = np.array(traj_adam)

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Standard GD Loss", f"{loss_gd[-1]:.3f}")
        m2.metric("Momentum Loss", f"{loss_mom[-1]:.3f}")
        m3.metric("RMSProp Loss", f"{loss_rms[-1]:.3f}")
        m4.metric("Adam Loss", f"{loss_adam[-1]:.3f}", delta=f"{loss_adam[-1] - loss_gd[-1]:.3f} vs GD", delta_color="inverse")

        tab_contour, tab_loss, tab_3d = st.tabs([
            "🏎️ 2D Trajectory Arena (Slides 319 & 320)",
            "📉 Convergence Loss Curves",
            "🏔️ 3D Surface Topology Flythrough",
        ])

        with tab_contour:
            st.markdown(f"##### 🏎️ Simultaneous Optimizer Paths on {landscape_choice}")

            # Dense contour grid
            gx = np.linspace(x_range[0], x_range[1], 120)
            gy = np.linspace(y_range[0], y_range[1], 120)
            gxx, gyy = np.meshgrid(gx, gy)
            gz = loss_fn(gxx, gyy)

            fig_cont = go.Figure()

            # Background contour
            fig_cont.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=gz,
                    colorscale="Blues",
                    reversescale=True,
                    opacity=0.55,
                    showscale=False,
                    contours=dict(start=np.min(gz), end=np.percentile(gz, 85), size=(np.percentile(gz, 85) - np.min(gz)) / 15),
                    hoverinfo="skip",
                )
            )

            # Start point
            fig_cont.add_trace(
                go.Scatter(
                    x=[w_start[0]],
                    y=[w_start[1]],
                    mode="markers",
                    marker=dict(color="#FFFFFF", size=11, symbol="star"),
                    name="Start θ₀",
                    hoverinfo="text",
                    text=f"Start: ({w_start[0]:.2f}, {w_start[1]:.2f})",
                )
            )

            # 1. Standard GD (Red)
            fig_cont.add_trace(
                go.Scatter(
                    x=traj_gd[:, 0],
                    y=traj_gd[:, 1],
                    mode="lines+markers",
                    name="Standard GD",
                    line=dict(color="#EF4444", width=2.2),
                    marker=dict(size=4, color="#EF4444"),
                    hovertemplate="GD Step %{pointNumber}: (%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )

            # 2. Momentum (Amber)
            fig_cont.add_trace(
                go.Scatter(
                    x=traj_mom[:, 0],
                    y=traj_mom[:, 1],
                    mode="lines+markers",
                    name=f"Momentum (α={momentum_alpha})",
                    line=dict(color="#F59E0B", width=2.5),
                    marker=dict(size=4, color="#F59E0B"),
                    hovertemplate="Mom Step %{pointNumber}: (%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )

            # 3. RMSProp (Sky Blue)
            fig_cont.add_trace(
                go.Scatter(
                    x=traj_rms[:, 0],
                    y=traj_rms[:, 1],
                    mode="lines+markers",
                    name="RMSProp (γ=0.9)",
                    line=dict(color="#38BDF8", width=2.5),
                    marker=dict(size=4, color="#38BDF8"),
                    hovertemplate="RMS Step %{pointNumber}: (%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )

            # 4. Adam (Emerald)
            fig_cont.add_trace(
                go.Scatter(
                    x=traj_adam[:, 0],
                    y=traj_adam[:, 1],
                    mode="lines+markers",
                    name="Adam (γ₁=0.9, γ₂=0.999)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    marker=dict(size=5, color=COLOR_HYPERPLANE),
                    hovertemplate="Adam Step %{pointNumber}: (%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )

            fig_cont.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Parameter w₁", font=dict(color="#CBD5E1", size=11)),
                    range=list(x_range),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Parameter w₂", font=dict(color="#CBD5E1", size=11)),
                    range=list(y_range),
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

            st.plotly_chart(fig_cont, width="stretch", key="opt_fig_cont")

        with tab_loss:
            st.markdown("##### 📉 Stepwise Loss Convergence E(θ_t)")

            fig_loss = go.Figure()
            steps_axis = list(range(n_steps + 1))

            fig_loss.add_trace(
                go.Scatter(
                    x=steps_axis,
                    y=loss_gd,
                    mode="lines",
                    name="Standard GD",
                    line=dict(color="#EF4444", width=2.0),
                    hovertemplate="Step %{x}: Loss = %{y:.4f}<extra></extra>",
                )
            )

            fig_loss.add_trace(
                go.Scatter(
                    x=steps_axis,
                    y=loss_mom,
                    mode="lines",
                    name="Momentum",
                    line=dict(color="#F59E0B", width=2.5),
                    hovertemplate="Step %{x}: Loss = %{y:.4f}<extra></extra>",
                )
            )

            fig_loss.add_trace(
                go.Scatter(
                    x=steps_axis,
                    y=loss_rms,
                    mode="lines",
                    name="RMSProp",
                    line=dict(color="#38BDF8", width=2.5),
                    hovertemplate="Step %{x}: Loss = %{y:.4f}<extra></extra>",
                )
            )

            fig_loss.add_trace(
                go.Scatter(
                    x=steps_axis,
                    y=loss_adam,
                    mode="lines",
                    name="Adam",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    hovertemplate="Step %{x}: Loss = %{y:.4f}<extra></extra>",
                )
            )

            fig_loss.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Optimization Step t", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Loss E(θ)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_loss, width="stretch", key="opt_fig_loss")

        with tab_3d:
            st.markdown("##### 🏔️ 3D Loss Surface with Adam vs Standard GD Trajectories")

            fig_3d = go.Figure()

            # Surface
            fig_3d.add_trace(
                go.Surface(
                    x=gx,
                    y=gy,
                    z=gz,
                    colorscale="Viridis",
                    opacity=0.75,
                    showscale=False,
                    hoverinfo="skip",
                )
            )

            # Adam Path
            fig_3d.add_trace(
                go.Scatter3d(
                    x=traj_adam[:, 0],
                    y=traj_adam[:, 1],
                    z=loss_adam,
                    mode="lines+markers",
                    name="Adam",
                    line=dict(color="#FFFFFF", width=5),
                    marker=dict(size=3, color=COLOR_HYPERPLANE),
                )
            )

            # GD Path
            fig_3d.add_trace(
                go.Scatter3d(
                    x=traj_gd[:, 0],
                    y=traj_gd[:, 1],
                    z=loss_gd,
                    mode="lines+markers",
                    name="Standard GD",
                    line=dict(color="#EF4444", width=4),
                    marker=dict(size=3, color="#EF4444"),
                )
            )

            fig_3d.update_layout(
                height=440,
                paper_bgcolor=BG_PAPER,
                margin=dict(l=20, r=20, t=15, b=45),
                scene=dict(
                    xaxis=dict(title="w₁", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    yaxis=dict(title="w₂", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                    zaxis=dict(title="Loss", gridcolor="#1E293B", backgroundcolor=BG_PLOT),
                ),
            )

            st.plotly_chart(fig_3d, width="stretch", key="opt_fig_3d")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Advanced First-Order Optimizers (Slides 319–320)", expanded=False):
        st.markdown(
            r"""
            **1. Momentum Method (Slide 319)**:
            - Overcomes high curvature oscillations in ill-conditioned ravines by accumulating a velocity vector:
              $$\Delta \theta^{(t+1)} = \alpha \Delta \theta^{(t)} - \eta \frac{\partial \mathcal{E}(\theta)}{\partial \theta}$$
              $$\theta^{(t+1)} = \theta^{(t)} + \Delta \theta^{(t+1)}$$
              where $\alpha \in [0, 1]$ is the exponential decay / friction factor (typically $0.9$).

            **2. RMSProp (Root Mean Square Propagation, Slide 319)**:
            - Dynamically adapts learning rates per parameter, scaling down steps along steep dimensions and scaling up along flat ones:
              $$v_\theta^{(t+1)} = \gamma v_\theta^{(t)} + (1 - \gamma) \left(\frac{\partial \mathcal{E}(\theta)}{\partial \theta}\right)^2$$
              $$\theta^{(t+1)} = \theta^{(t)} - \frac{\eta}{\sqrt{v_\theta^{(t+1)} + \epsilon}} \frac{\partial \mathcal{E}(\theta)}{\partial \theta}$$
              where $\gamma \in [0, 1]$ is the forgetting factor.

            **3. Adam (Adaptive Moment Estimation, Slide 320)**:
            - Combines the directional inertia of **Momentum** (1st moment $m_\theta$) with the adaptive per-coordinate scaling of **RMSProp** (2nd moment $v_\theta$), adding initialization bias corrections:
              1. $m_\theta^{(t+1)} \leftarrow \gamma_1 m_\theta^{(t)} + (1 - \gamma_1) \frac{\partial \mathcal{E}}{\partial \theta}$
              2. $v_\theta^{(t+1)} \leftarrow \gamma_2 v_\theta^{(t)} + (1 - \gamma_2) \left(\frac{\partial \mathcal{E}}{\partial \theta}\right)^2$
              3. Bias corrections: $\hat{m}_\theta = \frac{m_\theta^{(t+1)}}{1 - \gamma_1^{t+1}}, \quad \hat{v}_\theta = \frac{v_\theta^{(t+1)}}{1 - \gamma_2^{t+1}}$
              4. Step: $\theta^{(t+1)} \leftarrow \theta^{(t)} - \eta \frac{\hat{m}_\theta}{\sqrt{\hat{v}_\theta} + \epsilon}$
            """
        )
