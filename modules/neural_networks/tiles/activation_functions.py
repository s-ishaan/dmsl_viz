"""
Tile 2: Activation Functions & Vanishing Gradients
Focus: Slides 313-314 - Comparative analysis of Sigmoid σ(sv) across scaling factors s (Slide 313),
ReLU R(x) = max(0, x), Leaky ReLU (Slide 314), and an interactive simulator of the vanishing gradient
phenomenon across L deep layers.
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
            st.markdown("#### 🎛️ Activation Knobs")

            scale_s = st.slider(
                "Sigmoid Scale Parameter s",
                min_value=0.2,
                max_value=12.0,
                value=1.0,
                step=0.2,
                key="act_s",
                help="Scaling factor s in σ(sv) = 1 / (1 + e^-sv). Slide 313 highlights s = 0.5 (blue), s = 1.0 (red), s = 10.0 (purple).",
            )

            leaky_alpha = st.slider(
                "Leaky ReLU Slope α",
                min_value=0.01,
                max_value=0.25,
                value=0.01,
                step=0.01,
                key="act_alpha",
                help="Small positive slope when x <= 0: R(x) = αx. Prevents 'dead neuron' problem.",
            )

            depth_l = st.slider(
                "Network Layer Depth L",
                min_value=1,
                max_value=12,
                value=6,
                step=1,
                key="act_depth",
                help="Simulates backpropagation of gradient signal through L layers.",
            )

            v_probe = st.slider(
                "Input Coordinate Probe v",
                min_value=-6.0,
                max_value=6.0,
                value=1.5,
                step=0.25,
                key="act_probe",
                help="Probe a specific pre-activation value v.",
            )

    # Compute functions over domain [-6, 6]
    v_axis = np.linspace(-6.0, 6.0, 300)

    # Sigmoid variants matching Slide 313
    sig_half = 1.0 / (1.0 + np.exp(-0.5 * v_axis))
    sig_one = 1.0 / (1.0 + np.exp(-1.0 * v_axis))
    sig_ten = 1.0 / (1.0 + np.exp(-10.0 * v_axis))
    sig_user = 1.0 / (1.0 + np.exp(-scale_s * v_axis))

    # ReLU & Leaky ReLU matching Slide 314
    relu_curve = np.maximum(0.0, v_axis)
    leaky_curve = np.where(v_axis > 0, v_axis, leaky_alpha * v_axis)

    # Derivatives
    # d/dv [1 / (1 + e^-sv)] = s * σ(sv) * (1 - σ(sv))
    dsig_user = scale_s * sig_user * (1.0 - sig_user)
    dsig_one = 1.0 * sig_one * (1.0 - sig_one)
    drelu = np.where(v_axis > 0, 1.0, 0.0)
    dleaky = np.where(v_axis > 0, 1.0, leaky_alpha)

    # Values at probe
    probe_sig = 1.0 / (1.0 + np.exp(-scale_s * v_probe))
    probe_dsig = scale_s * probe_sig * (1.0 - probe_sig)
    probe_relu = max(0.0, v_probe)
    probe_drelu = 1.0 if v_probe > 0 else 0.0

    # Backpropagation signal decay simulation through L layers:
    # Gradients multiply by σ'(v) at each layer
    # For standard sigmoid, max derivative is 0.25 (at v=0). For probe, it is probe_dsig
    layers_axis = list(range(1, depth_l + 1))
    grad_signal_sig_max = [0.25 ** l for l in layers_axis]
    grad_signal_sig_probe = [probe_dsig ** l for l in layers_axis]
    grad_signal_relu = [1.0 ** l if v_probe > 0 else 0.0 for l in layers_axis]
    grad_signal_leaky = [1.0 ** l if v_probe > 0 else leaky_alpha ** l for l in layers_axis]

    final_sig_decay = grad_signal_sig_probe[-1]
    final_relu_flow = grad_signal_relu[-1]

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sigmoid σ(v)", f"{probe_sig:.3f}")
        m2.metric("Sigmoid Gradient σ'", f"{probe_dsig:.4f}", help="Max possible value is 0.25 * s")
        m3.metric("ReLU Gradient R'", f"{probe_drelu:.1f}")
        m4.metric(f"L={depth_l} Sigmoid Signal", f"{final_sig_decay:.2e}", help="Backpropagated gradient magnitude reaching layer 1")

        tab_activation, tab_derivative, tab_vanishing = st.tabs([
            "📈 Activation Curves (Slides 313 & 314)",
            "⚡ Derivatives & Saturation Zones",
            "📉 Vanishing Gradient Simulator across L Layers",
        ])

        with tab_activation:
            st.markdown("##### 📈 Activation Profiles: Sigmoid Scaling (Slide 313) vs ReLU & Leaky ReLU (Slide 314)")

            fig_act = go.Figure()

            # s = 0.5 (Blue)
            fig_act.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=sig_half,
                    mode="lines",
                    name="Sigmoid s = 0.5 (Blue)",
                    line=dict(color="#38BDF8", width=2.2, dash="dash"),
                    hovertemplate="v: %{x:.2f}<br>σ(0.5v): %{y:.3f}<extra></extra>",
                )
            )

            # s = 1.0 (Red)
            fig_act.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=sig_one,
                    mode="lines",
                    name="Sigmoid s = 1.0 (Red)",
                    line=dict(color="#EF4444", width=2.5),
                    hovertemplate="v: %{x:.2f}<br>σ(1v): %{y:.3f}<extra></extra>",
                )
            )

            # s = 10.0 (Purple)
            fig_act.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=sig_ten,
                    mode="lines",
                    name="Sigmoid s = 10.0 (Purple)",
                    line=dict(color="#A855F7", width=2.2, dash="dot"),
                    hovertemplate="v: %{x:.2f}<br>σ(10v): %{y:.3f}<extra></extra>",
                )
            )

            # ReLU (Emerald)
            fig_act.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=relu_curve,
                    mode="lines",
                    name="ReLU: max(0, v)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.0),
                    hovertemplate="v: %{x:.2f}<br>ReLU: %{y:.3f}<extra></extra>",
                )
            )

            # Leaky ReLU (Amber)
            fig_act.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=leaky_curve,
                    mode="lines",
                    name=f"Leaky ReLU (α = {leaky_alpha})",
                    line=dict(color="#F59E0B", width=2.0, dash="dashdot"),
                    hovertemplate="v: %{x:.2f}<br>Leaky: %{y:.3f}<extra></extra>",
                )
            )

            # Probe vertical marker
            fig_act.add_vline(x=v_probe, line_width=1.5, line_dash="dash", line_color="#FFFFFF")

            fig_act.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Input Signal v = α₀ + αᵀx", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Activation Output f(v)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    range=[-0.2, 3.5],
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

            st.plotly_chart(fig_act, width="stretch", key="act_fig_act")

        with tab_derivative:
            st.markdown("##### ⚡ Derivatives Profile: Maximum Gradient 0.25 for Sigmoid vs Constant 1.0 for ReLU")

            fig_der = go.Figure()

            # Sigmoid derivative (Red)
            fig_der.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=dsig_one,
                    mode="lines",
                    name="Sigmoid Derivative σ'(v) (Max 0.25)",
                    line=dict(color="#EF4444", width=3.0),
                    hovertemplate="v: %{x:.2f}<br>σ': %{y:.4f}<extra></extra>",
                )
            )

            # User Scaled Sigmoid derivative
            fig_der.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=dsig_user,
                    mode="lines",
                    name=f"Scaled Sigmoid s={scale_s} σ'(sv)",
                    line=dict(color="#A855F7", width=2.0, dash="dash"),
                    hovertemplate="v: %{x:.2f}<br>s·σ': %{y:.4f}<extra></extra>",
                )
            )

            # ReLU derivative (Emerald)
            fig_der.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=drelu,
                    mode="lines",
                    name="ReLU Derivative R'(v) (1 for v > 0)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.0),
                    hovertemplate="v: %{x:.2f}<br>R': %{y:.1f}<extra></extra>",
                )
            )

            # Leaky ReLU derivative (Amber)
            fig_der.add_trace(
                go.Scatter(
                    x=v_axis,
                    y=dleaky,
                    mode="lines",
                    name=f"Leaky ReLU Derivative (α = {leaky_alpha})",
                    line=dict(color="#F59E0B", width=2.0, dash="dot"),
                    hovertemplate="v: %{x:.2f}<br>Leaky': %{y:.3f}<extra></extra>",
                )
            )

            fig_der.add_vline(x=v_probe, line_width=1.5, line_dash="dash", line_color="#FFFFFF")

            fig_der.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Input Signal v", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Derivative df/dv", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    range=[-0.05, 1.25],
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

            st.plotly_chart(fig_der, width="stretch", key="act_fig_der")

        with tab_vanishing:
            st.markdown(f"##### 📉 Backpropagated Gradient Flow Across L = {depth_l} Hidden Layers")

            fig_flow = go.Figure()

            # Best-case Sigmoid (at v=0, derivative=0.25)
            fig_flow.add_trace(
                go.Scatter(
                    x=layers_axis,
                    y=grad_signal_sig_max,
                    mode="lines+markers",
                    name="Best-Case Sigmoid: (0.25)^L",
                    line=dict(color="#EF4444", width=2.8),
                    marker=dict(size=7, color="#EF4444"),
                    hovertemplate="Layer %{x}: Signal = %{y:.2e}<extra></extra>",
                )
            )

            # Probe Sigmoid
            fig_flow.add_trace(
                go.Scatter(
                    x=layers_axis,
                    y=grad_signal_sig_probe,
                    mode="lines+markers",
                    name=f"Current Sigmoid at v={v_probe}: ({probe_dsig:.2f})^L",
                    line=dict(color="#A855F7", width=2.2, dash="dash"),
                    marker=dict(size=6, color="#A855F7"),
                    hovertemplate="Layer %{x}: Signal = %{y:.2e}<extra></extra>",
                )
            )

            # ReLU flow
            fig_flow.add_trace(
                go.Scatter(
                    x=layers_axis,
                    y=grad_signal_relu,
                    mode="lines+markers",
                    name="ReLU Signal Flow (1.0)^L",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    marker=dict(size=8, color=COLOR_HYPERPLANE),
                    hovertemplate="Layer %{x}: Signal = %{y:.1f}<extra></extra>",
                )
            )

            fig_flow.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Backpropagation Depth (Layer Number L)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    tickmode="linear",
                    dtick=1,
                ),
                yaxis=dict(
                    title=dict(text="Gradient Multiplier Factor (Log Scale)", font=dict(color="#CBD5E1", size=11)),
                    type="log",
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

            st.plotly_chart(fig_flow, width="stretch", key="act_fig_flow")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The Vanishing Gradient Problem (Slides 313–314)", expanded=False):
        st.markdown(
            r"""
            **The Sigmoid Activation Function (Slide 313)**:
            $$\sigma(sv) = \frac{1}{1 + e^{-sv}}$$
            - Mimics a biological neuron firing: suppresses large negative signals toward 0 and saturates positive stimuli toward 1.
            - Controlled by scale parameter $s$:
              - $s = 0.5$: Gradual, broad transition.
              - $s = 1.0$: Standard logistic sigmoid.
              - $s = 10.0$: Sharp transition approaching the Heaviside step function $I(v \ge 0)$.

            **Why the Sigmoid Caused the Vanishing Gradient Problem (Slide 314)**:
            - The derivative of the standard sigmoid is:
              $$\sigma'(v) = \sigma(v)(1 - \sigma(v))$$
            - The maximum value of $\sigma'(v)$ occurs at $v = 0$, where $\sigma'(0) = 0.5 \times 0.5 = 0.25$. For $|v| > 3$, $\sigma'(v) \approx 0$.
            - In an $L$-layer network, the chain rule of backpropagation repeatedly multiplies activation derivatives:
              $$\frac{\partial \mathcal{E}}{\partial \theta_1} \propto \prod_{\ell=1}^L \sigma'(v_\ell) \le (0.25)^L$$
            - For $L = 6$, $(0.25)^6 \approx 0.00024$; for $L = 10$, $(0.25)^{10} \approx 9.5 \times 10^{-7}$.
            - Consequently, early layers receive virtually zero gradient updates, halting learning in deep architectures.

            **The Solution: ReLU & Leaky ReLU (Slide 314)**:
            - **Rectified Linear Unit (ReLU)**:
              $$R(x) = \max(0, x), \quad R'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$
              Because the derivative is strictly $1.0$ in the active regime, gradients flow backwards indefinitely through dozens of layers without decay!
            - **Leaky ReLU**:
              $$R(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases}$$
              Allows a small gradient $\alpha \approx 0.01$ when inactive, preventing "dead neuron" paralysis.
            """
        )
