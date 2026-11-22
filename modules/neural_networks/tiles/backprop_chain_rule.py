"""
Tile 3: Backpropagation & Computational Chain Rule
Focus: Slides 316, 321 - Reverse-mode chain rule propagation through neural networks.
Interactive step-by-step scrubber from forward activations to backward error signals delta
and weight gradients dE/dθ.
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
            st.markdown("#### 🎛️ Backprop Simulation Knobs")

            step_scrubber = st.slider(
                "Backprop Flow Step",
                min_value=1,
                max_value=5,
                value=4,
                step=1,
                key="bp_step",
                help="1: Input → 2: Hidden Forward → 3: Output & Loss → 4: Backprop Errors (δ) → 5: Weight Gradients & Update",
            )

            step_names = {
                1: "1. Input Presentation (x₁, x₂)",
                2: "2. Forward Pass: Hidden Activations Z_m",
                3: "3. Forward Pass: Output ŷ & Squared Loss E",
                4: "4. Backward Pass: Error Signals δ_k & δ_m",
                5: "5. Gradient Calculation & Weight Step",
            }
            st.info(f"**Current**: {step_names[step_scrubber]}")

            learning_rate_eta = st.slider(
                "Learning Rate η",
                min_value=0.05,
                max_value=1.0,
                value=0.25,
                step=0.05,
                key="bp_eta",
            )

            with st.expander("⚙️ Example Probe & Weights", expanded=False):
                x1_val = st.slider("Probe Feature x₁", -2.0, 2.0, 0.8, 0.1, key="bp_x1")
                x2_val = st.slider("Probe Feature x₂", -2.0, 2.0, -0.6, 0.1, key="bp_x2")
                target_y = st.radio("Target Class / Label y", [1.0, 0.0], index=0, key="bp_y")

    # Set up fixed 2-in, 3-hidden, 1-out network weights for deterministic teaching
    x_vec = np.array([x1_val, x2_val])

    # Weights: Layer 1 alpha: shape (2, 3), bias alpha_0: shape (3,)
    W_alpha = np.array([
        [0.75, -0.90, 0.40],
        [-0.50, 0.60, 0.85],
    ])
    b_alpha = np.array([0.10, -0.20, 0.05])

    # Weights: Layer 2 beta: shape (3, 1), bias beta_0: shape (1,)
    W_beta = np.array([[0.80], [-1.10], [0.65]])
    b_beta = np.array([0.15])

    # 1. Forward Pass
    # Pre-activations a_m = alpha_0m + alpha_m^T x
    a_hidden = b_alpha + x_vec @ W_alpha
    # Sigmoidal activations Z_m = sigma(a_m)
    Z_hidden = 1.0 / (1.0 + np.exp(-a_hidden))

    # Output pre-activation
    a_out = b_beta[0] + np.dot(Z_hidden, W_beta[:, 0])
    # Output prediction y_hat (linear or sigmoid; let's use sigmoid for [0, 1])
    y_hat = 1.0 / (1.0 + np.exp(-a_out))

    # Loss E_i = (y - y_hat)^2
    loss_val = (target_y - y_hat) ** 2

    # 2. Backward Pass
    # Output error: delta_out = dE/da_out = 2(y_hat - y) * sigma'(a_out)
    dsig_out = y_hat * (1.0 - y_hat)
    delta_out = 2.0 * (y_hat - target_y) * dsig_out

    # Hidden error: delta_m = (delta_out * beta_m) * sigma'(a_m)
    dsig_hidden = Z_hidden * (1.0 - Z_hidden)
    delta_hidden = (delta_out * W_beta[:, 0]) * dsig_hidden

    # Gradients
    # dE/dW_beta = delta_out * Z_hidden
    grad_W_beta = delta_out * Z_hidden
    grad_b_beta = delta_out

    # dE/dW_alpha = x (outer) delta_hidden
    grad_W_alpha = np.outer(x_vec, delta_hidden)
    grad_b_alpha = delta_hidden

    # Weight updates
    W_alpha_new = W_alpha - learning_rate_eta * grad_W_alpha
    W_beta_new = W_beta - learning_rate_eta * grad_W_beta.reshape(-1, 1)

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted ŷ", f"{y_hat:.3f}", delta=f"{y_hat - target_y:+.3f} vs target")
        m2.metric("Squared Loss E", f"{loss_val:.4f}")
        m3.metric("Output Error δ", f"{delta_out:.4f}")
        m4.metric("Max |Gradient|", f"{max(np.max(np.abs(grad_W_alpha)), np.max(np.abs(grad_W_beta))):.4f}")

        tab_graph, tab_heatmap, tab_math = st.tabs([
            "🔄 Computational Graph & Signal Flow (Slide 321)",
            "📊 Layer-wise Gradient Matrix (Slide 316)",
            "📝 Forward & Backward Equations",
        ])

        with tab_graph:
            st.markdown(f"##### 🔄 Step {step_scrubber}: {step_names[step_scrubber]}")

            fig_bp = go.Figure()

            # Coordinates
            # Input: x = 0, y = [-1, 1]
            # Hidden: x = 2.5, y = [-1.5, 0, 1.5]
            # Output: x = 5.0, y = [0]
            x_in_coords = [0, 0]
            y_in_coords = [0.8, -0.8]
            x_hid_coords = [2.5, 2.5, 2.5]
            y_hid_coords = [1.2, 0.0, -1.2]
            x_out_coords = [5.0]
            y_out_coords = [0.0]

            # Determine colors based on active step
            forward_color = "#38BDF8" if step_scrubber in [1, 2, 3] else "#334155"
            backward_color = "#EF4444" if step_scrubber in [4, 5] else "#334155"

            # Connections: Input to Hidden
            for i in range(2):
                for j in range(3):
                    line_c = forward_color if step_scrubber >= 2 else "#1E293B"
                    w_txt = f"α_{i+1},{j+1} = {W_alpha[i, j]:.2f}"
                    if step_scrubber == 5:
                        w_txt += f"<br>∇ = {grad_W_alpha[i, j]:.3f}"
                    fig_bp.add_trace(
                        go.Scatter(
                            x=[x_in_coords[i], x_hid_coords[j]],
                            y=[y_in_coords[i], y_hid_coords[j]],
                            mode="lines",
                            line=dict(color=line_c, width=2.0),
                            hoverinfo="text",
                            text=w_txt,
                            showlegend=False,
                        )
                    )

            # Connections: Hidden to Output
            for j in range(3):
                line_c = backward_color if step_scrubber >= 4 else (forward_color if step_scrubber == 3 else "#1E293B")
                w_txt = f"β_{j+1} = {W_beta[j, 0]:.2f}"
                if step_scrubber == 5:
                    w_txt += f"<br>∇ = {grad_W_beta[j]:.3f}"
                fig_bp.add_trace(
                    go.Scatter(
                        x=[x_hid_coords[j], x_out_coords[0]],
                        y=[y_hid_coords[j], y_out_coords[0]],
                        mode="lines",
                        line=dict(color=line_c, width=2.5),
                        hoverinfo="text",
                        text=w_txt,
                        showlegend=False,
                    )
                )

            # Draw Nodes with active step annotations
            # Inputs
            fig_bp.add_trace(
                go.Scatter(
                    x=x_in_coords,
                    y=y_in_coords,
                    mode="markers+text",
                    marker=dict(color="#38BDF8", size=26, line=dict(color="#FFFFFF", width=1.5)),
                    text=[f"x₁={x1_val:.2f}", f"x₂={x2_val:.2f}"],
                    textposition="middle left",
                    name="Inputs",
                    hoverinfo="text",
                )
            )

            # Hidden Nodes
            hid_texts = [f"Z_{j+1}={Z_hidden[j]:.2f}" for j in range(3)]
            if step_scrubber >= 4:
                hid_texts = [f"Z_{j+1}={Z_hidden[j]:.2f}<br>δ={delta_hidden[j]:.3f}" for j in range(3)]

            fig_bp.add_trace(
                go.Scatter(
                    x=x_hid_coords,
                    y=y_hid_coords,
                    mode="markers+text",
                    marker=dict(color="#10B981", size=24, line=dict(color="#FFFFFF", width=1.5)),
                    text=hid_texts,
                    textposition="top center",
                    name="Hidden Units",
                    hoverinfo="text",
                )
            )

            # Output Node
            out_text = f"ŷ={y_hat:.2f}"
            if step_scrubber >= 3:
                out_text += f"<br>Loss={loss_val:.3f}"
            if step_scrubber >= 4:
                out_text += f"<br>δ_out={delta_out:.3f}"

            fig_bp.add_trace(
                go.Scatter(
                    x=x_out_coords,
                    y=y_out_coords,
                    mode="markers+text",
                    marker=dict(color="#F59E0B", size=28, line=dict(color="#FFFFFF", width=1.5)),
                    text=[out_text],
                    textposition="middle right",
                    name="Output",
                    hoverinfo="text",
                )
            )

            fig_bp.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(visible=False, range=[-1.2, 6.2]),
                yaxis=dict(visible=False, range=[-1.8, 1.8]),
                margin=dict(l=40, r=20, t=15, b=65),
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

            st.plotly_chart(fig_bp, width="stretch", key="bp_fig_bp")

        with tab_heatmap:
            st.markdown("##### 📊 Parameter Gradient Matrices: ∂E/∂α (Input-to-Hidden) and ∂E/∂β (Hidden-to-Output)")

            fig_hm = go.Figure()

            # Combined visual of gradients
            all_param_names = [
                "α₁₁", "α₁₂", "α₁₃",
                "α₂₁", "α₂₂", "α₂₃",
                "b_α₁", "b_α₂", "b_α₃",
                "β₁", "β₂", "β₃", "b_β",
            ]
            all_grads = [
                grad_W_alpha[0, 0], grad_W_alpha[0, 1], grad_W_alpha[0, 2],
                grad_W_alpha[1, 0], grad_W_alpha[1, 1], grad_W_alpha[1, 2],
                grad_b_alpha[0], grad_b_alpha[1], grad_b_alpha[2],
                grad_W_beta[0], grad_W_beta[1], grad_W_beta[2], grad_b_beta,
            ]
            grad_colors = ["#EF4444" if g < 0 else "#10B981" for g in all_grads]

            fig_hm.add_trace(
                go.Bar(
                    x=all_param_names,
                    y=all_grads,
                    marker=dict(color=grad_colors),
                    text=[f"{g:+.3f}" for g in all_grads],
                    textposition="outside",
                    hovertemplate="%{x}: Gradient = %{y:.4f}<extra></extra>",
                )
            )

            fig_hm.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    gridcolor="#1E293B",
                    tickfont=dict(color="#CBD5E1", size=10),
                ),
                yaxis=dict(
                    title=dict(text="Gradient ∂E/∂θ", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                margin=dict(l=40, r=20, t=15, b=45),
            )

            st.plotly_chart(fig_hm, width="stretch", key="bp_fig_hm")

        with tab_math:
            st.markdown(
                r"""
                ##### 📝 Forward & Backward Chain-Rule Mathematics (Slides 316, 321)
                
                **1. Forward Activations**:
                - Pre-activations: $a_m = \alpha_{0m} + \sum_{\ell=1}^p \alpha_{\ell m} x_\ell$
                - Post-activations: $Z_m = \sigma(a_m) = \frac{1}{1 + e^{-a_m}}$
                - Output pre-activation: $a_{\text{out}} = \beta_0 + \sum_{m=1}^M \beta_m Z_m$
                - Final prediction: $\hat{y} = \sigma(a_{\text{out}})$
                
                **2. Backward Error Signals ($\delta$)**:
                - Output error:
                  $$\delta_{\text{out}} = \frac{\partial \mathcal{E}}{\partial a_{\text{out}}} = 2(\hat{y} - y) \sigma'(a_{\text{out}})$$
                - Hidden error (chain rule backward):
                  $$\delta_m = \frac{\partial \mathcal{E}}{\partial a_m} = (\delta_{\text{out}} \cdot \beta_m) \sigma'(a_m)$$
                
                **3. Parameter Gradients & Updates (Slide 316)**:
                - Hidden-to-Output weights:
                  $$\frac{\partial \mathcal{E}}{\partial \beta_m} = \delta_{\text{out}} \cdot Z_m \implies \beta_m^{\text{new}} = \beta_m - \eta \frac{\partial \mathcal{E}}{\partial \beta_m}$$
                - Input-to-Hidden weights:
                  $$\frac{\partial \mathcal{E}}{\partial \alpha_{\ell m}} = \delta_m \cdot x_\ell \implies \alpha_{\ell m}^{\text{new}} = \alpha_{\ell m} - \eta \frac{\partial \mathcal{E}}{\partial \alpha_{\ell m}}$$
                """
            )

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The Backpropagation Algorithm (Slide 321)", expanded=False):
        st.markdown(
            r"""
            **Why Backpropagation is Efficient (Slide 321)**:
            - When computing the gradient of the loss function $\mathcal{E}(\theta)$ with respect to all network weights, an optimization algorithm must apply the chain rule of calculus.
            - If computed naively for each parameter independently, redundant intermediate derivatives are repeatedly recalculated, requiring $O(W^2)$ computations.
            - **Backpropagation** accomplishes this in $O(W)$ time by caching forward activations and then propagating the error derivatives $\delta$ **backward, one layer at a time**, from the output layer to the input layer.
            - Each weight's gradient is simply the product of the incoming forward activation and the backward error signal!
            """
        )
