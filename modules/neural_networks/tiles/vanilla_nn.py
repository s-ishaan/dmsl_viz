"""
Tile 1: The Single Hidden Layer (Vanilla) Neural Network
Focus: Slides 311-312, 315 - Mathematical architecture of the vanilla neural net:
Z_m(x) = σ(α_{0m} + α_m^T x), m = 1..M
ŷ_k(x) = β_{0k} + ∑_{m=1}^M β_{mk} Z_m(x)
Parameter accounting ((p+1)M + (M+1)K weights) and universal approximation synthesis.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier, MLPRegressor
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Network Architecture Knobs")

            m_hidden = st.slider(
                "Hidden Layer Units M",
                min_value=1,
                max_value=16,
                value=4,
                step=1,
                key="vnn_m",
                help="Number of hidden neurons Z_1 ... Z_M in the single hidden layer.",
            )

            activation_choice = st.selectbox(
                "Hidden Layer Activation σ(·)",
                ["logistic", "relu", "tanh"],
                index=0,
                key="vnn_act",
                help="Activation function. 'logistic' corresponds to the classic sigmoid σ(v) = 1/(1 + e^-v).",
            )

            max_epochs = st.slider(
                "Training Epochs / Iterations",
                min_value=50,
                max_value=800,
                value=300,
                step=50,
                key="vnn_epochs",
            )

            learning_rate_val = st.slider(
                "Learning Rate η",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                key="vnn_lr",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                dataset_type = st.selectbox(
                    "Topology",
                    ["Moons", "Concentric Circles", "XOR / Checkerboard"],
                    index=0,
                    key="vnn_data",
                )
                n_samples = st.slider("Sample Count N", 60, 240, 120, 10, key="vnn_n")
                noise_lvl = st.slider("Noise Level", 0.05, 0.35, 0.15, 0.02, key="vnn_noise")

    # Generate 2D dataset
    from utils.datasets import get_dataset
    X, y_svm = get_dataset(dataset_type, n_samples=n_samples, noise=noise_lvl, random_state=42)
    y = np.where(y_svm == 1, 1, 0)
    p_in = X.shape[1]  # 2 inputs
    k_out = 1          # Binary output

    # Parameter accounting (Slide 315)
    # {α_lm: l=0..p, m=1..M} => (p + 1) * M
    # {β_mk: m=0..M, k=1..K} => (M + 1) * K
    weights_in_hidden = (p_in + 1) * m_hidden
    weights_hidden_out = (m_hidden + 1) * k_out
    total_weights = weights_in_hidden + weights_hidden_out

    # Fit MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(m_hidden,),
        activation=activation_choice,
        max_iter=max_epochs,
        learning_rate_init=learning_rate_val,
        solver="adam",
        random_state=42,
    )
    mlp.fit(X, y)
    train_acc = float(np.mean(mlp.predict(X) == y)) * 100.0
    final_loss = float(mlp.loss_)

    # Grid for contour
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    gx = np.linspace(x_min, x_max, 110)
    gy = np.linspace(y_min, y_max, 110)
    gxx, gyy = np.meshgrid(gx, gy)
    grid_pts = np.c_[gxx.ravel(), gyy.ravel()]

    prob_grid = mlp.predict_proba(grid_pts)[:, 1].reshape(gxx.shape)

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hidden Units M", f"{m_hidden}")
        m2.metric("Total Weights θ", f"{total_weights}", help=f"(p+1)M = {weights_in_hidden} input + (M+1)K = {weights_hidden_out} output")
        m3.metric("Training Accuracy", f"{train_acc:.1f}%")
        m4.metric("Loss E(θ)", f"{final_loss:.4f}")

        tab_network, tab_boundary, tab_basis = st.tabs([
            "🧠 Network Architecture & Weight Matrix (Slide 315)",
            "🗺️ Decision Manifold & Output Probability",
            "📶 Hidden Basis Activations Z_m(x)",
        ])

        with tab_network:
            st.markdown("##### 🧠 Interactive Network Architecture & Parameter Accounting (Slide 311–312, 315)")

            # Draw topological network diagram using Plotly
            fig_net = go.Figure()

            # Coordinates for layers
            # Input layer: x = 0, y spaced
            y_in = np.linspace(-1.5, 1.5, p_in)
            # Hidden layer: x = 2, y spaced
            y_hid = np.linspace(-2.2, 2.2, m_hidden)
            # Output layer: x = 4, y = 0
            y_out = np.array([0.0])

            # Draw connection lines
            # Layer 1 weights: alpha
            W1 = mlp.coefs_[0]  # shape (p, M)
            for i in range(p_in):
                for j in range(m_hidden):
                    w_val = W1[i, j]
                    w_norm = min(4.0, max(0.8, abs(w_val) * 1.5))
                    line_color = "rgba(16, 185, 129, 0.45)" if w_val >= 0 else "rgba(244, 63, 94, 0.45)"
                    fig_net.add_trace(
                        go.Scatter(
                            x=[0, 2],
                            y=[y_in[i], y_hid[j]],
                            mode="lines",
                            line=dict(color=line_color, width=w_norm),
                            hoverinfo="text",
                            text=f"α_{i+1},{j+1} = {w_val:.3f}",
                            showlegend=False,
                        )
                    )

            # Layer 2 weights: beta
            W2 = mlp.coefs_[1]  # shape (M, 1)
            for j in range(m_hidden):
                w_val = W2[j, 0]
                w_norm = min(4.0, max(0.8, abs(w_val) * 1.5))
                line_color = "rgba(56, 189, 248, 0.55)" if w_val >= 0 else "rgba(245, 158, 11, 0.55)"
                fig_net.add_trace(
                    go.Scatter(
                        x=[2, 4],
                        y=[y_hid[j], y_out[0]],
                        mode="lines",
                        line=dict(color=line_color, width=w_norm),
                        hoverinfo="text",
                        text=f"β_{j+1},1 = {w_val:.3f}",
                        showlegend=False,
                    )
                )

            # Node markers
            fig_net.add_trace(
                go.Scatter(
                    x=[0] * p_in,
                    y=y_in,
                    mode="markers+text",
                    marker=dict(color="#38BDF8", size=22, line=dict(color="#FFFFFF", width=1.5)),
                    text=[f"x{i+1}" for i in range(p_in)],
                    textposition="middle left",
                    name="Input Layer (p=2)",
                    hoverinfo="text",
                )
            )

            fig_net.add_trace(
                go.Scatter(
                    x=[2] * m_hidden,
                    y=y_hid,
                    mode="markers+text",
                    marker=dict(color="#10B981", size=20, line=dict(color="#FFFFFF", width=1.5)),
                    text=[f"Z{j+1}" for j in range(m_hidden)],
                    textposition="top center",
                    name=f"Hidden Layer (M={m_hidden})",
                    hoverinfo="text",
                )
            )

            fig_net.add_trace(
                go.Scatter(
                    x=[4] * k_out,
                    y=y_out,
                    mode="markers+text",
                    marker=dict(color="#F59E0B", size=24, line=dict(color="#FFFFFF", width=1.5)),
                    text=["ŷ"],
                    textposition="middle right",
                    name="Output Layer (K=1)",
                    hoverinfo="text",
                )
            )

            fig_net.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(visible=False, range=[-0.8, 4.8]),
                yaxis=dict(visible=False, range=[-2.6, 2.6]),
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

            st.plotly_chart(fig_net, width="stretch", key="vnn_fig_net")

        with tab_boundary:
            st.markdown(f"##### 🎯 Decision Boundary & Output Probability Surface (M = {m_hidden})")

            fig_bound = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=420)

            fig_bound.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=prob_grid,
                    colorscale="Tealgrn",
                    opacity=0.65,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="P(Y=1)", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    contours=dict(start=0.1, end=0.9, size=0.1),
                    hoverinfo="skip",
                )
            )

            fig_bound.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=prob_grid,
                    contours_coloring="none",
                    contours=dict(start=0.5, end=0.5, size=0.1, coloring="none"),
                    line=dict(color="#FFFFFF", width=2.5),
                    name="Boundary (P = 0.5)",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            fig_bound = add_scatter_points(fig_bound, X, y_svm, marker_size=7)

            st.plotly_chart(fig_bound, width="stretch", key="vnn_fig_bound")

        with tab_basis:
            st.markdown("##### 📶 Hidden Activation Surfaces: How Hidden Neurons Synthesize Non-linear Manifolds")

            active_unit = st.slider("Select Hidden Neuron Z_m", 1, m_hidden, 1, 1, key="vnn_basis_sel")
            j_idx = active_unit - 1

            # Compute hidden activation Z_m(x) on grid
            W1 = mlp.coefs_[0]
            b1 = mlp.intercepts_[0]
            linear_proj = grid_pts @ W1[:, j_idx] + b1[j_idx]
            if activation_choice == "logistic":
                zm_grid = 1.0 / (1.0 + np.exp(-linear_proj))
            elif activation_choice == "relu":
                zm_grid = np.maximum(0.0, linear_proj)
            else:
                zm_grid = np.tanh(linear_proj)
            zm_surf = zm_grid.reshape(gxx.shape)

            fig_basis = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=420)

            fig_basis.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=zm_surf,
                    colorscale="Viridis",
                    opacity=0.75,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=f"Activation Z_{active_unit}", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    hoverinfo="skip",
                )
            )

            st.plotly_chart(fig_basis, width="stretch", key="vnn_fig_basis")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The Vanilla Neural Net (Slides 311–312, 315)", expanded=False):
        st.markdown(
            r"""
            **Mathematical Setup (Slides 311–312)**:
            - **Input Vector**: $x = (x_1, \dots, x_p)^T$.
            - **Hidden Layer Activations**:
              $$Z_m(x) = \sigma\left(\alpha_{0m} + \alpha_m^T x\right), \quad m = 1, \dots, M$$
              where $\alpha_m = (\alpha_{1m}, \dots, \alpha_{pm})^T$ are input-to-hidden connection weights, $\alpha_{0m}$ is the hidden node bias, and $\sigma(\cdot)$ is the non-linear activation function.
            - **Output Layer Prediction**:
              $$\hat{y}_k(x) = g_k(x) = \beta_{0k} + \sum_{m=1}^M \beta_{mk} Z_m(x), \quad k = 1, \dots, K$$
              where $\beta_{mk}$ are hidden-to-output connection weights and $\beta_{0k}$ is the output node bias.

            **Parameter Accounting (Slide 315)**:
            - Input-to-hidden parameters: $\{\alpha_{\ell m} \mid \ell = 0, \dots, p, \, m = 1, \dots, M\} \implies (p + 1)M$ weights.
            - Hidden-to-output parameters: $\{\beta_{mk} \mid m = 0, \dots, M, \, k = 1, \dots, K\} \implies (M + 1)K$ weights.
            - Total parameter count:
              $$\theta = (p + 1)M + (M + 1)K$$
            """
        )
