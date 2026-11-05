"""
Tile 3: Localized Kernel Regression (Nadaraya-Watson Estimator)
Slides: 232-233
Focus: Exact interactive replication of Slide 233.
f̂(x₀) = ∑ k_λ(x₀, xᵢ) yᵢ / ∑ k_λ(x₀, xᵢ)
with Epanechnikov quadratic window and Gaussian kernel, query probe x₀, and window width λ.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.datasets import get_1d_regression_dataset
from utils.plotting import BG_PAPER, BG_PLOT, COLOR_CLASS_POS


def epanechnikov_kernel(u):
    """Epanechnikov quadratic kernel: 3/4 * (1 - u²) for |u| <= 1, else 0."""
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2), 0.0)


def gaussian_kernel(u):
    """Standard Gaussian kernel: exp(-0.5 * u²)."""
    return np.exp(-0.5 * (u ** 2))


def nadaraya_watson_predict(x_query, x_obs, y_obs, bandwidth, kernel_func):
    """Compute Nadaraya-Watson kernel-weighted averages for an array of query points."""
    preds = np.zeros_like(x_query)
    for idx, x0 in enumerate(x_query):
        u = (x_obs - x0) / bandwidth
        w = kernel_func(u)
        sum_w = np.sum(w)
        if sum_w > 1e-10:
            preds[idx] = np.sum(w * y_obs) / sum_w
        else:
            # Nearest neighbor fallback if window is completely empty
            nearest = np.argmin(np.abs(x_obs - x0))
            preds[idx] = y_obs[nearest]
    return preds


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Kernel Regression Controls")

            kernel_type = st.radio(
                "Kernel Function k_λ(x₀, x)",
                ["Epanechnikov Quadratic", "Gaussian / RBF"],
                index=0,
                key="kr_kernel_type",
                help="Weighting kernel shape. Epanechnikov has compact support on [x₀ - λ, x₀ + λ].",
            )

            x0_probe = st.slider(
                "Query Probe Location x₀",
                min_value=0.08,
                max_value=0.92,
                value=0.50,
                step=0.02,
                key="kr_x0",
                help="Point where localized prediction f̂(x₀) is evaluated.",
            )

            bandwidth = st.slider(
                "Kernel Window Width λ",
                min_value=0.04,
                max_value=0.35,
                value=0.20,
                step=0.02,
                key="kr_lambda",
                help="Kernel bandwidth parameter λ (half-width of window). Default is λ = 0.20.",
            )

            with st.expander("⚙️ Synthetic Data Settings", expanded=False):
                n_samples = st.slider("Observation Count N", 30, 120, 60, 10, key="kr_samples")
                noise = st.slider("Noise σ_ε", 0.05, 0.35, 0.18, 0.02, key="kr_noise")

    # Generate dataset matching Slide 233
    x_dense, y_true, x_obs, y_obs = get_1d_regression_dataset(n_samples=n_samples, noise=noise, random_state=42)

    kernel_func = epanechnikov_kernel if "Epanechnikov" in kernel_type else gaussian_kernel

    # Compute Nadaraya-Watson curve across dense grid
    y_pred_curve = nadaraya_watson_predict(x_dense, x_obs, y_obs, bandwidth, kernel_func)

    # Compute prediction at probe point x0
    y_pred_x0 = nadaraya_watson_predict(np.array([x0_probe]), x_obs, y_obs, bandwidth, kernel_func)[0]

    # Compute local weights for observations at probe x0
    u_obs = (x_obs - x0_probe) / bandwidth
    weights_obs = kernel_func(u_obs)
    sum_w = np.sum(weights_obs)
    norm_weights = weights_obs / (sum_w if sum_w > 1e-10 else 1.0)
    in_window_mask = np.abs(u_obs) <= 1.0 if "Epanechnikov" in kernel_type else (norm_weights > 0.01)
    active_count = np.sum(in_window_mask)

    # Mean squared error against noisy observations
    y_pred_obs = nadaraya_watson_predict(x_obs, x_obs, y_obs, bandwidth, kernel_func)
    mse = np.mean((y_obs - y_pred_obs) ** 2)

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Bandwidth λ", f"{bandwidth:.2f}")
            m2.metric("Active Window Points", f"{active_count} / {len(x_obs)}")
            m3.metric("Prediction f̂(x₀)", f"{y_pred_x0:.3f}")
            m4.metric("Empirical MSE", f"{mse:.3f}")

        tab_main, tab_weights = st.tabs([
            "📊 Localized Kernel Average f̂(x₀)",
            "⚖️ Normalized Weights wᵢ(x₀) Across Observations",
        ])

        with tab_main:
            st.markdown(f"##### 🎯 Nadaraya-Watson Regression (λ = {bandwidth:.2f}, x₀ = {x0_probe:.2f})")

            fig = go.Figure()

            # 1. Shaded localized kernel window centered at x0
            if "Epanechnikov" in kernel_type:
                x_win = np.linspace(max(0.0, x0_probe - bandwidth), min(1.0, x0_probe + bandwidth), 100)
                u_win = (x_win - x0_probe) / bandwidth
                # Scale window visually to match the solid orange dome in Slide 233
                w_win = epanechnikov_kernel(u_win) * 0.9 + y_pred_x0 - 0.4
                y_baseline = np.full_like(x_win, y_pred_x0 - 0.4)

                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_win, x_win[::-1]]),
                        y=np.concatenate([w_win, y_baseline[::-1]]),
                        fill="toself",
                        fillcolor="rgba(245, 158, 11, 0.28)",
                        line=dict(color="#F59E0B", width=1.8),
                        name="Epanechnikov Window k_λ(x₀, x)",
                        hoverinfo="skip",
                    )
                )
            else:
                x_win = np.linspace(max(0.0, x0_probe - 2.5 * bandwidth), min(1.0, x0_probe + 2.5 * bandwidth), 120)
                u_win = (x_win - x0_probe) / bandwidth
                w_win = gaussian_kernel(u_win) * 0.9 + y_pred_x0 - 0.4
                y_baseline = np.full_like(x_win, y_pred_x0 - 0.4)

                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_win, x_win[::-1]]),
                        y=np.concatenate([w_win, y_baseline[::-1]]),
                        fill="toself",
                        fillcolor="rgba(245, 158, 11, 0.28)",
                        line=dict(color="#F59E0B", width=1.8),
                        name="Gaussian Window k_λ(x₀, x)",
                        hoverinfo="skip",
                    )
                )

            # 2. Underlying true function (Blue curve, Slide 233)
            fig.add_trace(
                go.Scatter(
                    x=x_dense,
                    y=y_true,
                    mode="lines",
                    name="Underlying Function f(x)",
                    line=dict(color="#38BDF8", width=2.5),
                    hovertemplate="x: %{x:.2f}<br>True f(x): %{y:.2f}<extra></extra>",
                )
            )

            # 3. Fitted Nadaraya-Watson curve (Green curve, Slide 233)
            fig.add_trace(
                go.Scatter(
                    x=x_dense,
                    y=y_pred_curve,
                    mode="lines",
                    name="Fitted Kernel Regression f̂(x)",
                    line=dict(color="#10B981", width=3.5),
                    hovertemplate="x: %{x:.2f}<br>f̂(x): %{y:.2f}<extra></extra>",
                )
            )

            # 4. Inactive noisy observations (circles outside window)
            out_mask = ~in_window_mask
            if np.any(out_mask):
                fig.add_trace(
                    go.Scatter(
                        x=x_obs[out_mask],
                        y=y_obs[out_mask],
                        mode="markers",
                        name="Outside Window (Weight ≈ 0)",
                        marker=dict(size=7, color="rgba(148, 163, 184, 0.3)", line=dict(color="#64748B", width=1.2)),
                        hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<extra>Zero Weight</extra>",
                    )
                )

            # 5. Active noisy observations inside window (highlighted yellow circles, Slide 233)
            if np.any(in_window_mask):
                active_sizes = 8 + norm_weights[in_window_mask] * 40
                fig.add_trace(
                    go.Scatter(
                        x=x_obs[in_window_mask],
                        y=y_obs[in_window_mask],
                        mode="markers",
                        name="In Window (Active Weight)",
                        marker=dict(
                            size=active_sizes,
                            color="#FBBF24",
                            line=dict(color="#FFFFFF", width=1.5),
                        ),
                        customdata=norm_weights[in_window_mask],
                        hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>Norm Weight: %{customdata:.3f}<extra></extra>",
                    )
                )

            # 6. Probe line at x0 and predicted marker
            fig.add_vline(x=x0_probe, line_width=1.5, line_dash="dash", line_color="#E2E8F0")

            fig.add_trace(
                go.Scatter(
                    x=[x0_probe],
                    y=[y_pred_x0],
                    mode="markers",
                    name="Query Evaluation f̂(x₀)",
                    marker=dict(size=14, color="#F43F5E", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                    hovertemplate=f"Probe x₀: {x0_probe:.2f}<br>f̂(x₀): {y_pred_x0:.3f}<extra></extra>",
                )
            )

            fig.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Input Domain x", font=dict(color="#CBD5E1", size=11)),
                    range=[-0.02, 1.02],
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Output y", font=dict(color="#CBD5E1", size=11)),
                    range=[-1.4, 1.8],
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                margin=dict(l=45, r=25, t=15, b=75),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=10),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig, width="stretch", key="kr_fig_main")

        with tab_weights:
            st.markdown(f"##### 📶 Observation Weight Profile wᵢ = k_λ(x₀, xᵢ) / ∑ k_λ(x₀, x_j) at Probe x₀ = {x0_probe:.2f}")

            fig_bar = go.Figure()
            bar_colors = ["#FBBF24" if in_window_mask[i] else "#475569" for i in range(len(x_obs))]

            fig_bar.add_trace(
                go.Bar(
                    x=np.arange(len(x_obs)),
                    y=norm_weights,
                    marker=dict(color=bar_colors),
                    customdata=x_obs,
                    hovertemplate="Obs #%{x}<br>x: %{customdata:.2f}<br>Weight: %{y:.4f}<extra></extra>",
                    name="Normalized Weights",
                )
            )

            fig_bar.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Training Observation Index (Sorted by x)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Normalized Weight wᵢ", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                margin=dict(l=45, r=25, t=15, b=75),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=10),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_bar, width="stretch", key="kr_fig_bar")
