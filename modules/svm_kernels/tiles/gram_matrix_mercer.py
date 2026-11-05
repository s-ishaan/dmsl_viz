"""
Tile 4: Gram Matrix Spectrum & Mercer's Theorem
Slides: 234-237, 241
Focus: Exploring the Gram (Kernel) matrix K_ij = k(xᵢ, xⱼ) ⪰ 0,
Mercer's spectral expansion k(x, x') = ∑ δᵢ ψᵢ(x) ψᵢ(x'),
class-block similarity structures, and eigenvalue spectral decay (effective RKHS dimensionality).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from utils.datasets import get_dataset
from utils.plotting import BG_PAPER, BG_PLOT, COLOR_HYPERPLANE


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Gram Matrix Controls")

            kernel_name = st.selectbox(
                "Kernel Function",
                ["RBF / Gaussian: exp(-γ||x - x'||²)", "Polynomial: (1 + ⟨x, x'⟩)ᵈ", "Linear: ⟨x, x'⟩"],
                index=0,
                key="gm_kernel_name",
                help="Mathematical kernel used to compute inner products in RKHS (Slides 236-240).",
            )

            if "RBF" in kernel_name:
                gamma_exp = st.slider(
                    "Kernel log₁₀(γ)",
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.25,
                    key="gm_gamma",
                    help="RBF inverse bandwidth parameter γ.",
                )
                param_val = 10.0 ** gamma_exp
            elif "Polynomial" in kernel_name:
                degree = st.slider(
                    "Polynomial Degree d",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1,
                    key="gm_degree",
                    help="Degree d in (1 + ⟨x, x'⟩)ᵈ.",
                )
                param_val = degree
            else:
                param_val = 1

            dataset_type = st.selectbox(
                "Dataset Topology",
                ["Concentric Circles", "Moons", "Linearly Separable Blobs"],
                index=0,
                key="gm_dataset",
                help="Geometric structure of sample points.",
            )

            sort_by_class = st.checkbox(
                "Sort Points by Class Label {-1, +1}",
                value=True,
                key="gm_sort_class",
                help="Clusters points of the same class together to reveal block-diagonal similarity.",
            )

            with st.expander("⚙️ Matrix Sizing", expanded=False):
                n_samples = st.slider("Sample Points N", 20, 80, 40, 4, key="gm_samples")
                noise = st.slider("Dataset Noise", 0.05, 0.30, 0.10, 0.05, key="gm_noise")

    # Generate dataset
    X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)

    if sort_by_class:
        sort_idx = np.argsort(y)
        X = X[sort_idx]
        y = y[sort_idx]

    # Compute Gram Matrix K
    if "RBF" in kernel_name:
        K = rbf_kernel(X, gamma=param_val)
    elif "Polynomial" in kernel_name:
        K = polynomial_kernel(X, degree=param_val, coef0=1.0)
    else:
        K = linear_kernel(X)

    # Spectral decomposition of K (Mercer eigenvalues)
    eigvals, _ = np.linalg.eigh(K)
    # Sort eigenvalues in descending order
    eigvals = np.sort(eigvals)[::-1]
    min_eig = np.min(eigvals)
    pos_eigvals = np.maximum(eigvals, 0.0)
    total_energy = np.sum(pos_eigvals)
    cum_var = np.cumsum(pos_eigvals) / (total_energy if total_energy > 1e-10 else 1.0) * 100.0

    # Effective dimensionality: number of eigenvalues capturing >= 95% variance
    effective_dim = np.argmax(cum_var >= 95.0) + 1 if np.any(cum_var >= 95.0) else len(eigvals)

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Matrix Dimension", f"{n_samples} × {n_samples}")
            m2.metric("Min Eigenvalue λ_min", f"{min_eig:.4f}", help="λ_min >= 0 confirms Mercer Positive Semi-Definiteness (Slide 237).")
            m3.metric("Effective Rank (95%)", f"{effective_dim} / {n_samples}")
            m4.metric("Mercer Status", "✅ Positive Semi-Definite" if min_eig > -1e-5 else "⚠️ Numerical Warning")

        tab_heatmap, tab_spectrum = st.tabs([
            "🔥 N × N Gram Similarity Heatmap K_ij = k(xᵢ, xⱼ)",
            "📉 Mercer Spectral Eigenvalue Decay (RKHS Basis Energy)",
        ])

        with tab_heatmap:
            st.markdown(f"##### 🔲 Gram Matrix Heatmap ({kernel_name.split(':')[0]})")

            fig_hm = go.Figure()
            fig_hm.add_trace(
                go.Heatmap(
                    z=K,
                    x=[f"Obs {i+1} (y={y[i]})" for i in range(len(y))],
                    y=[f"Obs {i+1} (y={y[i]})" for i in range(len(y))],
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(text="k(xᵢ, xⱼ)", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    hovertemplate="Point i: %{y}<br>Point j: %{x}<br>Similarity K_ij: %{z:.3f}<extra></extra>",
                )
            )

            # If sorted by class, draw partition line between classes
            if sort_by_class:
                n_neg = np.sum(y == -1)
                fig_hm.add_hline(y=n_neg - 0.5, line_width=2, line_dash="dash", line_color="#F43F5E")
                fig_hm.add_vline(x=n_neg - 0.5, line_width=2, line_dash="dash", line_color="#F43F5E")

            fig_hm.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    showticklabels=False,
                    gridcolor="#1E293B",
                    title=dict(text="Instance Index j (Class -1 on Left, Class +1 on Right)" if sort_by_class else "Instance Index j", font=dict(color="#CBD5E1", size=11)),
                ),
                yaxis=dict(
                    showticklabels=False,
                    gridcolor="#1E293B",
                    autorange="reversed",
                    title=dict(text="Instance Index i (Class -1 on Top, Class +1 on Bottom)" if sort_by_class else "Instance Index i", font=dict(color="#CBD5E1", size=11)),
                ),
                margin=dict(l=45, r=25, t=15, b=60),
            )

            st.plotly_chart(fig_hm, width="stretch", key="gm_fig_hm")

        with tab_spectrum:
            st.markdown("##### 📊 Mercer Eigenvalues δᵢ & Cumulative Variance (Slide 241)")

            x_indices = np.arange(1, min(len(eigvals) + 1, 31))
            truncated_eigs = pos_eigvals[:len(x_indices)]
            truncated_cum = cum_var[:len(x_indices)]

            fig_spec = go.Figure()

            # Bar plot of individual eigenvalues
            fig_spec.add_trace(
                go.Bar(
                    x=x_indices,
                    y=truncated_eigs,
                    name="Eigenvalue δᵢ",
                    marker=dict(color=COLOR_HYPERPLANE),
                    hovertemplate="Basis #%{x}<br>Eigenvalue δᵢ: %{y:.3f}<extra></extra>",
                )
            )

            # Cumulative percentage line on secondary axis
            fig_spec.add_trace(
                go.Scatter(
                    x=x_indices,
                    y=truncated_cum,
                    mode="lines+markers",
                    name="Cumulative Energy %",
                    yaxis="y2",
                    line=dict(color="#38BDF8", width=2.5),
                    marker=dict(size=6, color="#38BDF8"),
                    hovertemplate="Basis #%{x}<br>Cumulative Energy: %{y:.1f}%<extra></extra>",
                )
            )

            fig_spec.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="RKHS Spectral Basis Index i", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Eigenvalue Magnitude δᵢ", font=dict(color=COLOR_HYPERPLANE, size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis2=dict(
                    title=dict(text="Cumulative Energy (%)", font=dict(color="#38BDF8", size=11)),
                    overlaying="y",
                    side="right",
                    range=[0, 105],
                    tickfont=dict(color="#94A3B8"),
                    showgrid=False,
                ),
                margin=dict(l=45, r=45, t=15, b=75),
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

            st.plotly_chart(fig_spec, width="stretch", key="gm_fig_spec")
