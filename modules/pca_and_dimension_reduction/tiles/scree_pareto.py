"""
Tile 2: Scree and Pareto Diagnostic Plots
Focus: Slides 334-336 - Selection of retained principal components q < p.
Directly replicates Slide 336: Scree plot (elbow bend heuristic) and Pareto plot
(individual eigenvalue bars + cumulative variance curve with 85% cut-off threshold).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from utils.plotting import (
    COLOR_HYPERPLANE,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Scree & Pareto Knobs")

            total_p = st.slider(
                "Total Feature Dimension p",
                min_value=5,
                max_value=16,
                value=8,
                step=1,
                key="sp_p",
                help="Total number of input features in the original data matrix X.",
            )

            decay_profile = st.selectbox(
                "Eigenvalue Decay Profile",
                [
                    "Sharp Elbow at k = 3",
                    "Steep Exponential Decay (1-2 Dominant PCs)",
                    "Gradual Linear Decay (Diffuse Variance)",
                ],
                index=0,
                key="sp_decay",
                help="Synthesizes realistic covariance spectrum profiles.",
            )

            cutoff_pct = st.slider(
                "Pareto Variance Cutoff (%)",
                min_value=60,
                max_value=98,
                value=85,
                step=5,
                key="sp_cutoff",
                help="Pre-set percentage rule (commonly 85%).",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Count N", 80, 400, 200, 20, key="sp_n")
                noise_scale = st.slider("Noise Floor", 0.05, 0.40, 0.15, 0.05, key="sp_noise")
                seed_val = st.number_input("Random Seed", value=42, step=1, key="sp_seed")

    # Generate synthetic p-dimensional covariance spectrum
    rng = np.random.RandomState(seed_val)
    p_indices = np.arange(1, total_p + 1)

    if "Sharp Elbow" in decay_profile:
        # Recreate exact Slide 336 pattern: 3.5, 1.4, 0.5, 0.25, 0.15, 0.08...
        raw_eigs = np.array([3.6, 1.4, 0.5, 0.25, 0.16, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])[:total_p]
    elif "Steep Exponential" in decay_profile:
        raw_eigs = 4.0 * np.exp(-0.8 * (p_indices - 1)) + noise_scale
    else:
        raw_eigs = np.maximum(0.1, 3.0 - 0.22 * p_indices) + rng.uniform(0, noise_scale, total_p)

    # Sort descending
    raw_eigs = np.sort(raw_eigs)[::-1]
    tot_var = np.sum(raw_eigs)
    pct_vars = (raw_eigs / tot_var) * 100.0
    cum_vars = np.cumsum(pct_vars)

    # Find cutoff q* for pre-set percentage (Slide 334)
    q_star = int(np.argmax(cum_vars >= cutoff_pct) + 1)

    # Automatic elbow detection via maximum second difference (curvature)
    if total_p >= 4:
        diff2 = np.diff(raw_eigs, 2)  # length p-2
        elbow_k = int(np.argmax(diff2) + 2)
    else:
        elbow_k = 2

    # Simulate reconstruction MSE as function of q: ∑_{j=q+1}^p λ_j
    recon_mse_curve = [float(np.sum(raw_eigs[q:])) for q in range(1, total_p + 1)]

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended q*", f"{q_star} of {total_p}", f"{(total_p - q_star) / total_p * 100:.0f}% compression")
        m2.metric("Cumulative Var at q*", f"{cum_vars[q_star - 1]:.1f}%", help=f"Meets >= {cutoff_pct}% threshold rule")
        m3.metric("Detected Elbow k", f"k = {elbow_k}", help="Point on Scree plot where eigenvalues level off")
        m4.metric("PC1 Dominance", f"{pct_vars[0]:.1f}%")

        tab_scree, tab_pareto, tab_recon = st.tabs([
            "📉 Scree Plot",
            "📊 Pareto Plot of Eigenvalues",
            "🗜️ Reconstruction Loss vs q",
        ])

        with tab_scree:
            st.markdown("##### 📉 Scree Plot: Ordered Eigenvalues λ_i vs Component Index i")

            fig_scree = go.Figure()

            # Scree line and markers
            fig_scree.add_trace(
                go.Scatter(
                    x=list(p_indices),
                    y=raw_eigs,
                    mode="lines+markers",
                    name="Eigenvalue λ_i",
                    line=dict(color=COLOR_HYPERPLANE, width=3.0),
                    marker=dict(size=8, color=COLOR_HYPERPLANE),
                    hovertemplate="PC %{x}: λ = %{y:.3f}<extra></extra>",
                )
            )

            # Highlight detected elbow
            fig_scree.add_trace(
                go.Scatter(
                    x=[elbow_k],
                    y=[raw_eigs[elbow_k - 1]],
                    mode="markers+text",
                    name=f"Elbow Bend (k = {elbow_k})",
                    marker=dict(size=14, color="#F59E0B", symbol="circle-open", line=dict(width=2.5)),
                    text=[f"Elbow (k={elbow_k})"],
                    textposition="top right",
                    textfont=dict(color="#F59E0B", size=11),
                    hoverinfo="text",
                    hovertext=f"Elbow point: k={elbow_k}, λ={raw_eigs[elbow_k-1]:.3f}",
                )
            )

            fig_scree.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Principal Component Index i", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    tickmode="linear",
                    dtick=1,
                ),
                yaxis=dict(
                    title=dict(text="Eigenvalue λ̂_i (Variance)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_scree, width="stretch", key="sp_fig_scree")

        with tab_pareto:
            st.markdown(f"##### 📊 Pareto Plot: Individual % Variance (Bars) & Cumulative Variance (Line)")

            fig_pareto = go.Figure()

            # Bar chart of individual variance percentages
            bar_colors = ["#10B981" if idx < q_star else "#64748B" for idx in range(total_p)]
            fig_pareto.add_trace(
                go.Bar(
                    x=list(p_indices),
                    y=pct_vars,
                    name="Individual Variance %",
                    marker=dict(color=bar_colors),
                    text=[f"{v:.1f}%" for v in pct_vars],
                    textposition="outside",
                    hovertemplate="PC %{x}: %{y:.1f}% Variance<extra></extra>",
                    yaxis="y1",
                )
            )

            # Cumulative percentage line curve
            fig_pareto.add_trace(
                go.Scatter(
                    x=list(p_indices),
                    y=cum_vars,
                    mode="lines+markers",
                    name="Cumulative Variance %",
                    line=dict(color="#38BDF8", width=3.0),
                    marker=dict(size=7, color="#38BDF8"),
                    hovertemplate="Through PC %{x}: %{y:.1f}% Total<extra></extra>",
                    yaxis="y2",
                )
            )

            # Cutoff threshold horizontal line
            fig_pareto.add_hline(
                y=cutoff_pct,
                yref="y2",
                line_width=1.8,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"Cutoff Threshold ({cutoff_pct}%) → Keep q* = {q_star} PCs",
                annotation_font_color="#F59E0B",
                annotation_position="bottom right",
            )

            fig_pareto.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Principal Components", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    tickmode="linear",
                    dtick=1,
                ),
                yaxis=dict(
                    title=dict(text="Individual Variance (%)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    range=[0, max(pct_vars) * 1.35],
                ),
                yaxis2=dict(
                    title=dict(text="Cumulative Variance (%)", font=dict(color="#38BDF8", size=11)),
                    overlaying="y",
                    side="right",
                    range=[0, 105],
                    showgrid=False,
                    tickfont=dict(color="#38BDF8", size=9.5),
                ),
                margin=dict(l=40, r=40, t=15, b=85),
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

            st.plotly_chart(fig_pareto, width="stretch", key="sp_fig_pareto")

        with tab_recon:
            st.markdown(r"##### 🗜️ Reconstruction Error $\sum_{j=q+1}^p \lambda_j$ as Function of Retained Components $q$")

            fig_recon = go.Figure()

            fig_recon.add_trace(
                go.Scatter(
                    x=list(p_indices),
                    y=recon_mse_curve,
                    mode="lines+markers",
                    name="Reconstruction Information Loss",
                    line=dict(color="#EF4444", width=2.8),
                    marker=dict(size=7, color="#EF4444"),
                    hovertemplate="Retain q = %{x} PCs: Loss = %{y:.4f}<extra></extra>",
                )
            )

            fig_recon.add_vline(
                x=q_star,
                line_width=1.8,
                line_dash="dot",
                line_color=COLOR_HYPERPLANE,
                annotation_text=f"q* = {q_star}",
                annotation_font_color=COLOR_HYPERPLANE,
                annotation_position="top right",
            )

            fig_recon.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Number of Retained Components q", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    tickmode="linear",
                    dtick=1,
                ),
                yaxis=dict(
                    title=dict(text="Information Loss ||X - X̂_q||²", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_recon, width="stretch", key="sp_fig_recon")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Retaining Principal Components", expanded=False):
        st.markdown(
            r"""
            **How Many PCs to Keep?**:
            - If we keep all $p$ components ($q = p$), the transformation is lossless ($\text{tr}(S_z) = \text{tr}(S_x)$), but no dimension reduction is achieved.
            - To compress the data space from $\mathbb{R}^p$ to $\mathbb{R}^q$ with minimal distortion, two complementary graphical tools are used:
            
            1. **The Pre-Set Percentage / Pareto Rule**:
               - Select the smallest $q^*$ such that the cumulative explained variance surpasses a prescribed threshold:
                 $$q^* = \arg\min_q q \quad \text{subject to } \frac{\sum_{i=1}^q \lambda_i}{\sum_{i=1}^p \lambda_i} \times 100\% \ge \text{threshold (e.g., 85\%)}$$
            
            2. **The Scree Plot & Elbow Criterion**:
               - Plot eigenvalues $\lambda_i$ in descending order against component index $i$.
               - Look for an **elbow (bend)** where the steep drop levels off and subsequent eigenvalues become small, roughly equal background noise.
               - Retain the components preceding the elbow.
            """
        )
