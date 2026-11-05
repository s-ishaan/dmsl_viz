"""
Tile 4: Non-Separable Data & Slack Variables (ξ)
Slides: 219-220
Focus: Geometric interpretation of slack variables ξᵢ, margin violations (0 < ξᵢ ≤ 1),
misclassifications (ξᵢ > 1), and slack vector visualization.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_linear_boundary_and_margins,
    COLOR_HYPERPLANE,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### ⚙️ Slack Configuration")
            dataset_type = st.selectbox(
                "Dataset Pattern",
                ["Overlapping Blobs", "Moons", "Linearly Separable Blobs"],
                index=0,
                key="sm_data_type",
            )

            cost_C = st.slider(
                "Cost Regularization C",
                min_value=0.05,
                max_value=10.0,
                value=1.0,
                step=0.2,
                key="sm_C",
                help="Penalizes slack sum: min 1/2 ||β||² + C ∑ ξᵢ",
            )

            noise_level = st.slider(
                "Overlap / Noise Level",
                min_value=0.1,
                max_value=0.8,
                value=0.35,
                step=0.05,
                key="sm_noise",
                help="Higher overlap creates points that cannot be separated without slack allowance",
            )

            with st.expander("⚙️ Display & Sampling Options", expanded=False):
                n_samples = st.slider("Sample Count N", 20, 100, 50, step=10, key="sm_n")
                show_slack_vectors = st.checkbox("Draw Directed Slack Vectors", value=True, key="sm_show_slack_vec")

    # Generate data
    X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise_level, random_state=42)

    # Fit linear SVM with soft margin
    clf = SVC(kernel="linear", C=cost_C)
    clf.fit(X, y)

    w = clf.coef_[0]
    b = clf.intercept_[0]
    norm_w = np.linalg.norm(w)

    # Functional margins: f_i = y_i * (w^T x_i + b)
    func_margins = y * (np.dot(X, w) + b)
    slacks = np.maximum(0.0, 1.0 - func_margins)

    idx_safe = np.where(slacks < 1e-4)[0]
    idx_margin_violation = np.where((slacks >= 1e-4) & (slacks <= 1.0))[0]
    idx_misclassified = np.where(slacks > 1.0)[0]

    total_slack = np.sum(slacks)
    n_safe = len(idx_safe)
    n_margin_viol = len(idx_margin_violation)
    n_misclass = len(idx_misclassified)

    with col_plot:
        # Metrics container
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Slack ∑ ξᵢ", f"{total_slack:.2f}")
            m2.metric("Safe Points (ξ=0)", f"{n_safe}")
            m3.metric("Inside Margin (0<ξ≤1)", f"{n_margin_viol}")
            m4.metric("Misclassified (ξ>1)", f"{n_misclass}")

        tab_2d, tab_dist = st.tabs(["🎯 2D Margin & Slack Vectors", "📊 Slack Distribution & Budget"])

        with tab_2d:
            st.markdown("##### 🎯 Non-Separable Margin & Slack Vectors (ξᵢ)")
            x_min, x_max = float(np.min(X[:, 0]) - 1.2), float(np.max(X[:, 0]) + 1.2)
            y_min, y_max = float(np.min(X[:, 1]) - 1.2), float(np.max(X[:, 1]) + 1.2)

            fig = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=440)

            # 1. Safe points (ξ = 0)
            if len(idx_safe) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=X[idx_safe, 0],
                        y=X[idx_safe, 1],
                        mode="markers",
                        name="Safe: ξᵢ = 0",
                        marker=dict(
                            color="#10B981",  # Emerald Green
                            size=9,
                            symbol="circle",
                            line=dict(color="#FFFFFF", width=1),
                        ),
                        hovertemplate="<b>Safe (ξ=0)</b><br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>",
                    )
                )

            # 2. Margin violations (0 < ξ ≤ 1)
            if len(idx_margin_violation) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=X[idx_margin_violation, 0],
                        y=X[idx_margin_violation, 1],
                        customdata=slacks[idx_margin_violation],
                        mode="markers",
                        name="Margin Violation: 0 < ξᵢ ≤ 1",
                        marker=dict(
                            color="#F59E0B",  # Amber
                            size=11,
                            symbol="diamond",
                            line=dict(color="#FFFFFF", width=1.2),
                        ),
                        hovertemplate="<b>Margin Violation</b><br>Slack ξ: %{customdata:.2f}<br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>",
                    )
                )

            # 3. Misclassifications (ξ > 1)
            if len(idx_misclassified) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=X[idx_misclassified, 0],
                        y=X[idx_misclassified, 1],
                        customdata=slacks[idx_misclassified],
                        mode="markers",
                        name="Misclassified: ξᵢ > 1",
                        marker=dict(
                            color="#EF4444",  # Red
                            size=12,
                            symbol="square",
                            line=dict(color="#FFFFFF", width=1.5),
                        ),
                        hovertemplate="<b>Misclassified</b><br>Slack ξ: %{customdata:.2f}<br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>",
                    )
                )

            # Decision boundary and margins
            fig = add_linear_boundary_and_margins(
                fig,
                w,
                b,
                (x_min, x_max),
                (y_min, y_max),
                show_margins=True,
                show_margin_band=True,
                boundary_color=COLOR_HYPERPLANE,
                boundary_name="Hyperplane: f(x)=0",
                line_width=3,
            )

            # Draw slack vectors
            if show_slack_vectors:
                viol_indices = np.where(slacks > 1e-4)[0]
                slack_x = []
                slack_y = []
                for idx in viol_indices:
                    xi = X[idx]
                    yi = y[idx]
                    current_val = np.dot(w, xi) + b
                    target_val = yi * 1.0
                    dist_shift = (target_val - current_val) / (norm_w ** 2)
                    x_margin_target = xi + dist_shift * w

                    slack_x.extend([xi[0], x_margin_target[0], None])
                    slack_y.extend([xi[1], x_margin_target[1], None])

                fig.add_trace(
                    go.Scatter(
                        x=slack_x,
                        y=slack_y,
                        mode="lines",
                        name="Slack Vectors (ξᵢ)",
                        line=dict(color="#EC4899", width=2, dash="dash"),
                        hoverinfo="skip",
                    )
                )

            st.plotly_chart(fig, width="stretch")

        with tab_dist:
            if (n_margin_viol + n_misclass) > 0:
                viol_indices = np.where(slacks > 1e-4)[0]
                viol_indices_sorted = viol_indices[np.argsort(-slacks[viol_indices])]
                sorted_slacks = slacks[viol_indices_sorted]
                labels = [f"Pt #{i+1} (y={y[i]})" for i in viol_indices_sorted]
                bar_colors = ["#EF4444" if s > 1.0 else "#F59E0B" for s in sorted_slacks]

                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=labels,
                        y=sorted_slacks,
                        marker=dict(color=bar_colors),
                        name="Slack ξᵢ",
                    )
                )
                fig_bar.add_hline(
                    y=1.0,
                    line_dash="dot",
                    line_color="#EF4444",
                    annotation_text="Misclassification (ξ = 1.0)",
                    annotation_position="top right",
                    annotation_font=dict(color="#EF4444", size=11),
                )
                st.markdown("##### 📊 Ranked Slack Magnitudes (ξᵢ)")
                fig_bar.update_layout(
                    height=340,
                    margin=dict(l=40, r=40, t=15, b=30),
                    paper_bgcolor="#1E293B",
                    plot_bgcolor="#0F172A",
                    xaxis=dict(tickfont=dict(color="#94A3B8", size=9)),
                    yaxis=dict(
                        title=dict(text="Slack ξᵢ", font=dict(color="#CBD5E1", size=11)),
                        tickfont=dict(color="#94A3B8"),
                        gridcolor="#334155",
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("No points violate the margin under current settings (All ξᵢ = 0).")
