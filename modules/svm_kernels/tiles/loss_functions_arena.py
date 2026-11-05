"""
Tile 5: The Loss Function Arena (ESL Figures 10.4 & 12.4)
Slides: 251-257
Focus: Comparative analysis of surrogate classification loss functions plotted against the margin yf:
- 0-1 Misclassification: I(yf < 0)
- SVM Hinge: [1 - yf]₊
- Binomial Deviance (Logistic): log₂(1 + e^(-yf))
- Exponential (AdaBoost): e^(-yf)
- Squared Error (LDA): (1 - yf)²
- Huberized Square Hinge
Visualizes why Hinge loss produces sparsity (flat 0 for yf >= 1), why squared error hurts confident points,
and why exponential loss explodes on outliers.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.plotting import BG_PAPER, BG_PLOT


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Margin & Loss Controls")

            yf_probe = st.slider(
                "Margin Value y·f(x)",
                min_value=-3.0,
                max_value=3.0,
                value=0.5,
                step=0.1,
                key="lfa_yf_probe",
                help="Signed margin y·f(x). y·f > 0 is correct classification; y·f >= 1 is beyond the margin.",
            )

            scale_mode = st.radio(
                "Scaling Reference",
                ["Figure 10.4 (Normalized at yf=0 to Loss=1)", "Figure 12.4 (Natural Textbook Scale)"],
                index=0,
                key="lfa_scale",
                help="ESL Fig 10.4 normalizes all losses so Loss(0) = 1.0 (Slide 255).",
            )

            st.markdown("##### 🔍 Active Loss Functions")
            show_hinge = st.checkbox("SVM Hinge: [1 - yf]₊", value=True, key="lfa_hinge")
            show_binom = st.checkbox("Binomial Deviance: log₂(1 + e^(-yf))", value=True, key="lfa_binom")
            show_exp = st.checkbox("Exponential: e^(-yf)", value=True, key="lfa_exp")
            show_sq = st.checkbox("Squared Error: (1 - yf)²", value=True, key="lfa_sq")
            show_huber = st.checkbox("Huberized Square Hinge", value=True, key="lfa_huber")
            show_zero_one = st.checkbox("0-1 Misclassification: I(yf < 0)", value=True, key="lfa_01")

    # Domain grid
    yf = np.linspace(-3.0, 3.0, 400)
    norm = (scale_mode.startswith("Figure 10.4"))

    # Definitions
    loss_01 = np.where(yf < 0, 1.0, 0.0)
    loss_hinge = np.maximum(0.0, 1.0 - yf)
    loss_binom = np.log2(1.0 + np.exp(-yf)) if norm else np.log(1.0 + np.exp(-yf))
    loss_exp = np.exp(-yf)
    loss_sq = (1.0 - yf) ** 2 if not norm else 0.25 * ((1.0 - yf) ** 2)
    # Huberized square hinge: -4*yf for yf < -1, else [1 - yf]_+^2
    loss_huber = np.where(yf < -1.0, -4.0 * yf, np.maximum(0.0, 1.0 - yf) ** 2)

    # Compute values at probe point yf_probe
    p_01 = 1.0 if yf_probe < 0 else 0.0
    p_hinge = max(0.0, 1.0 - yf_probe)
    p_binom = np.log2(1.0 + np.exp(-yf_probe)) if norm else np.log(1.0 + np.exp(-yf_probe))
    p_exp = np.exp(-yf_probe)
    p_sq = (1.0 - yf_probe) ** 2 if not norm else 0.25 * ((1.0 - yf_probe) ** 2)
    p_huber = -4.0 * yf_probe if yf_probe < -1.0 else max(0.0, 1.0 - yf_probe) ** 2

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Margin y·f", f"{yf_probe:+.2f}")
            m2.metric("Hinge Loss", f"{p_hinge:.2f}", help="Zero when y·f >= 1 (Support Vector sparsity condition).")
            m3.metric("Binomial Deviance", f"{p_binom:.2f}")
            m4.metric("Exponential Loss", f"{p_exp:.2f}", help="Explodes rapidly when y·f < -1 (outlier sensitivity).")

        tab_curves, tab_outlier = st.tabs([
            "📈 Loss Curves vs Margin y·f (ESL Replica)",
            "💥 Outlier Penalty Stress-Test (yf = -3.0 vs -0.5)",
        ])

        with tab_curves:
            st.markdown(f"##### 🎯 Loss Functions for Classification ({scale_mode.split('(')[0].strip()})")

            fig = go.Figure()

            # Shaded flat-zero region for Hinge loss (yf >= 1)
            fig.add_vrect(
                x0=1.0,
                x1=3.0,
                fillcolor="rgba(16, 185, 129, 0.08)",
                layer="below",
                line_width=0,
                annotation_text="<b>Zero Loss Region: yf ≥ 1 (Non-SVs)</b>",
                annotation_position="top left",
                annotation_font=dict(color="#10B981", size=10),
            )

            # Shaded margin violation zone (0 <= yf < 1)
            fig.add_vrect(
                x0=0.0,
                x1=1.0,
                fillcolor="rgba(245, 158, 11, 0.08)",
                layer="below",
                line_width=0,
                annotation_text="<b>Margin Slack: 0 ≤ yf < 1</b>",
                annotation_position="top left",
                annotation_font=dict(color="#F59E0B", size=10),
            )

            # Misclassification zone (yf < 0)
            fig.add_vrect(
                x0=-3.0,
                x1=0.0,
                fillcolor="rgba(244, 63, 94, 0.08)",
                layer="below",
                line_width=0,
                annotation_text="<b>Misclassified: yf < 0</b>",
                annotation_position="top left",
                annotation_font=dict(color="#F43F5E", size=10),
            )

            # 0-1 Loss
            if show_zero_one:
                fig.add_trace(
                    go.Scatter(
                        x=yf,
                        y=loss_01,
                        mode="lines",
                        name="0-1 Misclassification",
                        line=dict(color="#94A3B8", width=2, dash="dash"),
                        hovertemplate="Margin: %{x:.2f}<br>0-1 Loss: %{y:.2f}<extra></extra>",
                    )
                )

            # Hinge Loss (Red / Crimson)
            if show_hinge:
                fig.add_trace(
                    go.Scatter(
                        x=yf,
                        y=loss_hinge,
                        mode="lines",
                        name="SVM Hinge Loss",
                        line=dict(color="#F43F5E", width=3.5),
                        hovertemplate="Margin: %{x:.2f}<br>Hinge Loss: %{y:.2f}<extra></extra>",
                    )
                )

            # Binomial Deviance (Emerald Green)
            if show_binom:
                fig.add_trace(
                    go.Scatter(
                        x=yf,
                        y=loss_binom,
                        mode="lines",
                        name="Binomial Deviance",
                        line=dict(color="#10B981", width=2.5),
                        hovertemplate="Margin: %{x:.2f}<br>Binomial Loss: %{y:.2f}<extra></extra>",
                    )
                )

            # Exponential Loss (Sky Blue)
            if show_exp:
                fig.add_trace(
                    go.Scatter(
                        x=yf,
                        y=loss_exp,
                        mode="lines",
                        name="Exponential Loss",
                        line=dict(color="#38BDF8", width=2.5),
                        hovertemplate="Margin: %{x:.2f}<br>Exp Loss: %{y:.2f}<extra></extra>",
                    )
                )

            # Squared Error Loss (Amber)
            if show_sq:
                fig.add_trace(
                    go.Scatter(
                        x=yf,
                        y=loss_sq,
                        mode="lines",
                        name="Squared Error Loss",
                        line=dict(color="#F59E0B", width=2),
                        hovertemplate="Margin: %{x:.2f}<br>Squared Loss: %{y:.2f}<extra></extra>",
                    )
                )

            # Huberized Square Hinge (Purple)
            if show_huber:
                fig.add_trace(
                    go.Scatter(
                        x=yf,
                        y=loss_huber,
                        mode="lines",
                        name="Huberized Square Hinge",
                        line=dict(color="#A855F7", width=2, dash="dot"),
                        hovertemplate="Margin: %{x:.2f}<br>Huberized Loss: %{y:.2f}<extra></extra>",
                    )
                )

            # Probe line at yf_probe
            fig.add_vline(x=yf_probe, line_width=1.5, line_dash="dash", line_color="#FFFFFF")

            # Add probe markers
            probe_x = [yf_probe]
            if show_hinge:
                fig.add_trace(go.Scatter(x=probe_x, y=[p_hinge], mode="markers", marker=dict(size=9, color="#F43F5E"), showlegend=False, hoverinfo="skip"))
            if show_binom:
                fig.add_trace(go.Scatter(x=probe_x, y=[p_binom], mode="markers", marker=dict(size=9, color="#10B981"), showlegend=False, hoverinfo="skip"))
            if show_exp:
                fig.add_trace(go.Scatter(x=probe_x, y=[p_exp], mode="markers", marker=dict(size=9, color="#38BDF8"), showlegend=False, hoverinfo="skip"))

            fig.update_layout(
                height=450,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Margin y · f(x)", font=dict(color="#CBD5E1", size=11)),
                    range=[-3.0, 3.0],
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Loss L(y, f(x))", font=dict(color="#CBD5E1", size=11)),
                    range=[-0.1, 4.2],
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

            st.plotly_chart(fig, width="stretch", key="lfa_fig_curves")

        with tab_outlier:
            st.markdown("##### 💥 Outlier Penalty Comparison (Severe Outlier yf = -3.0 vs Moderate Error yf = -0.5)")

            loss_names = ["SVM Hinge", "Binomial Deviance", "Exponential (AdaBoost)", "Squared Error"]
            val_outlier = [
                max(0.0, 1.0 - (-3.0)),
                np.log2(1.0 + np.exp(3.0)),
                np.exp(3.0),
                (1.0 - (-3.0)) ** 2,
            ]
            val_moderate = [
                max(0.0, 1.0 - (-0.5)),
                np.log2(1.0 + np.exp(0.5)),
                np.exp(0.5),
                (1.0 - (-0.5)) ** 2,
            ]

            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    x=loss_names,
                    y=val_moderate,
                    name="Moderate Error (yf = -0.5)",
                    marker=dict(color="#10B981"),
                    hovertemplate="%{x}<br>Moderate Loss: %{y:.2f}<extra></extra>",
                )
            )
            fig_bar.add_trace(
                go.Bar(
                    x=loss_names,
                    y=val_outlier,
                    name="Extreme Outlier (yf = -3.0)",
                    marker=dict(color="#F43F5E"),
                    hovertemplate="%{x}<br>Outlier Loss: %{y:.2f}<extra></extra>",
                )
            )

            fig_bar.update_layout(
                height=450,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                barmode="group",
                xaxis=dict(
                    title=dict(text="Classification Surrogate Loss", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Loss Value (Penalty)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_bar, width="stretch", key="lfa_fig_bar")
