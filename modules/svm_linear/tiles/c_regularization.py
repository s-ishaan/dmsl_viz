"""
Tile 5: Cost Parameter C & Regularization Tradeoff
Slides: 221-226
Focus: Direct replication and interactive exploration of Slide 226 (C=10000 vs C=0.01),
margin width expansion, support vector percentage, and the bias-variance tradeoff.
"""

import warnings
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    add_linear_boundary_and_margins,
    COLOR_HYPERPLANE,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_and_eval_svm(X_train, y_train, X_test, y_test, C_val):
    clf = SVC(kernel="linear", C=C_val, max_iter=2000)
    clf.fit(X_train, y_train)

    w = clf.coef_[0]
    b = clf.intercept_[0]
    norm_w = np.linalg.norm(w)
    margin_M = 1.0 / (norm_w if norm_w > 1e-6 else 1e-6)
    margin_width = 2.0 * margin_M

    alpha_sv = np.abs(clf.dual_coef_[0])
    sv_indices = clf.support_
    X_sv = clf.support_vectors_
    n_sv = len(sv_indices)
    sv_pct = (n_sv / len(y_train)) * 100.0

    tol = 1e-4 * C_val
    is_bounded = (alpha_sv >= (C_val - tol))
    n_bounded = int(np.sum(is_bounded))
    n_margin_sv = n_sv - n_bounded

    train_acc = clf.score(X_train, y_train) * 100
    test_acc = clf.score(X_test, y_test) * 100
    train_err = (100.0 - train_acc) / 100.0
    test_err = (100.0 - test_acc) / 100.0

    return {
        "clf": clf, "w": w, "b": b, "margin_width": margin_width,
        "n_sv": n_sv, "sv_pct": sv_pct, "X_sv": X_sv,
        "is_bounded": is_bounded, "n_margin_sv": n_margin_sv, "n_bounded": n_bounded,
        "train_err": train_err, "test_err": test_err,
    }


def render():
    col_ctrl, col_main = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎚️ Regularization C")
            dataset_name = st.selectbox(
                "Dataset Pattern",
                ["Overlapping Blobs", "Moons", "Linearly Separable Blobs"],
                index=0,
                key="cr_dataset",
            )

            log_c = st.slider(
                "Continuous log₁₀(C)",
                min_value=-3.0,
                max_value=4.0,
                value=0.0,
                step=0.25,
                key="cr_log_c",
                help="C controls penalty on slack: large C -> narrow margin; small C -> wide margin.",
            )
            current_C = 10.0 ** log_c

            with st.expander("⚙️ Dataset & Split Settings", expanded=False):
                noise = st.slider("Dataset Noise", 0.1, 0.6, 0.3, step=0.05, key="cr_noise")
                n_pts = st.slider("Total Points N", 40, 200, 100, step=20, key="cr_pts")
                test_ratio = st.slider("Test Split Ratio", 0.1, 0.5, 0.3, step=0.05, key="cr_split")

    # Generate dataset & split
    X, y = get_dataset(dataset_name, n_samples=n_pts, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    x_min, x_max = float(np.min(X[:, 0]) - 1.2), float(np.max(X[:, 0]) + 1.2)
    y_min, y_max = float(np.min(X[:, 1]) - 1.2), float(np.max(X[:, 1]) + 1.2)

    with col_main:
        tab_dual, tab_sweep = st.tabs([
            "🪞 Slide 226 Dual Comparison (C=10000 vs C=0.01)",
            "🎛️ Continuous Sweep & Response Curves"
        ])

        # TAB 1: Slide 226 Direct Dual Comparison
        with tab_dual:
            res_high = fit_and_eval_svm(X_train, y_train, X_test, y_test, 10000.0)
            res_low = fit_and_eval_svm(X_train, y_train, X_test, y_test, 0.01)

            subtab_high, subtab_low, subtab_both = st.tabs([
                "⬅️ Left Panel: C = 10000 (Narrow Margin)",
                "➡️ Right Panel: C = 0.01 (Wide Margin)",
                "📑 Both Panels Stacked (Full Width)",
            ])

            with subtab_high:
                with st.container(border=True):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Regime", "Hard Margin (C=10000)")
                    m2.metric("Margin Width 2M", f"{res_high['margin_width']:.3f}")
                    m3.metric("Support Points %", f"{res_high['sv_pct']:.0f}% ({res_high['n_sv']}/{len(y_train)})")
                    m4.metric("Test Error", f"{res_high['test_err']:.3f}")

                st.markdown("##### 🪞 Slide 226 Left Panel: Large C = 10000 (Narrow Margin, Low SV Count)")
                fig_h = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=450)
                fig_h = add_scatter_points(fig_h, X_train, y_train, marker_size=9, show_legend=True)
                fig_h = add_linear_boundary_and_margins(
                    fig_h, res_high["w"], res_high["b"], (x_min, x_max), (y_min, y_max),
                    show_margins=True, show_margin_band=True, line_width=2.5
                )
                fig_h.add_trace(go.Scatter(
                    x=res_high["X_sv"][:, 0], y=res_high["X_sv"][:, 1], mode="markers",
                    name="Support Vectors",
                    marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#EAB308", width=2.5)),
                    hoverinfo="skip"
                ))
                st.plotly_chart(fig_h, width="stretch", key="cr_fig_high")

            with subtab_low:
                with st.container(border=True):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Regime", "Soft Margin (C=0.01)")
                    m2.metric("Margin Width 2M", f"{res_low['margin_width']:.3f}")
                    m3.metric("Support Points %", f"{res_low['sv_pct']:.0f}% ({res_low['n_sv']}/{len(y_train)})")
                    m4.metric("Test Error", f"{res_low['test_err']:.3f}")

                st.markdown("##### 🪞 Slide 226 Right Panel: Small C = 0.01 (Wide Margin, High SV Count)")
                fig_l = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=450)
                fig_l = add_scatter_points(fig_l, X_train, y_train, marker_size=9, show_legend=True)
                fig_l = add_linear_boundary_and_margins(
                    fig_l, res_low["w"], res_low["b"], (x_min, x_max), (y_min, y_max),
                    show_margins=True, show_margin_band=True, line_width=2.5
                )
                fig_l.add_trace(go.Scatter(
                    x=res_low["X_sv"][:, 0], y=res_low["X_sv"][:, 1], mode="markers",
                    name="Support Vectors",
                    marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#EAB308", width=2.5)),
                    hoverinfo="skip"
                ))
                st.plotly_chart(fig_l, width="stretch", key="cr_fig_low")

            with subtab_both:
                st.markdown("##### ⬅️ Left Panel: C = 10000 (Margin Width: {:.3f}, SV: {:.0f}%)".format(res_high['margin_width'], res_high['sv_pct']))
                st.plotly_chart(fig_h, width="stretch", key="cr_fig_high_both")
                st.markdown("---")
                st.markdown("##### ➡️ Right Panel: C = 0.01 (Margin Width: {:.3f}, SV: {:.0f}%)".format(res_low['margin_width'], res_low['sv_pct']))
                st.plotly_chart(fig_l, width="stretch", key="cr_fig_low_both")

        # TAB 2: Continuous sweep with responsive curves
        with tab_sweep:
            res_curr = fit_and_eval_svm(X_train, y_train, X_test, y_test, current_C)

            with st.container(border=True):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Current C", f"{current_C:.4g}")
                m2.metric("Margin Width (2M)", f"{res_curr['margin_width']:.3f}")
                m3.metric("Support Points %", f"{res_curr['sv_pct']:.1f}%")
                m4.metric("Train Error", f"{res_curr['train_err']:.3f}")
                m5.metric("Test Error", f"{res_curr['test_err']:.3f}")

            st.markdown(f"##### 🎛️ Continuous Decision Boundary (C = {current_C:.4g})")
            fig_sweep = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=420)
            fig_sweep = add_scatter_points(fig_sweep, X_train, y_train, marker_size=9, name_prefix="Train ")
            fig_sweep = add_linear_boundary_and_margins(
                fig_sweep, res_curr["w"], res_curr["b"], (x_min, x_max), (y_min, y_max),
                show_margins=True, show_margin_band=True, boundary_color=COLOR_HYPERPLANE, line_width=3
            )

            # Highlight Free vs Bounded SVs
            if res_curr["n_margin_sv"] > 0:
                X_free = res_curr["X_sv"][~res_curr["is_bounded"]]
                fig_sweep.add_trace(go.Scatter(
                    x=X_free[:, 0], y=X_free[:, 1], mode="markers",
                    name=f"Margin SV (0<α<C): {res_curr['n_margin_sv']}",
                    marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#EAB308", width=2.5)),
                    hoverinfo="skip",
                ))

            if res_curr["n_bounded"] > 0:
                X_bound = res_curr["X_sv"][res_curr["is_bounded"]]
                fig_sweep.add_trace(go.Scatter(
                    x=X_bound[:, 0], y=X_bound[:, 1], mode="markers",
                    name=f"Bounded SV (α=C, ξ>0): {res_curr['n_bounded']}",
                    marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#F43F5E", width=2.5, dash="dot")),
                    hoverinfo="skip",
                ))

            st.plotly_chart(fig_sweep, width="stretch", key="cr_fig_sweep")

            # Response curves
            c_grid = np.logspace(-3, 3, 14)
            widths, sv_pcts, test_errs = [], [], []

            for c_temp in c_grid:
                m_temp = SVC(kernel="linear", C=c_temp, max_iter=1000, tol=1e-2)
                m_temp.fit(X_train, y_train)
                w_temp = m_temp.coef_[0]
                norm_w_temp = np.linalg.norm(w_temp)
                widths.append(2.0 / (norm_w_temp if norm_w_temp > 1e-6 else 1e-6))
                sv_pcts.append((len(m_temp.support_) / len(y_train)) * 100)
                test_errs.append(1.0 - m_temp.score(X_test, y_test))

            fig_curves = go.Figure()
            fig_curves.add_trace(go.Scatter(
                x=c_grid, y=widths, mode="lines+markers", name="Margin Width (2M)",
                line=dict(color="#F59E0B", width=2), yaxis="y1"
            ))
            fig_curves.add_trace(go.Scatter(
                x=c_grid, y=sv_pcts, mode="lines+markers", name="Support Vector %",
                line=dict(color="#38BDF8", width=2), yaxis="y2"
            ))
            fig_curves.add_trace(go.Scatter(
                x=c_grid, y=[e * 100 for e in test_errs], mode="lines", name="Test Error %",
                line=dict(color="#F43F5E", width=2, dash="dash"), yaxis="y2"
            ))
            fig_curves.add_vline(
                x=current_C, line_dash="dot", line_color="#10B981",
                annotation_text=f"Current C={current_C:.2g}", annotation_position="top right",
                annotation_font=dict(color="#10B981", size=10),
            )
            st.markdown("##### 📈 Response Curves: Margin Width, Sparsity, and Error Rate")
            fig_curves.update_layout(
                height=260,
                margin=dict(l=45, r=45, t=15, b=55),
                paper_bgcolor="#1E293B",
                plot_bgcolor="#0F172A",
                xaxis=dict(
                    title=dict(text="Regularization C (Log Scale)", font=dict(color="#CBD5E1", size=11)),
                    type="log",
                    tickfont=dict(color="#94A3B8"),
                    gridcolor="#334155",
                ),
                yaxis=dict(
                    title=dict(text="Margin Width (2M)", font=dict(color="#F59E0B", size=11)),
                    tickfont=dict(color="#F59E0B"),
                    gridcolor="#334155",
                ),
                yaxis2=dict(
                    title=dict(text="SV % / Error %", font=dict(color="#38BDF8", size=11)),
                    tickfont=dict(color="#38BDF8"),
                    overlaying="y",
                    side="right",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                    font=dict(color="#CBD5E1", size=10),
                ),
            )
            st.plotly_chart(fig_curves, width="stretch", key="cr_fig_curves")
