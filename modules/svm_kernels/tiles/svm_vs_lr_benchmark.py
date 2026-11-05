"""
Tile 6: Kernel SVM vs. Kernel Logistic Regression (Slides 258-259)
Slides: 258-259
Focus: Direct replication of the lecture & ESL benchmark comparing:
1. Degree-4 Polynomial Kernel (Slide 258)
2. Radial Basis Function Kernel γ=1 (Slide 259)
Highlighting the fundamental structural contrast:
- SVM uses Hinge Loss -> Sparse dual representation (only SVs have αᵢ > 0).
- Kernel Logistic Regression uses Binomial Deviance -> Dense representation (all points have αᵢ > 0).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    add_contour_boundary_and_margins,
    highlight_support_vectors,
    BG_PAPER,
    BG_PLOT,
    COLOR_HYPERPLANE,
    COLOR_CLASS_NEG,
    COLOR_CLASS_POS,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Benchmark Configuration")

            preset = st.radio(
                "Benchmark Preset",
                [
                    "Degree-4 Polynomial (C = 1.0)",
                    "Radial Basis Kernel γ = 1.0 (C = 1.0)",
                    "Custom Hyperparameters",
                ],
                index=0,
                key="blr_preset",
                help="Pre-configured models benchmarking polynomial and RBF kernels.",
            )

            if preset.startswith("Degree-4"):
                kernel_choice = "poly"
                degree_val = 4
                gamma_val = "scale"
                c_val = 1.0
            elif preset.startswith("Radial Basis"):
                kernel_choice = "rbf"
                degree_val = 3
                gamma_val = 1.0
                c_val = 1.0
            else:
                kernel_choice = st.selectbox("Kernel Type", ["rbf", "poly"], index=0, key="blr_custom_k")
                if kernel_choice == "poly":
                    degree_val = st.slider("Degree d", 1, 5, 4, 1, key="blr_deg")
                    gamma_val = "scale"
                else:
                    gamma_val = st.slider("Bandwidth γ", 0.1, 5.0, 1.0, 0.1, key="blr_gam")
                    degree_val = 3
                c_val = st.slider("Penalty C", 0.1, 10.0, 1.0, 0.5, key="blr_c")

            dataset_type = st.selectbox(
                "Dataset Topology",
                ["Overlapping Blobs", "Moons", "Concentric Circles"],
                index=0,
                key="blr_dataset",
                help="Simulation data. Overlapping Blobs mimics ESL mixture simulation.",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Points N", 40, 140, 80, 10, key="blr_samples")
                noise = st.slider("Noise Level", 0.05, 0.40, 0.15, 0.05, key="blr_noise")
                show_surface = st.checkbox("Show Decision Heatmap", value=True, key="blr_surface")

    # Generate dataset
    X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)

    # 1. Fit Kernel SVM
    if kernel_choice == "poly":
        svm_model = SVC(kernel="poly", degree=degree_val, coef0=1.0, gamma=gamma_val, C=c_val)
    else:
        svm_model = SVC(kernel="rbf", gamma=gamma_val, C=c_val)
    svm_model.fit(X, y)

    svm_pred = svm_model.predict(X)
    svm_acc = np.mean(svm_pred == y) * 100.0
    svm_n_sv = len(svm_model.support_)
    svm_sv_pct = (svm_n_sv / len(y)) * 100.0

    # 2. Fit Kernel Logistic Regression via Nystroem kernel embedding
    # (Matches exact kernel evaluation while optimizing binomial deviance loss)
    if kernel_choice == "poly":
        nystroem = Nystroem(kernel="poly", degree=degree_val, coef0=1.0, n_components=min(n_samples, 60), random_state=42)
    else:
        nystroem = Nystroem(kernel="rbf", gamma=gamma_val if isinstance(gamma_val, float) else 1.0, n_components=min(n_samples, 60), random_state=42)

    klr_pipeline = make_pipeline(nystroem, LogisticRegression(C=c_val, max_iter=1000))
    klr_pipeline.fit(X, y)
    klr_pred = klr_pipeline.predict(X)
    klr_acc = np.mean(klr_pred == y) * 100.0

    # KLR decision function maps log-odds
    def klr_decision_func(pts):
        return klr_pipeline.decision_function(pts)

    # Dense weights: approximate point-wise influence in KLR via absolute log-odds derivative
    klr_probs = klr_pipeline.predict_proba(X)[:, 1]
    # In logistic regression, point weight is p_i * (1 - p_i)
    klr_point_weights = klr_probs * (1.0 - klr_probs)

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SVM Sparsity (SVs)", f"{svm_n_sv} / {len(y)} ({svm_sv_pct:.0f}%)")
            m2.metric("KLR Sparsity", f"{len(y)} / {len(y)} (0% Sparse)", help="Binomial deviance has non-zero gradient everywhere; all training points receive weights (Slide 257).")
            m3.metric("SVM Train Acc", f"{svm_acc:.1f}%")
            m4.metric("KLR Train Acc", f"{klr_acc:.1f}%")

        tab_svm, tab_klr, tab_weights = st.tabs([
            "🛡️ Kernel SVM (Sparse Hinge Loss)",
            "📈 Kernel Logistic Regression (Dense Binomial Deviance)",
            "⚖️ Dual Weights: Sparse vs Dense Distribution",
        ])

        x_min, x_max = float(np.min(X[:, 0]) - 1.2), float(np.max(X[:, 0]) + 1.2)
        y_min, y_max = float(np.min(X[:, 1]) - 1.2), float(np.max(X[:, 1]) + 1.2)

        with tab_svm:
            st.markdown(f"##### 🛡️ Support Vector Machine ({preset.split(':')[0]})")
            fig_svm = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Scatter points
            fig_svm = add_scatter_points(fig_svm, X, y, marker_size=10)

            # SVM decision boundary & margins
            fig_svm = add_contour_boundary_and_margins(
                fig_svm,
                svm_model.decision_function,
                (x_min, x_max),
                (y_min, y_max),
                resolution=120,
                show_margins=True,
                show_surface=show_surface,
                boundary_name="SVM Boundary f(x)=0",
                line_width=3,
            )

            # Support vectors
            X_sv = svm_model.support_vectors_
            y_sv = y[svm_model.support_]
            dual_coefs = np.abs(svm_model.dual_coef_[0])
            labels = [f"Class {y_sv[i]} | Dual αᵢ={dual_coefs[i]:.3f}" for i in range(len(X_sv))]
            fig_svm = highlight_support_vectors(
                fig_svm,
                X_sv,
                y_sv=y_sv,
                labels=labels,
                color="#EAB308",
                name="SVM Support Vectors (αᵢ > 0)",
            )

            st.plotly_chart(fig_svm, width="stretch", key="blr_fig_svm")

        with tab_klr:
            st.markdown(f"##### 📈 Kernel Logistic Regression ({preset.split(':')[0]})")
            fig_klr = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Scatter points
            fig_klr = add_scatter_points(fig_klr, X, y, marker_size=10)

            # KLR decision boundary (log-odds = 0) and probability margins (+-1 log-odds = prob 0.73/0.27)
            fig_klr = add_contour_boundary_and_margins(
                fig_klr,
                klr_decision_func,
                (x_min, x_max),
                (y_min, y_max),
                resolution=120,
                show_margins=True,
                show_surface=show_surface,
                boundary_color="#38BDF8",
                boundary_name="KLR Boundary P=0.5",
                line_width=3,
            )

            # Highlight all training points with weight halos, showing EVERY point is non-zero
            fig_klr.add_trace(
                go.Scatter(
                    x=X[:, 0],
                    y=X[:, 1],
                    mode="markers",
                    name="KLR Active Points (100% Non-Zero)",
                    marker=dict(
                        size=14,
                        color="rgba(0,0,0,0)",
                        line=dict(color="#A855F7", width=1.8),
                    ),
                    customdata=klr_point_weights,
                    hovertemplate="Obs weight p(1-p): %{customdata:.3f}<extra></extra>",
                )
            )

            st.plotly_chart(fig_klr, width="stretch", key="blr_fig_klr")

        with tab_weights:
            st.markdown("##### 📶 Dual Coefficient Distribution: SVM Sparsity vs KLR Dense Influence")

            # Dual coefficients for SVM (0 for non-SVs)
            svm_alphas = np.zeros(len(y))
            svm_alphas[svm_model.support_] = np.abs(svm_model.dual_coef_[0])

            # Normalize for side-by-side comparison
            norm_svm_alphas = svm_alphas / (np.max(svm_alphas) if np.max(svm_alphas) > 0 else 1.0)
            norm_klr_weights = klr_point_weights / (np.max(klr_point_weights) if np.max(klr_point_weights) > 0 else 1.0)

            fig_bar = go.Figure()
            indices = np.arange(len(y))

            fig_bar.add_trace(
                go.Bar(
                    x=indices,
                    y=norm_svm_alphas,
                    name="SVM Dual αᵢ (Sparse)",
                    marker=dict(color="#EAB308"),
                    hovertemplate="Point #%{x}<br>SVM αᵢ: %{y:.3f}<extra></extra>",
                )
            )

            fig_bar.add_trace(
                go.Bar(
                    x=indices,
                    y=norm_klr_weights,
                    name="KLR Weight p(1-p) (Dense)",
                    marker=dict(color="#38BDF8"),
                    hovertemplate="Point #%{x}<br>KLR Weight: %{y:.3f}<extra></extra>",
                )
            )

            fig_bar.update_layout(
                height=450,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                barmode="group",
                xaxis=dict(
                    title=dict(text="Training Observation Index", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Relative Influence Weight", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_bar, width="stretch", key="blr_fig_bar")
