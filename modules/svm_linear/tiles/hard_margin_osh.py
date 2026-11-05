"""
Tile 3: Optimal Separating Hyperplane (Hard Margin SVM)
Slides: 215-218, 224
Focus: Interactive exploration of the maximum margin M = 1/||β||, canonical boundaries f(x) = ±1,
and the invariance of the OSH to non-support vectors.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    add_linear_boundary_and_margins,
    highlight_support_vectors,
    COLOR_HYPERPLANE,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 📏 Margin Controls")
            separation = st.slider(
                "Cluster Separation",
                min_value=1.8,
                max_value=5.0,
                value=3.2,
                step=0.2,
                key="hm_sep",
                help="Distance between the two class centers",
            )

            st.markdown("##### 🧪 Sparsity Experiment (Slide 224)")
            perturb_non_sv = st.slider(
                "Perturb Non-SVs (Δx)",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.2,
                key="hm_perturb",
                help="Shift non-support points. Notice the OSH, margin M, and weights β remain 100% unchanged!",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                cluster_noise = st.slider(
                    "Cluster Spread (Noise)",
                    min_value=0.05,
                    max_value=0.40,
                    value=0.15,
                    step=0.05,
                    key="hm_noise",
                )
                n_samples = st.slider("Sample Count N", 20, 100, 40, step=10, key="hm_n")
                show_margin_band = st.checkbox("Shade Margin Corridor (2M)", value=True, key="hm_shade")

    # Generate base separable data
    X_base, y = get_dataset(
        "Linearly Separable Blobs",
        n_samples=n_samples,
        noise=cluster_noise,
        separation=separation,
        random_state=42,
    )

    # First pass to identify initial support vectors
    clf_init = SVC(kernel="linear", C=1000.0)
    clf_init.fit(X_base, y)
    initial_sv_indices = set(clf_init.support_)

    # Apply perturbation ONLY to non-support vectors
    X = X_base.copy()
    if abs(perturb_non_sv) > 1e-4:
        for i in range(len(X)):
            if i not in initial_sv_indices:
                direction = 1.0 if y[i] == 1 else -1.0
                X[i, 0] += perturb_non_sv * direction
                X[i, 1] += perturb_non_sv * direction

    # Fit final hard-margin SVM
    clf = SVC(kernel="linear", C=1000.0)
    clf.fit(X, y)

    w = clf.coef_[0]
    b = clf.intercept_[0]
    norm_w = np.linalg.norm(w)
    margin_M = 1.0 / (norm_w if norm_w > 1e-6 else 1e-6)
    margin_width = 2.0 * margin_M

    sv_indices = clf.support_
    X_sv = clf.support_vectors_
    y_sv = y[sv_indices]
    n_sv = len(sv_indices)
    sv_ratio = (n_sv / len(y)) * 100.0

    # Functional margins for SVs
    func_margins = y_sv * (np.dot(X_sv, w) + b)
    is_separable = np.all(y * (np.dot(X, w) + b) >= 0.99)

    with col_plot:
        # Metrics container
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Half-Margin M", f"{margin_M:.3f}")
            m2.metric("Margin Width 2M", f"{margin_width:.3f}")
            m3.metric("Norm ||β||", f"{norm_w:.3f}")
            m4.metric("Support Vectors", f"{n_sv} / {len(y)} ({sv_ratio:.0f}%)")

        if is_separable:
            st.success("✅ **Dataset Status: Strictly Linearly Separable** (Hard-Margin OSH formulation is exact).")
        else:
            st.warning("⚠️ **Dataset Status: Non-Separable Overlap** (Hard margin is violated; slack variables required).")

        st.markdown("##### 📏 Maximum Margin Separator: M = 1 / ||β||")
        x_min, x_max = float(np.min(X[:, 0]) - 1.2), float(np.max(X[:, 0]) + 1.2)
        y_min, y_max = float(np.min(X[:, 1]) - 1.2), float(np.max(X[:, 1]) + 1.2)

        fig = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=440)

        # Scatter points
        fig = add_scatter_points(fig, X, y, marker_size=10)

        # Decision boundary and canonical margins
        fig = add_linear_boundary_and_margins(
            fig,
            w,
            b,
            (x_min, x_max),
            (y_min, y_max),
            show_margins=True,
            show_margin_band=show_margin_band,
            boundary_color=COLOR_HYPERPLANE,
            boundary_name="OSH: β₀ + βᵀx = 0",
            line_width=3,
        )

        # Highlight support vectors
        labels = [f"Class {y_sv[k]}, y(f(x))={func_margins[k]:.3f}" for k in range(n_sv)]
        fig = highlight_support_vectors(
            fig,
            X_sv,
            y_sv=y_sv,
            labels=labels,
            color="#EAB308",
            name="Support Vectors (αᵢ > 0)",
        )

        st.plotly_chart(fig, width="stretch")
