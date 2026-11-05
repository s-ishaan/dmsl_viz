"""
Tile 2: Perceptron vs. Least Squares vs. Maximum Margin SVM
Slides: 210, 211, 213
Focus: Visualizing the masking problem of Least Squares for classification, the iterative Perceptron,
and the robustness of SVM's optimal margin.
"""

from typing import Tuple
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    add_linear_boundary_and_margins,
    COLOR_HYPERPLANE,
)


def fit_least_squares(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit OLS regression on -1/+1 labels: y ~ X w + b."""
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    w_full, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    w = w_full[:2]
    b = w_full[2]
    return w, b


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎯 Benchmark Setup")
            dataset_name = st.selectbox(
                "Benchmark Scenario",
                ["Outlier Challenge (Masking)", "Linearly Separable Blobs", "Overlapping Blobs"],
                index=0,
                key="pvs_dataset",
                help="Outlier Challenge recreates Slide 213: a far point on the correct side breaks Least Squares!",
            )

            outlier_dist = st.slider(
                "Outlier Leverage Distance",
                min_value=2.5,
                max_value=9.0,
                value=5.5,
                step=0.5,
                key="pvs_outlier_dist",
                help="Distance of the extreme point belonging to Class +1",
            )

            st.markdown("##### 🔲 Active Models")
            show_ls = st.checkbox("Least Squares (OLS on ±1)", value=True, key="pvs_show_ls")
            show_perc = st.checkbox("Rosenblatt's Perceptron", value=True, key="pvs_show_perc")
            show_svm = st.checkbox("SVM Maximum Margin", value=True, key="pvs_show_svm")

            with st.expander("⚙️ Perceptron & Tuning Knobs", expanded=False):
                perc_max_iter = st.slider("Perceptron Epochs", 1, 50, 15, step=1, key="pvs_perc_iter")
                perc_lr = st.slider("Learning Rate η", 0.01, 1.0, 0.1, step=0.05, key="pvs_perc_lr")
                n_samples = st.slider("Sample Count N", 20, 80, 40, step=10, key="pvs_n_samples")

    # Generate data
    X, y = get_dataset(
        dataset_name,
        n_samples=n_samples,
        noise=0.15,
        separation=2.8,
        outlier_distance=outlier_dist,
        random_state=42,
    )

    # 1. Fit Least Squares
    w_ls, b_ls = fit_least_squares(X, y)
    y_pred_ls = np.sign(np.dot(X, w_ls) + b_ls)
    acc_ls = np.mean(y_pred_ls == y) * 100
    errors_ls = int(np.sum(y_pred_ls != y))

    # 2. Fit Perceptron
    clf_perc = Perceptron(max_iter=perc_max_iter, eta0=perc_lr, random_state=42, tol=1e-3)
    clf_perc.fit(X, y)
    w_perc = clf_perc.coef_[0]
    b_perc = clf_perc.intercept_[0]
    y_pred_perc = clf_perc.predict(X)
    acc_perc = np.mean(y_pred_perc == y) * 100
    errors_perc = int(np.sum(y_pred_perc != y))

    # 3. Fit Linear SVM
    clf_svm = SVC(kernel="linear", C=100.0)
    clf_svm.fit(X, y)
    w_svm = clf_svm.coef_[0]
    b_svm = clf_svm.intercept_[0]
    y_pred_svm = clf_svm.predict(X)
    acc_svm = np.mean(y_pred_svm == y) * 100
    errors_svm = int(np.sum(y_pred_svm != y))

    with col_plot:
        # Metrics container
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Least Squares Errors", f"{errors_ls} ({acc_ls:.0f}% acc)", delta=f"-{errors_ls}" if errors_ls > 0 else "0", delta_color="inverse")
            m2.metric("Perceptron Errors", f"{errors_perc} ({acc_perc:.0f}% acc)", delta=f"-{errors_perc}" if errors_perc > 0 else "0", delta_color="inverse")
            m3.metric("SVM Margin Errors", f"{errors_svm} ({acc_svm:.0f}% acc)", delta=f"-{errors_svm}" if errors_svm > 0 else "0", delta_color="inverse")

        # Masking visual alert if relevant
        if dataset_name == "Outlier Challenge (Masking)" and errors_ls > 0:
            st.info(
                f"💡 **Masking Effect in Action (Slide 213)**: The far outlier pulls the OLS plane, causing **{errors_ls} misclassification(s)** "
                "on points that are easily separated by the SVM optimal margin.",
                icon="⚠️",
            )

        st.markdown("##### ⚔️ Classifier Decision Boundary Comparison")
        x_min, x_max = float(np.min(X[:, 0]) - 1.0), float(np.max(X[:, 0]) + 1.0)
        y_min, y_max = float(np.min(X[:, 1]) - 1.0), float(np.max(X[:, 1]) + 1.0)
        fig = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=440)

        # Scatter points
        fig = add_scatter_points(fig, X, y, marker_size=11)

        # Draw boundaries
        if show_ls:
            fig = add_linear_boundary_and_margins(
                fig,
                w_ls,
                b_ls,
                (x_min, x_max),
                (y_min, y_max),
                show_margins=False,
                boundary_color="#F97316",  # Bright Orange (Slide 213)
                boundary_name="Least Squares (OLS)",
                line_width=3,
            )

        if show_perc:
            fig = add_linear_boundary_and_margins(
                fig,
                w_perc,
                b_perc,
                (x_min, x_max),
                (y_min, y_max),
                show_margins=False,
                boundary_color="#A855F7",  # Purple
                boundary_name="Perceptron",
                line_width=2.5,
            )

        if show_svm:
            fig = add_linear_boundary_and_margins(
                fig,
                w_svm,
                b_svm,
                (x_min, x_max),
                (y_min, y_max),
                show_margins=True,
                show_margin_band=True,
                boundary_color=COLOR_HYPERPLANE,
                boundary_name="SVM (Optimal Margin)",
                line_width=3,
            )

        st.plotly_chart(fig, width="stretch")
