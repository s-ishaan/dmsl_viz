"""
Tile 3: Cost-Complexity Pruning & 1-SE Validation Rule
Slides: 271-278
Focus: Exploring weakest-link bottom-up pruning via cost-complexity:
C_α(T) = (1/N) ∑ (y_i - T(x_i))² + α |T|
Replicating the 10-fold cross-validation error curve of Slide 278,
and comparing the "1-Standard-Error ('se')" rule vs the "Minimum CV Error ('min')" rule.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from utils.datasets import get_dataset
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
            st.markdown("#### 🎛️ Pruning Knobs")

            selection_rule = st.radio(
                "Pruning Selection Rule",
                ["One-Standard-Error Rule ('se')", "Smallest CV Error Rule ('min')", "Manual α Slider"],
                index=0,
                key="ccp_rule",
                help="Rule for choosing best tree size (Slide 275 & 278). 'se' picks the simplest tree within 1 SE of minimum CV error.",
            )

            dataset_type = st.selectbox(
                "Dataset Topology",
                ["Moons", "Concentric Circles", "XOR / Checkerboard", "Overlapping Blobs"],
                index=0,
                key="ccp_dataset",
            )

            max_depth = st.slider(
                "Initial Unpruned Depth",
                min_value=4,
                max_value=10,
                value=7,
                step=1,
                key="ccp_init_depth",
                help="Depth of the large overfitted initial tree T_max before pruning.",
            )

            with st.expander("⚙️ Dataset & Cross-Validation", expanded=False):
                n_samples = st.slider("Sample Count N", 60, 200, 120, 10, key="ccp_samples")
                noise = st.slider("Noise Level", 0.05, 0.40, 0.22, 0.02, key="ccp_noise")
                n_folds = st.slider("CV Folds K", 5, 10, 10, 5, key="ccp_folds")

    # Generate noisy dataset prone to overfitting
    X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)

    # 1. Fit large unpruned tree and extract cost-complexity path
    full_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=2, random_state=42)
    path = full_tree.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Subsample path if too dense for speed
    if len(ccp_alphas) > 25:
        idx_sub = np.linspace(0, len(ccp_alphas) - 1, 25, dtype=int)
        ccp_alphas = ccp_alphas[idx_sub]

    # 2. Run K-fold Cross Validation across the sequence of alphas
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_errors_per_fold = np.zeros((len(ccp_alphas), n_folds))
    train_errors = np.zeros(len(ccp_alphas))
    tree_sizes = []

    for i, alpha in enumerate(ccp_alphas):
        # Fit on full data to get tree size
        t_model = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=alpha, random_state=42)
        t_model.fit(X, y)
        tree_sizes.append(t_model.get_n_leaves())
        train_errors[i] = np.mean(t_model.predict(X) != y)

        # Cross-validation
        for f_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            fold_clf = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=alpha, random_state=42)
            fold_clf.fit(X[train_idx], y[train_idx])
            pred_val = fold_clf.predict(X[val_idx])
            cv_errors_per_fold[i, f_idx] = np.mean(pred_val != y[val_idx])

    mean_cv_errors = np.mean(cv_errors_per_fold, axis=1)
    std_cv_errors = np.std(cv_errors_per_fold, axis=1) / np.sqrt(n_folds)
    tree_sizes = np.array(tree_sizes)

    # 3. Determine selected tree based on chosen rule
    min_idx = int(np.argmin(mean_cv_errors))
    min_cv_err = mean_cv_errors[min_idx]
    threshold_1se = min_cv_err + std_cv_errors[min_idx]

    if selection_rule.startswith("One-Standard-Error"):
        # Most parsimonious tree (smallest size / highest alpha) within 1 SE of minimum error
        candidates = np.where(mean_cv_errors <= threshold_1se)[0]
        # Smallest tree size among candidates
        best_idx = candidates[np.argmin(tree_sizes[candidates])]
    elif selection_rule.startswith("Smallest CV"):
        best_idx = min_idx
    else:
        # Manual selection slider
        with col_ctrl:
            with st.container(border=True):
                manual_idx = st.slider("Select α Subtree Step", 0, len(ccp_alphas) - 1, min_idx, 1, key="ccp_manual_idx")
                best_idx = manual_idx

    best_alpha = float(ccp_alphas[best_idx])
    best_size = int(tree_sizes[best_idx])

    # Fit final pruned model
    pruned_tree = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=best_alpha, random_state=42)
    pruned_tree.fit(X, y)
    pruned_acc = np.mean(pruned_tree.predict(X) == y) * 100.0

    full_tree.fit(X, y)
    full_acc = np.mean(full_tree.predict(X) == y) * 100.0
    full_size = full_tree.get_n_leaves()

    # Precompute 2D grid for boundary plot
    x_min, x_max = float(np.min(X[:, 0]) - 0.5), float(np.max(X[:, 0]) + 0.5)
    y_min, y_max = float(np.min(X[:, 1]) - 0.5), float(np.max(X[:, 1]) + 0.5)
    gx = np.linspace(x_min, x_max, 130)
    gy = np.linspace(y_min, y_max, 130)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]
    ZZ_pruned = pruned_tree.predict(grid_pts).reshape(XX.shape)

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Selected Size |T_α|", f"{best_size} leaves", delta=f"{best_size - full_size} pruned")
            m2.metric("Pruning Penalty α", f"{best_alpha:.4f}")
            m3.metric("Pruned CV Error", f"{mean_cv_errors[best_idx] * 100.0:.1f}%")
            m4.metric("Rule Applied", "1-SE Rule" if "One-Standard" in selection_rule else ("Min CV Error" if "Smallest" in selection_rule else "Manual"))

        tab_cv, tab_boundary, tab_compare = st.tabs([
            "📉 Cost-Complexity CV Error Curve (Slide 278 Replica)",
            "🗺️ Pruned 2D Decision Boundary (Size |T_α|)",
            "⚖️ Full Overfitted Tree vs Pruned Subtree",
        ])

        with tab_cv:
            st.markdown("##### 📊 10-Fold Cross-Validation vs Tree Size |T_α| & Penalty α (Slide 278)")

            fig_cv = go.Figure()

            # 1. 1-SE Upper Confidence Band (Shaded)
            fig_cv.add_trace(
                go.Scatter(
                    x=np.concatenate([tree_sizes, tree_sizes[::-1]]),
                    y=np.concatenate([mean_cv_errors + std_cv_errors, (mean_cv_errors - std_cv_errors)[::-1]]),
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="±1 SE Confidence Band",
                    hoverinfo="skip",
                )
            )

            # 2. 10-fold CV Error (Blue curve with points & error bars)
            fig_cv.add_trace(
                go.Scatter(
                    x=tree_sizes,
                    y=mean_cv_errors,
                    mode="lines+markers",
                    name="CV Error (Blue - Slide 278)",
                    line=dict(color="#38BDF8", width=2.5),
                    marker=dict(size=6, color="#38BDF8"),
                    error_y=dict(type="data", array=std_cv_errors, visible=True, color="rgba(56, 189, 248, 0.5)"),
                    customdata=ccp_alphas,
                    hovertemplate="Tree Size: %{x}<br>CV Error: %{y:.3f}<br>Penalty α: %{customdata:.4f}<extra></extra>",
                )
            )

            # 3. Training Error (Orange curve)
            fig_cv.add_trace(
                go.Scatter(
                    x=tree_sizes,
                    y=train_errors,
                    mode="lines+markers",
                    name="Training Error (Orange - Slide 278)",
                    line=dict(color="#F59E0B", width=2.2, dash="dash"),
                    marker=dict(size=5, color="#F59E0B"),
                    hovertemplate="Tree Size: %{x}<br>Train Error: %{y:.3f}<extra></extra>",
                )
            )

            # 4. 1-SE Threshold Horizontal Line
            fig_cv.add_hline(
                y=threshold_1se,
                line_width=1.5,
                line_dash="dot",
                line_color="#94A3B8",
                annotation_text=f"1-SE Cutoff ({threshold_1se:.3f})",
                annotation_position="top left",
                annotation_font=dict(color="#94A3B8", size=10),
            )

            # 5. Selected tree size vertical line (matching dashed line at 18 in Slide 278)
            fig_cv.add_vline(
                x=best_size,
                line_width=2.5,
                line_dash="dash",
                line_color=COLOR_HYPERPLANE,
                annotation_text=f"Optimal Size = {best_size}",
                annotation_position="bottom right",
                annotation_font=dict(color=COLOR_HYPERPLANE, size=11),
            )

            fig_cv.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Subtree Complexity (Number of Terminal Leaves |T_α|)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Misclassification Rate", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_cv, width="stretch", key="ccp_fig_cv")

        with tab_boundary:
            st.markdown(f"##### 🎯 Pruned Subtree Decision Boundary (|T_α| = {best_size} Leaves, α = {best_alpha:.4f})")

            fig_bound = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Shaded regions
            fig_bound.add_trace(
                go.Heatmap(
                    x=gx,
                    y=gy,
                    z=ZZ_pruned,
                    colorscale=[[0, "rgba(244, 63, 94, 0.35)"], [1, "rgba(56, 189, 248, 0.35)"]],
                    showscale=False,
                    hoverinfo="skip",
                )
            )

            # Decision Boundary contour
            fig_bound.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=ZZ_pruned,
                    contours_coloring="none",
                    contours=dict(start=0, end=0, size=1, coloring="none"),
                    line=dict(color="#FFFFFF", width=2.5),
                    name=f"Pruned Boundary (Leaves={best_size})",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # Scatter points
            fig_bound = add_scatter_points(fig_bound, X, y, marker_size=10)

            st.plotly_chart(fig_bound, width="stretch", key="ccp_fig_bound")

        with tab_compare:
            st.markdown("##### ⚖️ Overfitting vs Regularization Comparison")

            c_comp1, c_comp2 = st.columns(2)
            with c_comp1:
                with st.container(border=True):
                    st.markdown("###### 🌲 Unpruned Initial Tree T_max")
                    st.write(f"- **Terminal Leaves**: `{full_size}`")
                    st.write(f"- **Max Depth**: `{full_tree.get_depth()}`")
                    st.write(f"- **Training Accuracy**: `{full_acc:.1f}%`")
                    st.write(f"- **Estimated CV Error**: `{mean_cv_errors[0] * 100.0:.1f}%`")
                    st.caption("High variance: captures idiosyncratic sample noise.")

            with c_comp2:
                with st.container(border=True):
                    st.markdown("###### ✂️ Cost-Complexity Pruned Tree T_α")
                    st.write(f"- **Terminal Leaves**: `{best_size}`")
                    st.write(f"- **Max Depth**: `{pruned_tree.get_depth()}`")
                    st.write(f"- **Training Accuracy**: `{pruned_acc:.1f}%`")
                    st.write(f"- **Estimated CV Error**: `{mean_cv_errors[best_idx] * 100.0:.1f}%`")
                    st.caption("Balanced bias-variance trade-off via 10-fold cross-validation.")
