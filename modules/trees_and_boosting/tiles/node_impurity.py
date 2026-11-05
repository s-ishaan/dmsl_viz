"""
Tile 2: Classification Node Impurity & Purity Metrics
Slide: 270
Focus: Exploring the mathematical behavior of splitting criteria for classification trees:
- Gini Index: ∑ p_k (1 - p_k)
- Cross-Entropy / Deviance: -∑ p_k log₂(p_k)
- 0-1 Misclassification Error: 1 - max_k p_k
Visualizing why Gini & Entropy are strictly concave, sensitive to node purity, and preferred over 0-1 error.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
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
            st.markdown("#### 🎛️ Impurity & Tree Knobs")

            p_probe = st.slider(
                "Class +1 Proportion p",
                min_value=0.0,
                max_value=1.0,
                value=0.35,
                step=0.01,
                key="ni_p_probe",
                help="Proportion p = p̂_j1 of class +1 in a binary node (Slide 270).",
            )

            criterion = st.selectbox(
                "Tree Splitting Criterion",
                ["gini", "entropy"],
                index=0,
                key="ni_criterion",
                help="Loss function used to evaluate split quality in CART (MATLAB default is Gini).",
            )

            max_depth = st.slider(
                "Tree Max Depth",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
                key="ni_depth",
                help="Depth limit of the classification tree.",
            )

            dataset_type = st.selectbox(
                "Dataset Topology",
                ["Moons", "Concentric Circles", "XOR / Checkerboard", "Overlapping Blobs"],
                index=0,
                key="ni_dataset",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Points N", 40, 160, 90, 10, key="ni_samples")
                noise = st.slider("Noise Level", 0.05, 0.40, 0.15, 0.05, key="ni_noise")

    # Impurity calculations for binary node at p_probe
    p = p_probe
    p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)

    val_gini = 2.0 * p * (1.0 - p)
    val_entropy = - (p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))
    val_entropy_scaled = 0.5 * val_entropy  # Scaled to peak at 0.5 like Gini for comparison
    val_misclass = min(p, 1.0 - p)

    # Fit classification tree
    X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = np.mean(y_pred == y) * 100.0
    n_leaves = clf.get_n_leaves()

    # Precompute 2D grid
    x_min, x_max = float(np.min(X[:, 0]) - 0.5), float(np.max(X[:, 0]) + 0.5)
    y_min, y_max = float(np.min(X[:, 1]) - 0.5), float(np.max(X[:, 1]) + 0.5)
    gx = np.linspace(x_min, x_max, 130)
    gy = np.linspace(y_min, y_max, 130)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]
    probs = clf.predict_proba(grid_pts)
    # Purity is max class probability in the leaf
    purity_grid = np.max(probs, axis=1).reshape(XX.shape)
    decision_grid = clf.predict(grid_pts).reshape(XX.shape)

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Gini Impurity (p)", f"{val_gini:.3f}", help="2p(1-p). Peaks at 0.5 for equal mix, 0 for pure node.")
            m2.metric("Cross-Entropy (p)", f"{val_entropy:.3f}", help="Information entropy -∑ p log₂(p).")
            m3.metric("0-1 Misclass (p)", f"{val_misclass:.3f}")
            m4.metric("Train Accuracy", f"{acc:.1f}%")

        tab_curves, tab_purity, tab_leaves = st.tabs([
            "📈 Impurity Functions vs Proportion p",
            "🗺️ 2D Decision Space Shaded by Leaf Purity",
            "📊 Leaf-by-Leaf Class Purity & Distribution",
        ])

        with tab_curves:
            st.markdown("##### 📐 Node Impurity Comparison: Gini vs Scaled Entropy vs 0-1 Error (Slide 270)")

            p_axis = np.linspace(0.001, 0.999, 300)
            gini_curve = 2.0 * p_axis * (1.0 - p_axis)
            entropy_scaled_curve = 0.5 * (- (p_axis * np.log2(p_axis) + (1.0 - p_axis) * np.log2(1.0 - p_axis)))
            misclass_curve = np.minimum(p_axis, 1.0 - p_axis)

            fig_imp = go.Figure()

            # Gini Curve (Emerald)
            fig_imp.add_trace(
                go.Scatter(
                    x=p_axis,
                    y=gini_curve,
                    mode="lines",
                    name="Gini Index: 2p(1 - p)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    hovertemplate="p: %{x:.2f}<br>Gini: %{y:.3f}<extra></extra>",
                )
            )

            # Cross-Entropy (Scaled 1/2) (Sky Blue)
            fig_imp.add_trace(
                go.Scatter(
                    x=p_axis,
                    y=entropy_scaled_curve,
                    mode="lines",
                    name="Cross-Entropy (Scaled × 0.5)",
                    line=dict(color="#38BDF8", width=2.5, dash="dash"),
                    hovertemplate="p: %{x:.2f}<br>Scaled Entropy: %{y:.3f}<extra></extra>",
                )
            )

            # Misclassification Error (Amber)
            fig_imp.add_trace(
                go.Scatter(
                    x=p_axis,
                    y=misclass_curve,
                    mode="lines",
                    name="0-1 Misclassification: min(p, 1-p)",
                    line=dict(color="#F59E0B", width=2.5, dash="dot"),
                    hovertemplate="p: %{x:.2f}<br>Misclass: %{y:.3f}<extra></extra>",
                )
            )

            # Probe line at current p
            fig_imp.add_vline(x=p_probe, line_width=1.5, line_dash="dash", line_color="#FFFFFF")

            # Markers on the probe line
            fig_imp.add_trace(go.Scatter(x=[p_probe], y=[val_gini], mode="markers", marker=dict(size=9, color=COLOR_HYPERPLANE), showlegend=False, hoverinfo="skip"))
            fig_imp.add_trace(go.Scatter(x=[p_probe], y=[val_entropy_scaled], mode="markers", marker=dict(size=9, color="#38BDF8"), showlegend=False, hoverinfo="skip"))
            fig_imp.add_trace(go.Scatter(x=[p_probe], y=[val_misclass], mode="markers", marker=dict(size=9, color="#F59E0B"), showlegend=False, hoverinfo="skip"))

            fig_imp.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Node Class +1 Proportion p̂_j1", font=dict(color="#CBD5E1", size=11)),
                    range=[-0.02, 1.02],
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Node Impurity Index", font=dict(color="#CBD5E1", size=11)),
                    range=[-0.02, 0.55],
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

            st.plotly_chart(fig_imp, width="stretch", key="ni_fig_imp")

        with tab_purity:
            st.markdown(f"##### 🎯 Decision Space Partitioned & Shaded by Leaf Purity max_k(p̂_jk) ({criterion.upper()})")

            fig_pur = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Heatmap of purity
            fig_pur.add_trace(
                go.Heatmap(
                    x=gx,
                    y=gy,
                    z=purity_grid,
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(text="Leaf Purity", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    zmin=0.5,
                    zmax=1.0,
                    opacity=0.75,
                    hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>Purity: %{z:.2f}<extra></extra>",
                )
            )

            # Boundary contour
            fig_pur.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=decision_grid,
                    contours_coloring="none",
                    contours=dict(start=0, end=0, size=1, coloring="none"),
                    line=dict(color="#FFFFFF", width=2.5),
                    name="Class Boundary",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # Scatter points
            fig_pur = add_scatter_points(fig_pur, X, y, marker_size=10)

            st.plotly_chart(fig_pur, width="stretch", key="ni_fig_pur")

        with tab_leaves:
            st.markdown(f"##### 📶 Leaf Node Class Proportions & Impurities ({n_leaves} Terminal Leaves)")

            # Extract leaf values from tree
            leaf_ids = np.unique(clf.apply(X))
            leaf_ginis = []
            leaf_p_pos = []
            leaf_counts = []
            for lid in leaf_ids:
                mask = (clf.apply(X) == lid)
                n_node = np.sum(mask)
                p1 = np.mean(y[mask] == 1)
                leaf_counts.append(n_node)
                leaf_p_pos.append(p1)
                leaf_ginis.append(2.0 * p1 * (1.0 - p1))

            fig_bar = go.Figure()
            leaf_labels = [f"Leaf #{i+1} (N={leaf_counts[i]})" for i in range(len(leaf_ids))]

            fig_bar.add_trace(
                go.Bar(
                    x=leaf_labels,
                    y=leaf_p_pos,
                    name="Class +1 Proportion p̂_j1",
                    marker=dict(color="#38BDF8"),
                    hovertemplate="%{x}<br>Class +1 Prop: %{y:.2f}<extra></extra>",
                )
            )

            fig_bar.add_trace(
                go.Bar(
                    x=leaf_labels,
                    y=leaf_ginis,
                    name="Leaf Gini Impurity",
                    marker=dict(color=COLOR_HYPERPLANE),
                    hovertemplate="%{x}<br>Gini Impurity: %{y:.3f}<extra></extra>",
                )
            )

            fig_bar.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                barmode="group",
                xaxis=dict(
                    title=dict(text="Terminal Leaf Node", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Proportion / Impurity", font=dict(color="#CBD5E1", size=11)),
                    range=[0, 1.05],
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

            st.plotly_chart(fig_bar, width="stretch", key="ni_fig_bar")
