"""
Tile 1: Recursive Binary Splitting & 2D Space Partition Geometry
Slides: 261-268
Focus: Exploring how recursive binary splitting partitions the input space into J disjoint
axis-aligned rectangular regions {R_j}, assigning piecewise constants c_j = avg(y_i in R_j)
for regression or majority class for classification (recreating the 2D & 3D diagrams of Slides 262 & 268).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from utils.datasets import get_dataset, get_2d_regression_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    COLOR_CLASS_NEG,
    COLOR_CLASS_POS,
    BG_PAPER,
    BG_PLOT,
)


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Tree Partition Knobs")

            mode = st.radio(
                "Model Task",
                ["Regression Tree (Slide 262)", "Classification Tree (Slide 268)"],
                index=0,
                key="rp_mode",
                help="Switch between continuous piecewise constant regression and discrete regional classification.",
            )

            max_depth = st.slider(
                "Maximum Tree Depth",
                min_value=1,
                max_value=7,
                value=3,
                step=1,
                key="rp_depth",
                help="Controls maximum depth of recursive binary splitting (tree size J <= 2^depth).",
            )

            min_samples_split = st.slider(
                "Minimum Split Size (splitmin)",
                min_value=2,
                max_value=35,
                value=10,
                step=1,
                key="rp_splitmin",
                help="Minimum number of samples required to split an internal node (MATLAB splitmin, Slide 271).",
            )

            if "Regression" in mode:
                dataset_type = st.selectbox(
                    "Regression Topology",
                    ["Sinusoidal Surface", "Step Pyramid (Blocks)", "Non-linear Saddle"],
                    index=0,
                    key="rp_reg_data",
                )
            else:
                dataset_type = st.selectbox(
                    "Classification Topology",
                    ["XOR / Checkerboard", "Moons", "Concentric Circles", "Overlapping Blobs"],
                    index=0,
                    key="rp_clf_data",
                )

            with st.expander("⚙️ Sample Generation", expanded=False):
                n_samples = st.slider("Sample Points N", 40, 200, 100, 10, key="rp_samples")
                noise = st.slider("Noise Level", 0.02, 0.40, 0.12, 0.02, key="rp_noise")

    # Generate data and fit tree
    is_regression = ("Regression" in mode)
    if is_regression:
        X, y = get_2d_regression_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        metric_err_name = "Training MSE"
        metric_err_val = f"{np.mean((y - y_pred) ** 2):.3f}"
    else:
        X, y = get_dataset(dataset_type, n_samples=n_samples, noise=noise, random_state=42)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        metric_err_name = "Misclassification"
        metric_err_val = f"{np.mean(y != y_pred) * 100.0:.1f}%"

    n_leaves = model.get_n_leaves()
    actual_depth = model.get_depth()

    # Precompute 2D grid for surface and partition plots
    x_min, x_max = float(np.min(X[:, 0]) - 0.5), float(np.max(X[:, 0]) + 0.5)
    y_min, y_max = float(np.min(X[:, 1]) - 0.5), float(np.max(X[:, 1]) + 0.5)
    gx = np.linspace(x_min, x_max, 140)
    gy = np.linspace(y_min, y_max, 140)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]
    ZZ = model.predict(grid_pts).reshape(XX.shape)

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Leaves (Regions J)", f"{n_leaves}")
            m2.metric("Tree Depth", f"{actual_depth} / {max_depth}")
            m3.metric("Min Split Size", f"{min_samples_split}")
            m4.metric(metric_err_name, metric_err_val)

        tab_2d, tab_3d, tab_tree = st.tabs([
            "🗺️ 2D Axis-Aligned Space Partitioning",
            "🧊 3D Piecewise Constant Surface (Slide 262)",
            "🌳 Hierarchical Decision Rules",
        ])

        with tab_2d:
            st.markdown(f"##### 📐 2D Decision Partition: J = {n_leaves} Disjoint Rectangles (Slide 268)")

            fig_2d = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=460)

            # Shaded rectangular regions
            if is_regression:
                fig_2d.add_trace(
                    go.Heatmap(
                        x=gx,
                        y=gy,
                        z=ZZ,
                        colorscale="Blues",
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Leaf Constant c_j", font=dict(color="#CBD5E1", size=10)),
                            tickfont=dict(color="#94A3B8"),
                        ),
                        opacity=0.75,
                        hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>f̂(x) = c_j: %{z:.2f}<extra>Region Value</extra>",
                    )
                )
                # Scatter regression observations
                fig_2d.add_trace(
                    go.Scatter(
                        x=X[:, 0],
                        y=X[:, 1],
                        mode="markers",
                        name="Observations y",
                        marker=dict(
                            size=8,
                            color=y,
                            colorscale="Viridis",
                            line=dict(color="#FFFFFF", width=1.2),
                            colorbar=dict(
                                title=dict(text="True y", font=dict(color="#CBD5E1", size=10)),
                                tickfont=dict(color="#94A3B8"),
                                x=1.15,
                            ),
                        ),
                        customdata=y,
                        hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>True y: %{customdata:.2f}<extra></extra>",
                    )
                )
            else:
                # Classification region surface
                fig_2d.add_trace(
                    go.Heatmap(
                        x=gx,
                        y=gy,
                        z=ZZ,
                        colorscale=[[0, "rgba(244, 63, 94, 0.4)"], [1, "rgba(56, 189, 248, 0.4)"]],
                        showscale=False,
                        hoverinfo="skip",
                    )
                )
                # Contour border between classes
                fig_2d.add_trace(
                    go.Contour(
                        x=gx,
                        y=gy,
                        z=ZZ,
                        contours_coloring="none",
                        contours=dict(start=0, end=0, size=1, coloring="none"),
                        line=dict(color="#FFFFFF", width=2.5),
                        name="Axis-Aligned Boundary",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )
                # Scatter points
                fig_2d = add_scatter_points(fig_2d, X, y, marker_size=10)

            st.plotly_chart(fig_2d, width="stretch", key="rp_fig_2d")

        with tab_3d:
            st.markdown("##### 🧊 Piecewise Constant Surface: f̂(x) = ∑ c_j · I(x ∈ R_j) (Slide 262)")

            fig_3d = go.Figure()

            # 3D Step surface
            fig_3d.add_trace(
                go.Surface(
                    x=gx,
                    y=gy,
                    z=ZZ,
                    colorscale="Viridis",
                    opacity=0.88,
                    showscale=False,
                    hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>Surface f̂(x): %{z:.2f}<extra>Step Plateaus</extra>",
                )
            )

            # Scatter 3D observations
            if is_regression:
                fig_3d.add_trace(
                    go.Scatter3d(
                        x=X[:, 0],
                        y=X[:, 1],
                        z=y,
                        mode="markers",
                        name="Observed y_i",
                        marker=dict(size=4.5, color="#F43F5E", symbol="circle", opacity=0.9),
                        hovertemplate="x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>Obs y: %{z:.2f}<extra></extra>",
                    )
                )

            fig_3d.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                scene=dict(
                    bgcolor=BG_PLOT,
                    xaxis=dict(title="Feature x₁", backgroundcolor=BG_PLOT, gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                    yaxis=dict(title="Feature x₂", backgroundcolor=BG_PLOT, gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                    zaxis=dict(title="Response f̂(x)", backgroundcolor=BG_PLOT, gridcolor="#334155", tickfont=dict(color="#94A3B8")),
                    camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
                ),
                margin=dict(l=20, r=20, t=10, b=50),
            )

            st.plotly_chart(fig_3d, width="stretch", key="rp_fig_3d")

        with tab_tree:
            st.markdown(f"##### 🌲 Interpretable Binary Split Hierarchy (Tree Depth {actual_depth}, {n_leaves} Leaves)")
            feature_names = ["Feature x₁", "Feature x₂"]
            tree_text = export_text(model, feature_names=feature_names, max_depth=5)
            st.code(tree_text, language="text")
            st.caption("Each branch represents an axis-aligned conditional cut: {x | x_j <= s} vs {x | x_j > s} (Slide 265).")
