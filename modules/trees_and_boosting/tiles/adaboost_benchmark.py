"""
Tile 6: AdaBoost Power Benchmark (Stump vs Deep Tree vs Committee)
Slides: 288-290
Focus: Direct replication of the lecture benchmark on Slides 288-290.
Compares:
1. Single Decision Stump (error ~0.458, barely above random guessing)
2. 244-node Complex Tree (error ~0.247)
3. AdaBoost of Stumps (error plunging to ~0.058 over M iterations)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from utils.datasets import get_dataset
from utils.plotting import (
    get_base_2d_figure,
    add_scatter_points,
    BG_PAPER,
    BG_PLOT,
)


def generate_10d_sphere_dataset(n_train=600, n_test=1200, random_state=42):
    """
    Generate the 10-dimensional simulation:
    X ~ N(0, I_10)
    Y = 1 if sum(X_j^2) > chi2_10(0.5) = 9.34, else -1.
    """
    rng = np.random.RandomState(random_state)
    X_train = rng.randn(n_train, 10)
    y_train = np.where(np.sum(X_train ** 2, axis=1) > 9.34, 1, -1)

    X_test = rng.randn(n_test, 10)
    y_test = np.where(np.sum(X_test ** 2, axis=1) > 9.34, 1, -1)

    return X_train, y_train, X_test, y_test


def render():
    col_ctrl, col_plot = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### 🎛️ Benchmark Controls")

            sim_dataset = st.radio(
                "Benchmark Experiment",
                [
                    "10D Sphere Simulation (Spherical Boundary)",
                    "2D Concentric Circles (Visualizable Boundary)",
                    "2D XOR Checkerboard",
                ],
                index=0,
                key="bm_dataset",
                help="Tests 10 standard normal variables with spherical Chi-squared boundary.",
            )

            max_iters = st.slider(
                "Max Boosting Iterations M",
                min_value=20,
                max_value=250,
                value=120,
                step=20,
                key="bm_max_m",
                help="Number of sequential boosting iterations M.",
            )

            deep_tree_depth = st.slider(
                "Deep Complex Tree Max Depth",
                min_value=6,
                max_value=14,
                value=9,
                step=1,
                key="bm_deep_depth",
                help="Simulates an unpruned complex decision tree.",
            )

            with st.expander("⚙️ Simulation Size", expanded=False):
                n_train = st.slider("Training Samples", 200, 1000, 500, 100, key="bm_ntrain")
                n_test = st.slider("Testing Samples", 500, 2000, 1000, 250, key="bm_ntest")

    # Generate data
    if sim_dataset.startswith("10D Sphere"):
        X_train, y_train, X_test, y_test = generate_10d_sphere_dataset(n_train=n_train, n_test=n_test, random_state=42)
        is_2d = False
    else:
        is_2d = True
        top_name = "Concentric Circles" if "Concentric" in sim_dataset else "XOR / Checkerboard"
        X_all, y_all = get_dataset(top_name, n_samples=n_train + n_test, noise=0.15, random_state=42)
        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_test, y_test = X_all[n_train:], y_all[n_train:]

    # Model 1: Single Decision Stump (depth=1)
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(X_train, y_train)
    err_stump_test = np.mean(stump.predict(X_test) != y_test)

    # Model 2: Complex Unpruned Deep Tree
    deep_tree = DecisionTreeClassifier(max_depth=deep_tree_depth, random_state=42)
    deep_tree.fit(X_train, y_train)
    err_deep_test = np.mean(deep_tree.predict(X_test) != y_test)
    n_deep_leaves = deep_tree.get_n_leaves()

    # Model 3: AdaBoost of Stumps across M iterations
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=max_iters,
        random_state=42,
    )
    ada.fit(X_train, y_train)

    # Compute test error progression across all iterations m = 1 ... M
    test_errors_ada = []
    for pred in ada.staged_predict(X_test):
        test_errors_ada.append(np.mean(pred != y_test))
    test_errors_ada = np.array(test_errors_ada)
    final_ada_err = test_errors_ada[-1]

    with col_plot:
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Single Stump Error", f"{err_stump_test:.3f}", help="Barely better than 0.50 random guessing.")
            m2.metric(f"Deep Tree ({n_deep_leaves} leaves)", f"{err_deep_test:.3f}", help="Complex unpruned tree error.")
            m3.metric(f"AdaBoost (M={max_iters})", f"{final_ada_err:.3f}", delta=f"{(final_ada_err - err_deep_test)*100:.1f}% vs Deep", delta_color="inverse")
            m4.metric("Error Reduction", f"{(1.0 - final_ada_err / err_stump_test)*100:.0f}%")

        tabs_list = ["📉 Test Error vs Boosting Iterations M", "📊 Final Model Benchmark Summary"]
        if is_2d:
            tabs_list.insert(1, "🗺️ 2D Decision Boundaries (Stump vs Deep vs AdaBoost)")

        tabs = st.tabs(tabs_list)

        with tabs[0]:
            st.markdown(f"##### 📉 Test Error Convergence vs Iterations M")

            fig_curve = go.Figure()
            m_axis = np.arange(1, len(test_errors_ada) + 1)

            # 1. Single Stump horizontal benchmark (0.458 on Slide 290)
            fig_curve.add_hline(
                y=err_stump_test,
                line_width=2.0,
                line_dash="dot",
                line_color="#94A3B8",
                annotation_text=f"Single Stump ({err_stump_test:.3f})",
                annotation_position="top right",
                annotation_font=dict(color="#94A3B8", size=10),
            )

            # 2. Deep Tree horizontal benchmark (0.247 on Slide 290)
            fig_curve.add_hline(
                y=err_deep_test,
                line_width=2.0,
                line_dash="dash",
                line_color="#38BDF8",
                annotation_text=f"Complex Tree ({err_deep_test:.3f})",
                annotation_position="top right",
                annotation_font=dict(color="#38BDF8", size=10),
            )

            # 3. AdaBoost Test Error curve (Orange line matching Slide 290)
            fig_curve.add_trace(
                go.Scatter(
                    x=m_axis,
                    y=test_errors_ada,
                    mode="lines",
                    name="AdaBoost (Stumps)",
                    line=dict(color="#F97316", width=3.0),
                    hovertemplate="Iteration M: %{x}<br>Test Error: %{y:.4f}<extra></extra>",
                )
            )

            # 4. Final AdaBoost marker
            fig_curve.add_trace(
                go.Scatter(
                    x=[len(test_errors_ada)],
                    y=[final_ada_err],
                    mode="markers+text",
                    name=f"Final AdaBoost Error ({final_ada_err:.3f})",
                    marker=dict(size=10, color="#F43F5E"),
                    text=[f"  {final_ada_err:.3f}"],
                    textposition="bottom right",
                    textfont=dict(color="#F43F5E", size=11),
                    hoverinfo="skip",
                )
            )

            fig_curve.update_layout(
                height=460,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Boosting Iterations M", font=dict(color="#CBD5E1", size=11)),
                    range=[0, max_iters + 5],
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Test Error Rate", font=dict(color="#CBD5E1", size=11)),
                    range=[0.0, 0.55],
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                margin=dict(l=45, r=45, t=15, b=75),
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

            st.plotly_chart(fig_curve, width="stretch", key="bm_fig_curve")
            st.caption("Notice how AdaBoost of trivial 1-split stumps plunges well beneath the complex deep tree.")

        if is_2d:
            with tabs[1]:
                st.markdown("##### 🗺️ 2D Decision Boundaries: Stump vs Deep Tree vs AdaBoost")

                model_choice = st.radio(
                    "View Model Boundary",
                    ["(a) Single Decision Stump", "(b) Deep Complex Tree", "(c) AdaBoost Committee"],
                    horizontal=True,
                    key="bm_view_choice",
                )

                x_min, x_max = float(np.min(X_train[:, 0]) - 0.5), float(np.max(X_train[:, 0]) + 0.5)
                y_min, y_max = float(np.min(X_train[:, 1]) - 0.5), float(np.max(X_train[:, 1]) + 0.5)
                gx = np.linspace(x_min, x_max, 130)
                gy = np.linspace(y_min, y_max, 130)
                XX, YY = np.meshgrid(gx, gy)
                grid_pts = np.c_[XX.ravel(), YY.ravel()]

                if model_choice.startswith("(a)"):
                    active_model = stump
                    lbl = "Single Stump"
                elif model_choice.startswith("(b)"):
                    active_model = deep_tree
                    lbl = f"Deep Tree ({n_deep_leaves} leaves)"
                else:
                    active_model = ada
                    lbl = f"AdaBoost (M={max_iters})"

                ZZ_bm = active_model.predict(grid_pts).reshape(XX.shape)

                fig_2d_bm = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=450)

                # Region shading
                fig_2d_bm.add_trace(
                    go.Heatmap(
                        x=gx,
                        y=gy,
                        z=ZZ_bm,
                        colorscale=[[0, "rgba(244, 63, 94, 0.35)"], [1, "rgba(56, 189, 248, 0.35)"]],
                        showscale=False,
                        hoverinfo="skip",
                    )
                )

                # Boundary line
                fig_2d_bm.add_trace(
                    go.Contour(
                        x=gx,
                        y=gy,
                        z=ZZ_bm,
                        contours_coloring="none",
                        contours=dict(start=0, end=0, size=1, coloring="none"),
                        line=dict(color="#FFFFFF", width=2.8),
                        name=f"{lbl} Boundary",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

                # Scatter test points
                fig_2d_bm = add_scatter_points(fig_2d_bm, X_test[:150], y_test[:150], marker_size=8)

                st.plotly_chart(fig_2d_bm, width="stretch", key="bm_fig_2d")

        with tabs[-1]:
            st.markdown("##### 📊 Comparative Test Error Rate")

            fig_bar_bench = go.Figure()
            models_list = ["Single Decision Stump", f"Complex Tree ({n_deep_leaves} Leaves)", f"AdaBoost ({max_iters} Stumps)"]
            err_list = [err_stump_test, err_deep_test, final_ada_err]
            colors = ["#94A3B8", "#38BDF8", "#10B981"]

            fig_bar_bench.add_trace(
                go.Bar(
                    x=models_list,
                    y=err_list,
                    marker=dict(color=colors),
                    text=[f"{e:.3f}" for e in err_list],
                    textposition="auto",
                    textfont=dict(color="#FFFFFF", size=12),
                    hovertemplate="%{x}<br>Test Error: %{y:.4f}<extra></extra>",
                )
            )

            fig_bar_bench.update_layout(
                height=450,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Model Architecture", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    tickfont=dict(color="#94A3B8"),
                ),
                yaxis=dict(
                    title=dict(text="Test Error Rate", font=dict(color="#CBD5E1", size=11)),
                    range=[0, 0.55],
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                margin=dict(l=45, r=25, t=15, b=60),
            )

            st.plotly_chart(fig_bar_bench, width="stretch", key="bm_fig_summary")
