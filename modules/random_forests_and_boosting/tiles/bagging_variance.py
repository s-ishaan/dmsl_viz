"""
Tile 3: Bootstrap Aggregation (Bagging) & Variance Reduction
Focus: Slides 299-302 - Parallel bootstrap averaging of unpruned high-variance trees.
Demonstrates tree diversity across bootstrap samples (Slide 300) and error stabilization
under Majority Voting (Consensus) vs Class Probability Averaging (Slide 301).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
            st.markdown("#### 🎛️ Bagging Knobs")

            total_b = st.slider(
                "Bootstrap Samples B",
                min_value=5,
                max_value=120,
                value=40,
                step=5,
                key="bag_total_b",
                help="Total number of parallel bootstrap trees grown on resampled data.",
            )

            inspect_tree_id = st.slider(
                "Inspect Individual Tree b",
                min_value=1,
                max_value=min(10, total_b),
                value=1,
                step=1,
                key="bag_inspect_id",
                help="Select one constituent tree to inspect its individual erratic boundary.",
            )

            aggregation_rule = st.radio(
                "Ensemble Aggregation Rule",
                ["Probability Averaging", "Consensus (Majority Vote)"],
                index=0,
                key="bag_agg_rule",
                help="Probability: average predicted class probabilities. Consensus: each tree casts a 0/1 vote.",
            )

            tree_depth = st.slider(
                "Base Tree Max Depth",
                min_value=2,
                max_value=8,
                value=6,
                step=1,
                key="bag_depth",
                help="Deep unpruned trees have low bias and high variance—ideal candidates for bagging.",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                dataset_type = st.selectbox(
                    "Topology",
                    ["Moons", "Concentric Circles", "XOR / Checkerboard", "Overlapping Blobs"],
                    index=0,
                    key="bag_data",
                )
                n_samples = st.slider("Sample Count N", 60, 250, 120, 10, key="bag_n")
                noise_lvl = st.slider("Noise Level", 0.05, 0.40, 0.20, 0.02, key="bag_noise")

    # Generate dataset
    X, y_svm = get_dataset(dataset_type, n_samples=n_samples, noise=noise_lvl, random_state=42)
    y = np.where(y_svm == 1, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
    n_tr = len(X_train)

    # 1. Fit Single Unbagged Tree as benchmark
    single_tree = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
    single_tree.fit(X_train, y_train)
    single_err = float(np.mean(single_tree.predict(X_test) != y_test))

    # 2. Grow B bootstrap trees
    trees = []
    rng = np.random.RandomState(42)

    # Grid for contour evaluations
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    gx = np.linspace(x_min, x_max, 110)
    gy = np.linspace(y_min, y_max, 110)
    gxx, gyy = np.meshgrid(gx, gy)
    grid_pts = np.c_[gxx.ravel(), gyy.ravel()]

    test_probs_all = []
    test_preds_all = []
    grid_probs_all = []

    for b in range(total_b):
        # Sampling with replacement
        boot_idx = rng.choice(n_tr, size=n_tr, replace=True)
        X_b, y_b = X_train[boot_idx], y_train[boot_idx]

        tb = DecisionTreeClassifier(max_depth=tree_depth, random_state=b)
        tb.fit(X_b, y_b)
        trees.append(tb)

        p_test = tb.predict_proba(X_test)[:, 1]
        test_probs_all.append(p_test)
        test_preds_all.append(tb.predict(X_test))

        grid_probs_all.append(tb.predict_proba(grid_pts)[:, 1])

    # Convert to arrays: shape (B, N_test)
    test_probs_all = np.array(test_probs_all)
    test_preds_all = np.array(test_preds_all)
    grid_probs_all = np.array(grid_probs_all)

    # Running cumulative error vs B
    err_curve_prob = []
    err_curve_vote = []

    for b_sub in range(1, total_b + 1):
        # Probability averaging
        cum_prob = np.mean(test_probs_all[:b_sub], axis=0)
        pred_p = (cum_prob >= 0.5).astype(int)
        err_curve_prob.append(float(np.mean(pred_p != y_test)))

        # Majority vote
        cum_votes = np.mean(test_preds_all[:b_sub], axis=0)
        pred_v = (cum_votes >= 0.5).astype(int)
        err_curve_vote.append(float(np.mean(pred_v != y_test)))

    final_bag_err = err_curve_prob[-1] if aggregation_rule == "Probability Averaging" else err_curve_vote[-1]
    err_reduction = single_err - final_bag_err

    # Active individual tree predictions on grid
    inspect_tree = trees[inspect_tree_id - 1]
    single_grid_prob = inspect_tree.predict_proba(grid_pts)[:, 1].reshape(gxx.shape)

    # Full Bagged Ensemble grid predictions
    if aggregation_rule == "Probability Averaging":
        bag_grid_prob = np.mean(grid_probs_all, axis=0).reshape(gxx.shape)
    else:
        bag_grid_votes = np.mean(np.array([t.predict(grid_pts) for t in trees]), axis=0).reshape(gxx.shape)
        bag_grid_prob = bag_grid_votes

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bootstrap Trees B", f"{total_b}")
        m2.metric("Single Tree Error", f"{single_err * 100:.1f}%")
        m3.metric("Bagged Error", f"{final_bag_err * 100:.1f}%", delta=f"{-err_reduction * 100:.1f}%", delta_color="inverse")
        m4.metric("Variance Reduction", f"{max(0.0, err_reduction / max(1e-5, single_err) * 100):.1f}%")

        tab_boundary, tab_inspect, tab_curve = st.tabs([
            "🛡️ Bagged Ensemble Boundary",
            f"🌳 Individual Tree b = {inspect_tree_id} (Erratic)",
            "📉 Test Error vs B (Recreating Slide 301)",
        ])

        with tab_boundary:
            st.markdown(f"##### 🛡️ Stabilized Bagged Ensemble Manifold ({aggregation_rule}, B = {total_b})")

            fig_bag = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=420)

            # Ensemble probability shading
            fig_bag.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=bag_grid_prob,
                    colorscale="Tealgrn",
                    opacity=0.65,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="P(Y=1)", font=dict(color="#CBD5E1", size=10)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    contours=dict(start=0.1, end=0.9, size=0.1),
                    hoverinfo="skip",
                )
            )

            # Decision Boundary (0.5)
            fig_bag.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=bag_grid_prob,
                    contours_coloring="none",
                    contours=dict(start=0.5, end=0.5, size=0.1, coloring="none"),
                    line=dict(color="#FFFFFF", width=2.5),
                    name="Bagged Boundary",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            fig_bag = add_scatter_points(fig_bag, X, y_svm, marker_size=7)

            st.plotly_chart(fig_bag, width="stretch", key="bag_fig_ensemble")

        with tab_inspect:
            st.markdown(f"##### 🌳 Constituent Tree b = {inspect_tree_id} Grown on Resampled Data (Slide 300 Diversity)")

            fig_indiv = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=420)

            # Individual tree contour
            fig_indiv.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=single_grid_prob,
                    colorscale="Tealgrn",
                    opacity=0.55,
                    showscale=False,
                    contours=dict(start=0.1, end=0.9, size=0.1),
                    hoverinfo="skip",
                )
            )

            fig_indiv.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=single_grid_prob,
                    contours_coloring="none",
                    contours=dict(start=0.5, end=0.5, size=0.1, coloring="none"),
                    line=dict(color="#F59E0B", width=2.2, dash="dash"),
                    name=f"Tree b={inspect_tree_id} Boundary",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            fig_indiv = add_scatter_points(fig_indiv, X, y_svm, marker_size=7)

            st.plotly_chart(fig_indiv, width="stretch", key="bag_fig_indiv")

        with tab_curve:
            st.markdown("##### 📉 Test Misclassification Error vs Bootstrap Count B (Recreating Slide 301)")

            fig_curve = go.Figure()

            b_axis = list(range(1, total_b + 1))

            # Horizontal line for Single Tree
            fig_curve.add_hline(
                y=single_err * 100.0,
                line_width=1.8,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Single Tree ({single_err * 100:.1f}%)",
                annotation_font_color="#EF4444",
                annotation_position="top left",
            )

            # Consensus (Majority Voting)
            fig_curve.add_trace(
                go.Scatter(
                    x=b_axis,
                    y=[e * 100.0 for e in err_curve_vote],
                    mode="lines",
                    name="Consensus (Majority Vote)",
                    line=dict(color="#F59E0B", width=2.5),
                    hovertemplate="B=%{x}: Vote Err = %{y:.1f}%<extra></extra>",
                )
            )

            # Probability Averaging
            fig_curve.add_trace(
                go.Scatter(
                    x=b_axis,
                    y=[e * 100.0 for e in err_curve_prob],
                    mode="lines",
                    name="Probability Averaging",
                    line=dict(color=COLOR_HYPERPLANE, width=3.0),
                    hovertemplate="B=%{x}: Prob Err = %{y:.1f}%<extra></extra>",
                )
            )

            fig_curve.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Number of Bootstrap Samples B", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    title=dict(text="Test Error Rate (%)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                margin=dict(l=40, r=20, t=15, b=85),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#CBD5E1", size=9.5),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    bordercolor="#334155",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_curve, width="stretch", key="bag_fig_curve")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Why Bagging Works (Slides 299–302)", expanded=False):
        st.markdown(
            r"""
            **Bootstrap Aggregation (Bagging)**:
            - Given training dataset $Z = \{(x_1, y_1), \dots, (x_N, y_N)\}$, draw $B$ bootstrap datasets $Z^{*b}$ by sampling with replacement.
            - Fit an unpruned deep decision tree $\hat{f}^{*b}(x)$ on each bootstrap sample.
            - Combine predictions:
              $$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b}(x)$$
            - **Bias-Variance Dynamics (Slide 302)**:
              - Trees grown deep have **low bias** (capable of capturing complex interaction surfaces).
              - However, single trees are notoriously **unstable with high variance** (small perturbations in data cascade into massive partition changes).
              - Because each bootstrap tree is identically distributed (i.d.), $\mathbb{E}[\hat{f}_{\text{bag}}(x)] = \mathbb{E}[\hat{f}^{*b}(x)]$, meaning **bias is unchanged**.
              - Averaging across $B$ models dramatically reduces prediction **variance**, yielding a smooth, robust decision boundary.
            """
        )
