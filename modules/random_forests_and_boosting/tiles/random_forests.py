"""
Tile 5: Random Forests & Feature Subsampling (m <= p)
Focus: Slides 306-308 - Algorithm 15.1 (Leo Breiman).
De-correlating trees by randomly selecting m candidate features at each split.
Demonstrates why m = floor(sqrt(p)) drastically suppresses pairwise tree correlation ρ,
lowering the variance floor without inflating bias.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from utils.datasets import get_multidimensional_dataset, get_dataset
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
            st.markdown("#### 🎛️ Random Forest Knobs")

            total_p = st.slider(
                "Total Feature Count p",
                min_value=4,
                max_value=16,
                value=9,
                step=1,
                key="rf_total_p",
                help="Total number of features (informative signals + noisy dimensions).",
            )

            rec_m = int(np.floor(np.sqrt(total_p)))

            m_features = st.slider(
                f"Candidate Split Features m (Optimum: {rec_m})",
                min_value=1,
                max_value=total_p,
                value=rec_m,
                step=1,
                key="rf_m_features",
                help=f"Number of randomly sampled candidate variables considered at each node split. Breiman's recommendation for classification is m = ⌊√p⌋ = {rec_m}.",
            )

            n_trees_b = st.slider(
                "Number of Forest Trees B",
                min_value=10,
                max_value=150,
                value=60,
                step=10,
                key="rf_n_trees",
                help="Number of de-correlated trees in the ensemble.",
            )

            max_depth = st.slider(
                "Tree Max Depth (Unpruned = Deep)",
                min_value=2,
                max_value=10,
                value=7,
                step=1,
                key="rf_depth",
                help="Deep trees have minimal bias. The forest variance reduction eliminates their individual high variance.",
            )

            with st.expander("⚙️ Dataset Settings", expanded=False):
                n_samples = st.slider("Sample Count N", 100, 400, 250, 25, key="rf_n_samples")
                noise_ratio = st.slider("Label Noise Ratio", 0.02, 0.30, 0.10, 0.02, key="rf_noise")
                n_info = st.slider("Informative Features", 2, min(total_p, 6), 3, key="rf_info")

    # Generate multi-dimensional dataset
    X, y = get_multidimensional_dataset(
        n_samples=n_samples,
        p=total_p,
        n_informative=n_info,
        noise_ratio=noise_ratio,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)

    # 1. Fit Random Forest with selected m
    rf_model = RandomForestClassifier(
        n_estimators=n_trees_b,
        max_features=m_features,
        max_depth=max_depth,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_test_err = float(np.mean(rf_model.predict(X_test) != y_test))

    # 2. Fit Standard Bagging (m = p, all features considered)
    bag_model = RandomForestClassifier(
        n_estimators=n_trees_b,
        max_features=total_p,
        max_depth=max_depth,
        random_state=42,
    )
    bag_model.fit(X_train, y_train)
    bag_test_err = float(np.mean(bag_model.predict(X_test) != y_test))

    # Estimate empirical pairwise tree correlation rho across predictions
    def estimate_tree_correlation(forest, X_eval):
        tree_preds = np.array([t.predict(X_eval) for t in forest.estimators_])  # shape (B, N)
        # Pearson correlation matrix across trees
        corr_mat = np.corrcoef(tree_preds)
        # Average off-diagonal elements
        n_est = len(forest.estimators_)
        if n_est <= 1:
            return 0.0
        mask = ~np.eye(n_est, dtype=bool)
        mean_corr = np.nanmean(corr_mat[mask])
        return float(max(0.0, mean_corr))

    rho_rf = estimate_tree_correlation(rf_model, X_test)
    rho_bag = estimate_tree_correlation(bag_model, X_test)

    # Sweep across m = 1 .. p to show the trade-off curve
    m_sweep_vals = list(range(1, total_p + 1))
    sweep_errors = []
    sweep_rhos = []

    for m_val in m_sweep_vals:
        rf_sweep = RandomForestClassifier(
            n_estimators=40,
            max_features=m_val,
            max_depth=max_depth,
            random_state=42,
        )
        rf_sweep.fit(X_train, y_train)
        err_m = float(np.mean(rf_sweep.predict(X_test) != y_test))
        rho_m = estimate_tree_correlation(rf_sweep, X_test)
        sweep_errors.append(err_m * 100.0)
        sweep_rhos.append(rho_m)

    # Feature Importances
    importances = rf_model.feature_importances_
    feat_names = [f"x{i+1} {'(Signal)' if i < n_info else '(Noise)'}" for i in range(total_p)]

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected m", f"{m_features} of {total_p}", help=f"Optimal Breiman rule: m = floor(sqrt({total_p})) = {rec_m}")
        m2.metric("RF Test Error", f"{rf_test_err * 100:.1f}%")
        m3.metric("Bagging Test Error", f"{bag_test_err * 100:.1f}%", delta=f"{rf_test_err * 100 - bag_test_err * 100:+.1f}%", delta_color="inverse")
        m4.metric("Tree Correlation ρ", f"{rho_rf:.2f}", delta=f"{rho_rf - rho_bag:+.2f} vs bag", delta_color="inverse")

        tab_tradeoff, tab_importance, tab_2d = st.tabs([
            "📉 Error & Correlation vs m (Slide 306)",
            "📊 Feature Importance Profile (MDI)",
            "🗺️ 2D De-correlated Boundary",
        ])

        with tab_tradeoff:
            st.markdown(f"##### 📉 Effect of Candidate Size m on Tree Correlation ρ and Test Error (p = {total_p})")

            fig_tradeoff = go.Figure()

            # Correlation curve (Sky Blue, secondary axis style or scaled)
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=m_sweep_vals,
                    y=[r * 100.0 for r in sweep_rhos],
                    mode="lines+markers",
                    name="Tree Correlation ρ (%)",
                    line=dict(color="#38BDF8", width=2.5, dash="dash"),
                    marker=dict(size=6, color="#38BDF8"),
                    hovertemplate="m = %{x}: Correlation ρ = %{y:.1f}%<extra></extra>",
                )
            )

            # Test Error curve (Emerald)
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=m_sweep_vals,
                    y=sweep_errors,
                    mode="lines+markers",
                    name="Test Error Rate (%)",
                    line=dict(color=COLOR_HYPERPLANE, width=3.2),
                    marker=dict(size=7, color=COLOR_HYPERPLANE),
                    hovertemplate="m = %{x}: Test Error = %{y:.1f}%<extra></extra>",
                )
            )

            # Highlight Breiman's recommendation m = floor(sqrt(p))
            fig_tradeoff.add_vline(
                x=rec_m,
                line_width=2.0,
                line_dash="dot",
                line_color="#F59E0B",
                annotation_text=f"Breiman m=⌊√p⌋ ({rec_m})",
                annotation_font_color="#F59E0B",
                annotation_position="top left",
            )

            fig_tradeoff.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Candidate Features Considered at Each Split (m)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    tickmode="linear",
                    dtick=1,
                ),
                yaxis=dict(
                    title=dict(text="Percentage (%)", font=dict(color="#CBD5E1", size=11)),
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

            st.plotly_chart(fig_tradeoff, width="stretch", key="rf_fig_tradeoff")

        with tab_importance:
            st.markdown("##### 📊 Mean Decrease in Impurity (MDI) Across All Features")

            sorted_indices = np.argsort(importances)
            bar_colors = ["#10B981" if sorted_indices[i] < n_info else "#64748B" for i in range(total_p)]

            fig_imp = go.Figure()

            fig_imp.add_trace(
                go.Bar(
                    y=[feat_names[idx] for idx in sorted_indices],
                    x=[importances[idx] for idx in sorted_indices],
                    orientation="h",
                    marker=dict(color=bar_colors),
                    hovertemplate="Feature %{y}: MDI = %{x:.3f}<extra></extra>",
                )
            )

            fig_imp.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    title=dict(text="Relative Importance (Gini Impurity Decrease)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                ),
                yaxis=dict(
                    tickfont=dict(color="#CBD5E1", size=9.5),
                ),
                margin=dict(l=70, r=20, t=15, b=45),
            )

            st.plotly_chart(fig_imp, width="stretch", key="rf_fig_imp")

        with tab_2d:
            st.markdown("##### 🗺️ 2D Decision Manifold: Random Forest (m = ⌊√p⌋) on Non-linear Moons")

            # 2D projection on moons
            X2, y2_svm = get_dataset("Moons", n_samples=180, noise=0.18, random_state=42)
            rf_2d = RandomForestClassifier(n_estimators=n_trees_b, max_features=1, max_depth=max_depth, random_state=42)
            rf_2d.fit(X2, y2_svm)

            x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
            y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
            gx = np.linspace(x_min, x_max, 110)
            gy = np.linspace(y_min, y_max, 110)
            gxx, gyy = np.meshgrid(gx, gy)
            grid_pts = np.c_[gxx.ravel(), gyy.ravel()]

            prob_2d = rf_2d.predict_proba(grid_pts)[:, 1].reshape(gxx.shape)

            fig_2d = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=420)

            fig_2d.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=prob_2d,
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

            fig_2d.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=prob_2d,
                    contours_coloring="none",
                    contours=dict(start=0.5, end=0.5, size=0.1, coloring="none"),
                    line=dict(color="#FFFFFF", width=2.5),
                    name="RF Boundary",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            fig_2d = add_scatter_points(fig_2d, X2, y2_svm, marker_size=7)

            st.plotly_chart(fig_2d, width="stretch", key="rf_fig_2d")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: Algorithm 15.1 (Random Forest)", expanded=False):
        st.markdown(
            r"""
            **Leo Breiman's Random Forest Algorithm (Algorithm 15.1, Slide 306–307)**:
            
            1. **For $b = 1$ to $B$**:
               - Draw a bootstrap sample $Z^*$ of size $N$ from the training data with replacement.
               - Grow a tree $T_b$ to the bootstrapped data by recursively repeating the following steps for each terminal node until the minimum node size $n_{\min}$ is reached:
                 1. Select $m \le p$ variables at random from the $p$ available input variables.
                 2. Pick the best split variable and split-point among the candidate subset of size $m$.
                 3. Split the node into two daughter nodes.
            2. **Output the Ensemble**: $\{T_b\}_{b=1}^B$.
            3. **Prediction**:
               - **Classification**: $\hat{C}_{\text{rf}}^B(x) = \text{majority vote}\{\hat{C}_b(x)\}_{b=1}^B$ (or averaged probability).
               - **Regression**: $\hat{f}_{\text{rf}}^B(x) = \frac{1}{B}\sum_{b=1}^B T_b(x)$.
            
            **Why $m = \lfloor\sqrt{p}\rfloor$ Works So Well (Slide 308)**:
            - If one or two variables are dominant predictors, standard bagging splits on them at the top of almost every tree, keeping tree predictions highly correlated ($\rho \approx 0.4 - 0.5$).
            - By forcing each split to pick from a random sub-sample $m < p$, other features get chosen, forcing trees to explore orthogonal projection directions.
            - This drives pairwise correlation down to $\rho \le 0.05$, dramatically reducing the irreducible variance floor $\rho \sigma^2$ with virtually no increase in single-tree bias.
            """
        )
