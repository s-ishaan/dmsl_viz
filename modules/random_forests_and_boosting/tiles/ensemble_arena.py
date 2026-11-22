"""
Tile 6: The Great Ensemble Benchmark Arena
Focus: Slides 298 & 308 - Direct head-to-head empirical comparison of:
Single CART vs Bagged Trees vs Random Forest vs Gradient Tree Boosting (MART).
Directly replicates the benchmark hierarchy from the spam problem:
CART (8.7%) > Bagging (5.4%) > Random Forest (4.88%) > MART (4.5%).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from utils.datasets import get_dataset, get_multidimensional_dataset
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
            st.markdown("#### 🎛️ Arena Benchmark Knobs")

            benchmark_scenario = st.selectbox(
                "Benchmark Dataset Challenge",
                [
                    "Spam Problem Simulation (p=57)",
                    "Noisy Moons (Non-linear 2D)",
                    "Concentric Circles (Radial 2D)",
                    "XOR Checkerboard (High Interaction)",
                ],
                index=0,
                key="arena_scenario",
                help="Switch between the textbook Spam Problem (Slides 298 & 308) and 2D geometric boundary challenges.",
            )

            n_trees = st.slider(
                "Ensemble Trees B / Steps M",
                min_value=20,
                max_value=200,
                value=80,
                step=20,
                key="arena_n_trees",
                help="Number of trees grown for Bagging, Random Forest, and Gradient Boosting.",
            )

            test_split = st.slider(
                "Test Set Split Ratio",
                min_value=0.20,
                max_value=0.50,
                value=0.35,
                step=0.05,
                key="arena_split",
            )

            with st.expander("⚙️ Advanced Hyperparameters", expanded=False):
                mart_lr = st.slider("MART Learning Rate ν", 0.02, 0.40, 0.10, 0.02, key="arena_lr")
                max_tree_depth = st.slider("CART / Bag / RF Max Depth", 3, 10, 6, 1, key="arena_depth")
                random_seed = st.number_input("Random Seed", value=42, step=1, key="arena_seed")

    # Generate data based on scenario
    if "Spam" in benchmark_scenario:
        # Recreate the 57-feature spam environment
        p_features = 57
        X, y_raw = get_multidimensional_dataset(
            n_samples=600,
            p=p_features,
            n_informative=18,
            noise_ratio=0.08,
            random_state=random_seed,
        )
    elif "Moons" in benchmark_scenario:
        p_features = 2
        X, y_raw = get_dataset("Moons", n_samples=300, noise=0.22, random_state=random_seed)
    elif "Concentric" in benchmark_scenario:
        p_features = 2
        X, y_raw = get_dataset("Concentric Circles", n_samples=300, noise=0.20, random_state=random_seed)
    else:
        p_features = 2
        X, y_raw = get_dataset("XOR / Checkerboard", n_samples=300, noise=0.20, random_state=random_seed)

    y = np.where(y_raw == 1, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_seed, stratify=y)

    # 1. Single CART
    cart = DecisionTreeClassifier(max_depth=max_tree_depth, random_state=random_seed)
    cart.fit(X_train, y_train)
    cart_err = float(np.mean(cart.predict(X_test) != y_test)) * 100.0

    # 2. Bagged Trees (m = p)
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_tree_depth),
        n_estimators=n_trees,
        random_state=random_seed,
        n_jobs=-1,
    )
    bag.fit(X_train, y_train)
    bag_err = float(np.mean(bag.predict(X_test) != y_test)) * 100.0

    # 3. Random Forest (m = floor(sqrt(p)))
    m_rf = max(1, int(np.floor(np.sqrt(p_features))))
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_features=m_rf,
        max_depth=max_tree_depth,
        random_state=random_seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_err = float(np.mean(rf.predict(X_test) != y_test)) * 100.0

    # 4. MART / Gradient Tree Boosting (J=6, learning_rate=ν)
    mart = GradientBoostingClassifier(
        n_estimators=n_trees,
        learning_rate=mart_lr,
        max_leaf_nodes=6,
        random_state=random_seed,
    )
    mart.fit(X_train, y_train)
    mart_err = float(np.mean(mart.predict(X_test) != y_test)) * 100.0

    models_dict = {
        "Single CART Tree": cart_err,
        "Bagged Trees": bag_err,
        "Random Forest": rf_err,
        "MART (Gradient Boosting)": mart_err,
    }

    best_model_name = min(models_dict, key=models_dict.get)
    best_err = models_dict[best_model_name]

    with col_plot:
        # Top KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Single CART", f"{cart_err:.1f}%")
        m2.metric("Bagged Trees", f"{bag_err:.1f}%", delta=f"{cart_err - bag_err:+.1f}% vs CART", delta_color="normal")
        m3.metric("Random Forest", f"{rf_err:.1f}%", delta=f"{cart_err - rf_err:+.1f}% vs CART", delta_color="normal")
        m4.metric("MART Boosting", f"{mart_err:.1f}%", delta=f"{cart_err - mart_err:+.1f}% vs CART", delta_color="normal")

        tab_leaderboard, tab_boundaries, tab_radar = st.tabs([
            "🏆 Leaderboard (Slides 298 & 308)",
            "🗺️ 4-Way Decision Manifold Comparison",
            "🛡️ Method Architecture Radar",
        ])

        with tab_leaderboard:
            st.markdown(f"##### 🏆 Test Misclassification Error Leaderboard ({benchmark_scenario})")

            model_names = ["Single CART", "Bagged Trees", "Random Forest", "MART (GBM)"]
            errors = [cart_err, bag_err, rf_err, mart_err]
            colors = ["#EF4444", "#F59E0B", "#38BDF8", "#10B981"]

            fig_bar = go.Figure()

            fig_bar.add_trace(
                go.Bar(
                    x=model_names,
                    y=errors,
                    marker=dict(color=colors),
                    text=[f"{e:.1f}%" for e in errors],
                    textposition="outside",
                    hovertemplate="%{x}: Error = %{y:.2f}%<extra></extra>",
                )
            )

            # Slide 308 textbook references line if in Spam mode
            if "Spam" in benchmark_scenario:
                fig_bar.add_annotation(
                    text="Textbook Reference (Slide 308): CART 8.7% | Bagging 5.4% | RF 4.88% | MART 4.5%",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=1.12,
                    showarrow=False,
                    font=dict(color="#F59E0B", size=10.5),
                )

            fig_bar.update_layout(
                height=420,
                paper_bgcolor=BG_PAPER,
                plot_bgcolor=BG_PLOT,
                xaxis=dict(
                    gridcolor="#1E293B",
                    tickfont=dict(color="#CBD5E1", size=10),
                ),
                yaxis=dict(
                    title=dict(text="Test Error Rate (%)", font=dict(color="#CBD5E1", size=11)),
                    gridcolor="#1E293B",
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94A3B8", size=9.5),
                    range=[0, max(errors) * 1.25],
                ),
                margin=dict(l=40, r=20, t=35, b=45),
            )

            st.plotly_chart(fig_bar, width="stretch", key="arena_fig_bar")

        with tab_boundaries:
            if p_features == 2:
                st.markdown("##### 🗺️ Visualizing Decision Manifolds: From Blocky Steps to Smooth Curved Ensembles")

                active_viz_model = st.selectbox(
                    "Inspect Model Boundary",
                    ["Single CART Tree", "Bagged Trees", "Random Forest", "MART (Gradient Boosting)"],
                    index=2,
                    key="arena_viz_sel",
                )

                chosen_clf = (
                    cart if active_viz_model == "Single CART Tree"
                    else bag if active_viz_model == "Bagged Trees"
                    else rf if active_viz_model == "Random Forest"
                    else mart
                )

                x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                gx = np.linspace(x_min, x_max, 110)
                gy = np.linspace(y_min, y_max, 110)
                gxx, gyy = np.meshgrid(gx, gy)
                grid_pts = np.c_[gxx.ravel(), gyy.ravel()]

                prob_grid = chosen_clf.predict_proba(grid_pts)[:, 1].reshape(gxx.shape)

                fig_arena_bound = get_base_2d_figure((x_min, x_max), (y_min, y_max), title="", height=420)

                fig_arena_bound.add_trace(
                    go.Contour(
                        x=gx,
                        y=gy,
                        z=prob_grid,
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

                fig_arena_bound.add_trace(
                    go.Contour(
                        x=gx,
                        y=gy,
                        z=prob_grid,
                        contours_coloring="none",
                        contours=dict(start=0.5, end=0.5, size=0.1, coloring="none"),
                        line=dict(color="#FFFFFF", width=2.5),
                        name="Decision Boundary (P = 0.5)",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

                fig_arena_bound = add_scatter_points(fig_arena_bound, X, y_raw, marker_size=7)

                st.plotly_chart(fig_arena_bound, width="stretch", key="arena_fig_bound")
            else:
                st.info("💡 2D Boundary Contour is available for 2D datasets. Switch to 'Noisy Moons' or 'Concentric Circles' in the knobs panel to visually inspect the 2D decision boundary.")

        with tab_radar:
            st.markdown("##### 🛡️ Architectural Trade-off Radar across Statistical Learning Paradigms")

            categories = ["Low Bias", "Low Variance", "Speed (Parallelism)", "No Tuning Needed", "Interpretability"]

            fig_radar = go.Figure()

            # Single CART
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[4, 1, 5, 4, 5],
                    theta=categories,
                    fill="toself",
                    name="Single CART",
                    line=dict(color="#EF4444"),
                )
            )

            # Bagging
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[4, 3, 5, 5, 2],
                    theta=categories,
                    fill="toself",
                    name="Bagged Trees",
                    line=dict(color="#F59E0B"),
                )
            )

            # Random Forest
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[4, 5, 5, 5, 2],
                    theta=categories,
                    fill="toself",
                    name="Random Forest",
                    line=dict(color="#38BDF8"),
                )
            )

            # MART
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[5, 5, 2, 2, 3],
                    theta=categories,
                    fill="toself",
                    name="MART (Gradient Boosting)",
                    line=dict(color=COLOR_HYPERPLANE),
                )
            )

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                    angularaxis=dict(gridcolor="#1E293B", tickfont=dict(color="#CBD5E1", size=10)),
                    bgcolor=BG_PLOT,
                ),
                height=420,
                paper_bgcolor=BG_PAPER,
                margin=dict(l=40, r=40, t=15, b=85),
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

            st.plotly_chart(fig_radar, width="stretch", key="arena_fig_radar")

    st.markdown("---")
    with st.expander("📚 Theoretical Formulation: The Hierarchy of Tree Ensembles (Slides 298 & 308)", expanded=False):
        st.markdown(
            r"""
            **The Textbook Spam Classification Benchmark (Slides 298 & 308)**:
            - **Problem Setting**: 4601 email messages, 57 features (relative frequencies of common spam words and punctuation marks).
            - **Empirical Leaderboard**:
              1. **MART / Gradient Tree Boosting**: **4.5% test error** (Best overall; sequentially optimizes loss gradient).
              2. **Random Forests**: **4.88% test error** (De-correlates trees via $m = \lfloor\sqrt{p}\rfloor$ feature subsampling).
              3. **Bagged Trees**: **5.4% test error** (Averages bootstrap trees, but limited by tree correlation floor $\rho\sigma^2$).
              4. **Additive Logistic Regression / MARS**: **5.5% test error**.
              5. **Single CART Tree**: **8.7% test error** (High variance, rigid axis-aligned cuts).

            **Key Takeaway**:
            - **Bagging** reduces variance through simple bootstrap averaging.
            - **Random Forests** dramatically enhances bagging by de-correlating the trees ($\rho \to 0$), lowering the variance floor.
            - **MART / Gradient Boosting** iteratively attacks residuals, achieving state-of-the-art predictive accuracy at the cost of being strictly sequential (slower to train than parallel forests).
            """
        )
