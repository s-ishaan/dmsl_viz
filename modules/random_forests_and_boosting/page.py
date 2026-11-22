"""
Module Page: Random Forests & Gradient Tree Boosting (MART)
Conceptual Flow: MART Residual Fitting, Regularization by Shrinkage, Bagging Variance Decomposition,
Leo Breiman's De-correlated Random Forests, and the Benchmark Arena.
"""

import streamlit as st
from .tiles import (
    mart_dynamics,
    shrinkage_interaction,
    bagging_variance,
    correlation_bottleneck,
    random_forests,
    ensemble_arena,
)


def render():
    st.title("Random Forests & Gradient Tree Boosting (MART)")
    st.caption("Multiple Additive Regression Trees, Variance Decomposition & De-correlated Ensembles")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌲 1. MART / Gradient Tree Boosting",
        "🎚️ 2. Shrinkage ν & Tree Size J",
        "🎲 3. Bootstrap Aggregation (Bagging)",
        "📉 4. Tree Variance & Correlation Floor",
        "🌳 5. Random Forests & Feature Subsampling",
        "🏆 6. Ensemble Benchmark Arena",
    ])

    with tab1:
        mart_dynamics.render()

    with tab2:
        shrinkage_interaction.render()

    with tab3:
        bagging_variance.render()

    with tab4:
        correlation_bottleneck.render()

    with tab5:
        random_forests.render()

    with tab6:
        ensemble_arena.render()
