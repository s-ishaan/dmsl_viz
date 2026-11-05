"""
Module Page: Decision Trees, Pruning & AdaBoost Ensemble Learning
Senior Data Presentation Architecture: 6 Dedicated Conceptual Tabs, Each with Full-Block Visualizations.
"""

import streamlit as st
from .tiles import (
    recursive_partitioning,
    node_impurity,
    cost_complexity_pruning,
    tree_instability,
    adaboost_dynamics,
    adaboost_benchmark,
)


def render():
    st.title("Decision Trees, Pruning & AdaBoost Ensemble Learning")
    st.caption("Recursive Partitioning, Cost-Complexity Pruning & Sequential Committee Learning")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ 1. Recursive Binary Splitting",
        "📐 2. Node Impurity Metrics",
        "✂️ 3. Cost-Complexity Pruning",
        "⚡ 4. Tree Variance & Instability",
        "🚀 5. AdaBoost.M1 Dynamics",
        "🏆 6. AdaBoost Power Benchmark",
    ])

    with tab1:
        recursive_partitioning.render()

    with tab2:
        node_impurity.render()

    with tab3:
        cost_complexity_pruning.render()

    with tab4:
        tree_instability.render()

    with tab5:
        adaboost_dynamics.render()

    with tab6:
        adaboost_benchmark.render()
