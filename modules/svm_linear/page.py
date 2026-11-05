"""
Module Page: Support Vector Machines (Linear & Foundations)
ISYE-7406 Data Mining & Statistical Learning (Slides 210-226)
Senior Data Presentation Architecture: 6 Dedicated Conceptual Tabs, Each with Full-Block Visualizations.
"""

import streamlit as st
from .tiles import (
    hyperplane_algebra,
    perceptron_vs_ls_vs_svm,
    hard_margin_osh,
    soft_margin_slack,
    c_regularization,
    kernel_mapping,
)


def render():
    st.title("Support Vector Machines: Linear Hyperplanes & Formulation")
    st.caption("ISYE-7406 Slide Deck Module | Slides 210–226 | Real-Time Interactive Visual Sandbox")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📐 1. Hyperplane Geometry",
        "⚔️ 2. OLS vs Perceptron vs SVM",
        "📏 3. Hard-Margin OSH",
        "📉 4. Soft-Margin & Slack (ξ)",
        "🎚️ 5. Cost Parameter C (Slide 226)",
        "🚀 6. Feature Space Lift (RKHS)",
    ])

    with tab1:
        hyperplane_algebra.render()

    with tab2:
        perceptron_vs_ls_vs_svm.render()

    with tab3:
        hard_margin_osh.render()

    with tab4:
        soft_margin_slack.render()

    with tab5:
        c_regularization.render()

    with tab6:
        kernel_mapping.render()
