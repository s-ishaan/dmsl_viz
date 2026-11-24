"""
Module Page: Principal Component Analysis & Autoencoders
Conceptual Flow: Spectral Decomposition & Geometry, Scree & Pareto Component Selection,
The Variance Blind Spot, Non-linear Manifold Collapse on Swiss Roll,
Deep Autoencoder Manifold Unrolling, and the TWO-NN Intrinsic Dimension Estimator.
"""

import streamlit as st
from .tiles import (
    pca_geometry,
    scree_pareto,
    variance_blind_spot,
    manifold_collapse,
    deep_autoencoder_unrolling,
    two_nn_dimension,
)


def render():
    st.title("Principal Component Analysis & Autoencoders")
    st.caption("Spectral Projections, Variance Blind Spots, Manifold Learning & Intrinsic Dimension")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌐 1. PCA Geometry & Spectral Decomposition",
        "📉 2. Scree & Pareto Diagnostic Plots",
        "⚠️ 3. The Variance Blind Spot",
        "🌀 4. Manifold Collapse on Swiss Roll",
        "🗜️ 5. Deep Autoencoder Manifold Unrolling",
        "📏 6. TWO-NN Intrinsic Dimension Estimator",
    ])

    with tab1:
        pca_geometry.render()

    with tab2:
        scree_pareto.render()

    with tab3:
        variance_blind_spot.render()

    with tab4:
        manifold_collapse.render()

    with tab5:
        deep_autoencoder_unrolling.render()

    with tab6:
        two_nn_dimension.render()
