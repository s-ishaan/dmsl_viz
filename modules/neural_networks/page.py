"""
Module Page: Neural Networks & Deep Learning Foundations
Conceptual Flow: Vanilla Architecture & Parameters, Activation Functions & Vanishing Gradients,
Backpropagation Chain Rule, Optimization Arena (GD/SGD/Momentum/RMSProp/Adam),
Double Descent Phenomenon, and Autoencoders (Non-linear PCA).
"""

import streamlit as st
from .tiles import (
    vanilla_nn,
    activation_functions,
    backprop_chain_rule,
    optimizer_arena,
    double_descent,
    autoencoder_pca,
)


def render():
    st.title("Neural Networks & Deep Learning Foundations")
    st.caption("Architectures, Activation Dynamics, Optimization Landscapes, Double Descent & Autoencoders")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 1. Vanilla Architecture & Weights",
        "⚡ 2. Activation Functions & Saturation",
        "🔄 3. Backpropagation & Chain Rule",
        "🏎️ 4. Optimization Arena (Adam/SGD)",
        "🏔️ 5. Over-parametrization & Double Descent",
        "🗜️ 6. Autoencoders & Dimension Reduction",
    ])

    with tab1:
        vanilla_nn.render()

    with tab2:
        activation_functions.render()

    with tab3:
        backprop_chain_rule.render()

    with tab4:
        optimizer_arena.render()

    with tab5:
        double_descent.render()

    with tab6:
        autoencoder_pca.render()
