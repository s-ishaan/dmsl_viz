"""
Module Page: Support Vector Machines (Nonlinear Decision Boundaries, Kernels & RKHS)
Senior Data Presentation Architecture: 6 Dedicated Conceptual Tabs, Each with Full-Block Visualizations.
"""

import streamlit as st
from .tiles import (
    poly_kernel,
    rbf_kernel,
    kernel_regression,
    gram_matrix_mercer,
    loss_functions_arena,
    svm_vs_lr_benchmark,
)


def render():
    st.title("Support Vector Machines: Kernels, RKHS & Regularization")
    st.caption("Non-linear Decision Boundaries, RKHS Mercer Spectra & Loss Function Landscapes")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📐 1. Polynomial Kernel Dynamics",
        "🌐 2. RBF Kernel & Localization",
        "📊 3. Localized Kernel Regression",
        "🔥 4. Gram Matrix & Mercer Spectrum",
        "🎯 5. Loss Function Arena (ESL)",
        "⚖️ 6. SVM vs KLR Benchmark",
    ])

    with tab1:
        poly_kernel.render()

    with tab2:
        rbf_kernel.render()

    with tab3:
        kernel_regression.render()

    with tab4:
        gram_matrix_mercer.render()

    with tab5:
        loss_functions_arena.render()

    with tab6:
        svm_vs_lr_benchmark.render()
